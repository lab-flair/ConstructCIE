import os, copy, logging, tqdm, pprint
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ..ACtrainer import E2EACMixin
from .E2Etrainer import DyGIEppE2ETrainer
from .data import IEDataset
from .util import generate_vocabs

logger = logging.getLogger(__name__)


class DyGIEppACTrainer(E2EACMixin, DyGIEppE2ETrainer):
    """DyGIEpp trainer for the AC (Accident Classification) task.

    Training is identical to E2E — the full two-level event hierarchy
    (root event + sub-events) is trained jointly on the complete document.

    Inference uses two-stage hierarchical prediction via predict_ac_hierarchical
    from utils.py:
      Stage 1 — predict() on the full acc_txt → main factor entity spans
      Stage 2 — predict() on each truncated factor span → sub-factors,
                 positions shifted back to original token space

    The predict() override converts DyGIEpp's internal tuple-based events
    format into the standard IE format (entity_mentions + event_mentions)
    expected by predict_ac_hierarchical.
    """

    def train(self, train_data, dev_data, **kwargs):
        self.load_tokenizer_()

        train_set = IEDataset(train_data, self.tokenizer, self.config, max_length=self.config.max_length, test=False)
        dev_set   = IEDataset(dev_data,   self.tokenizer, self.config, max_length=self.config.max_length, test=False)
        self.vocabs = generate_vocabs([train_set, dev_set])

        train_set.numberize(self.vocabs)
        dev_set.numberize(self.vocabs)

        self.load_model_()

        batch_num = len(train_set) // self.config.batch_size + (len(train_set) % self.config.batch_size != 0)

        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if n.startswith('bert')],
                'lr': self.config.bert_learning_rate, 'weight_decay': self.config.bert_weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not n.startswith('bert')],
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
            },
        ]
        optimizer = AdamW(params=param_groups)
        schedule = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=batch_num * self.config.warmup_epoch,
                                                   num_training_steps=batch_num * self.config.max_epoch)

        best_scores = {"ac_f": 0.0}
        best_loss = float("inf")
        best_epoch  = -1

        logger.info('================Start Training================')
        for epoch in range(self.config.max_epoch):
            logger.info('Epoch: {}'.format(epoch))
            progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
            optimizer.zero_grad()
            cummulate_loss = 0.
            for batch_idx, batch in enumerate(DataLoader(
                    train_set, batch_size=self.config.batch_size // self.config.accumulate_step,
                    shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
                loss = self.model(batch)
                loss = loss * (1 / self.config.accumulate_step)
                cummulate_loss += loss
                loss.backward()
                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad()

            progress.close()
            avg_loss = float((cummulate_loss / batch_idx).data)
            logger.info({"average training loss": avg_loss})

            dev_f = self._ac_dev_eval(dev_data)
            if best_epoch < 0 or dev_f > best_scores["ac_f"] or (dev_f == 0.0 and avg_loss < best_loss):
                best_scores["ac_f"] = dev_f
                best_loss = avg_loss
                logger.info('Saving best model')
                state = dict(model=self.model.state_dict(), vocabs=self.vocabs, type_set=self.type_set)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                state = dict(tokenizer=self.tokenizer)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.tokenizer"))
                best_epoch = epoch
            logger.info(pprint.pformat({"epoch": epoch, "ac_overall_f1": dev_f}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))

    # ------------------------------------------------------------------ #
    # Prediction helpers
    # ------------------------------------------------------------------ #

    def _to_ie_format(self, pred):
        """Convert DyGIEpp event-tuple output to standard IE dict format."""
        entity_mentions = []
        event_mentions  = []
        seen_spans      = set()

        for event in pred.get("events", []):
            trig_start, trig_end, event_type = event["trigger"]

            ie_args = []
            for arg_start, arg_end, role in event.get("arguments", []):
                ie_args.append({"role": role, "start": arg_start, "end": arg_end})
                span_key = (arg_start, arg_end, role)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entity_mentions.append({"entity_type": role, "start": arg_start, "end": arg_end})

            event_mentions.append({
                "event_type": event_type,
                "trigger":    {"start": trig_start, "end": trig_end},
                "arguments":  ie_args,
            })

        return {
            "doc_id":          pred["doc_id"],
            "wnd_id":          pred["wnd_id"],
            "tokens":          pred["tokens"],
            "entity_mentions": entity_mentions,
            "event_mentions":  event_mentions,
        }

    def predict(self, data, **kwargs):
        """E2E prediction returning standard IE format (entity_mentions + event_mentions)."""
        raw_preds = self.internal_predict(data, **kwargs)
        return [self._to_ie_format(p) for p in raw_preds]

    # predictAC inherited from E2EACMixin
