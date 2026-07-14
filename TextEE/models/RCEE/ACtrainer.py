import os, logging, tqdm, pprint
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ..ACtrainer import EAEACMixin
from .EAEtrainer import RCEEEAETrainer, EAE_collate_fn

logger = logging.getLogger(__name__)


class RCEEACTrainer(EAEACMixin, RCEEEAETrainer):
    """AC trainer for RCEE (EAE-based, two-stage hierarchical inference)."""

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data(train_data)

        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.base_model_learning_rate, 'weight_decay': self.config.base_model_weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
            },
        ]

        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        optimizer = AdamW(params=param_groups)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_batch_num * self.config.warmup_epoch,
                                                    num_training_steps=train_batch_num * self.config.max_epoch)

        best_scores = {"ac_f": 0.0}
        best_loss = float("inf")
        best_epoch = -1

        for epoch in range(1, self.config.max_epoch + 1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")

            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train {}'.format(epoch))
            self.model.train()
            optimizer.zero_grad()
            cummulate_loss = []
            for batch_idx, batch in enumerate(DataLoader(
                    internal_train_data,
                    batch_size=self.config.train_batch_size // self.config.accumulate_step,
                    shuffle=True, drop_last=False, collate_fn=EAE_collate_fn)):

                loss = self.model(batch)
                loss = loss * (1 / self.config.accumulate_step)
                cummulate_loss.append(loss.item())
                loss.backward()

                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            progress.close()
            avg_loss = float(np.mean(cummulate_loss))
            logger.info(f"Average training loss: {avg_loss}")

            dev_f = self._ac_dev_eval(dev_data)
            if best_epoch < 0 or dev_f > best_scores["ac_f"] or (dev_f == 0.0 and avg_loss < best_loss):
                best_scores["ac_f"] = dev_f
                best_loss = avg_loss
                logger.info("Saving best model")
                state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                best_epoch = epoch

            logger.info(pprint.pformat({"epoch": epoch, "ac_overall_f1": dev_f}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))

    def _predict_batch(self, eval_data, split="Test"):
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)

        predictions = []
        for batch_idx, batch in enumerate(DataLoader(
                eval_data, batch_size=self.config.eval_batch_size,
                shuffle=False, collate_fn=EAE_collate_fn)):
            progress.update(1)
            batch_pred_arguments = self.model.predict(batch)
            for doc_id, wnd_id, tokens, text, trigger, pred_arguments in zip(
                    batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, batch.batch_text,
                    batch.batch_trigger, batch_pred_arguments):
                predictions.append({
                    "doc_id":    doc_id,
                    "wnd_id":    wnd_id,
                    "tokens":    tokens,
                    "text":      text,
                    "trigger":   trigger,
                    "arguments": pred_arguments,
                })
        progress.close()
        return predictions

    def internal_predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        internal_data = self.process_data(data)
        return self._predict_batch(internal_data, split="Test")
