import os, logging, tqdm, pprint
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ..ACtrainer import EAEACMixin
from .EAEtrainer import QueryAndExtractEAETrainer, EAE_collate_fn

logger = logging.getLogger(__name__)


class QueryAndExtractACTrainer(EAEACMixin, QueryAndExtractEAETrainer):
    """AC trainer for QueryAndExtract (EAE-based, two-stage hierarchical inference)."""

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data_for_training(train_data)

        param_optimizer1 = list(self.model.earl_model.bert.named_parameters())
        param_optimizer2 = list(self.model.earl_model.linear.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_groups = [
            {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.base_model_weight_decay},
            {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay, 'lr': self.config.learning_rate},
            {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.config.learning_rate},
        ]
        ner_param_groups = [
            {
                'params': [p for n, p in self.model.ner_model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.ner_base_model_learning_rate, 'weight_decay': self.config.ner_base_model_weight_decay
            },
            {
                'params': [p for n, p in self.model.ner_model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.ner_learning_rate, 'weight_decay': self.config.ner_weight_decay
            },
        ]
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)

        ner_optimizer = AdamW(params=ner_param_groups)
        ner_scheduler = get_linear_schedule_with_warmup(ner_optimizer,
                                                        num_warmup_steps=train_batch_num * self.config.warmup_epoch,
                                                        num_training_steps=train_batch_num * self.config.max_epoch)
        optimizer = AdamW(params=param_groups, lr=self.config.base_model_learning_rate, weight_decay=0)
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
            ner_optimizer.zero_grad()
            cummulate_loss = []
            cummulate_ner_loss = []
            for batch_idx, batch in enumerate(DataLoader(
                    internal_train_data,
                    batch_size=self.config.train_batch_size // self.config.accumulate_step,
                    shuffle=True, drop_last=False, collate_fn=EAE_collate_fn)):

                loss, _, ner_loss = self.model(batch)
                loss = loss * (1 / self.config.accumulate_step)
                cummulate_loss.append(loss.item())
                loss.backward()

                ner_loss = ner_loss * (1 / self.config.accumulate_step)
                cummulate_ner_loss.append(ner_loss.item())
                ner_loss.backward()

                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.model.earl_model.parameters(), self.config.grad_clipping)
                    torch.nn.utils.clip_grad_norm_(self.model.ner_model.parameters(), self.config.ner_grad_clipping)
                    optimizer.step()
                    ner_optimizer.step()
                    scheduler.step()
                    ner_scheduler.step()
                    optimizer.zero_grad()
                    ner_optimizer.zero_grad()

            progress.close()
            avg_loss = float(np.mean(cummulate_loss))
            logger.info(f"Average training loss: {avg_loss}")
            logger.info(f"Average training ner_loss: {np.mean(cummulate_ner_loss)}")

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
        """Two-stage NER → EARL prediction on pre-structured evaluation data."""
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)

        internal_data1 = self.process_data_for_testing_ner(eval_data)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        ner_predictions = []
        for batch_idx, batch in enumerate(DataLoader(
                internal_data1, batch_size=self.config.eval_batch_size,
                shuffle=False, collate_fn=EAE_collate_fn)):
            progress.update(1)
            batch_pred_entities = self.model.ner_model.predict(batch)
            for doc_id, wnd_id, trigger, pred_entities in zip(
                    batch.batch_doc_id, batch.batch_wnd_id, batch.batch_trigger, batch_pred_entities):
                ner_predictions.append({
                    "doc_id":    doc_id,
                    "wnd_id":    wnd_id,
                    "trigger":   trigger,
                    "entities":  pred_entities,
                })
        progress.close()

        internal_data2 = self.process_data_for_testing(eval_data, ner_predictions)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(
                internal_data2, batch_size=self.config.eval_batch_size,
                shuffle=False, collate_fn=EAE_collate_fn)):
            progress.update(1)
            batch_pred_arguments = self.model.earl_model.predict(batch)
            for doc_id, wnd_id, trigger, pred_arguments in zip(
                    batch.batch_doc_id, batch.batch_wnd_id, batch.batch_trigger, batch_pred_arguments):
                predictions.append({
                    "doc_id":    doc_id,
                    "wnd_id":    wnd_id,
                    "trigger":   trigger,
                    "arguments": [p for p in pred_arguments if p[2] is not None],
                })
        progress.close()
        return predictions

    def internal_predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        return self._predict_batch(data, split="Test")
