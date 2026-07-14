import os, copy, logging, tqdm, pprint
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ..ACtrainer import E2EACMixin
from .E2Etrainer import AMRIEE2ETrainer
from .data import IEDataset
from .util import generate_vocabs, load_valid_patterns

logger = logging.getLogger(__name__)


class AMRIEACTrainer(E2EACMixin, AMRIEE2ETrainer):
    """AC trainer for AMRIE (E2E joint extraction, two-stage hierarchical inference).

    AMRIE requires pre-computed AMR graphs for both training and inference.
    During training, graphs are loaded from self.config.processed_train_amr and
    self.config.processed_dev_amr. During inference, predict() falls back to
    processed_dev_amr when no explicit AMR path is provided.
    """

    def train(self, train_data, dev_data, **kwargs):
        logger.info("Loading graphs")
        org_train_graphs, train_align, train_exist = torch.load(self.config.processed_train_amr, weights_only=False)
        org_dev_graphs, dev_align, dev_exist = torch.load(self.config.processed_dev_amr, weights_only=False)

        train_graphs, dev_graphs = [], []
        for g in org_train_graphs:
            train_graphs.append(g.to(self.config.gpu_device))
        for g in org_dev_graphs:
            dev_graphs.append(g.to(self.config.gpu_device))

        self.load_tokenizer_()

        train_set = IEDataset(train_data, self.tokenizer, train_graphs, train_align, train_exist, gpu=True,
                              max_length=self.config.max_length,
                              relation_mask_self=self.config.relation_mask_self,
                              relation_directional=self.config.relation_directional,
                              symmetric_relations=self.config.symmetric_relations)

        dev_set = IEDataset(dev_data, self.tokenizer, dev_graphs, dev_align, dev_exist, gpu=True,
                            max_length=self.config.max_length,
                            relation_mask_self=self.config.relation_mask_self,
                            relation_directional=self.config.relation_directional,
                            symmetric_relations=self.config.symmetric_relations)

        self.vocabs = generate_vocabs([train_set, dev_set])
        train_set.numberize(self.tokenizer, self.vocabs)
        dev_set.numberize(self.tokenizer, self.vocabs)

        self.load_model_()

        batch_num = len(train_set) // self.config.batch_size + (len(train_set) % self.config.batch_size != 0)

        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if n.startswith('bert')],
                'lr': self.config.bert_learning_rate, 'weight_decay': self.config.bert_weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not n.startswith('bert')
                           and 'crf' not in n and 'global_feature' not in n],
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not n.startswith('bert')
                           and ('crf' in n or 'global_feature' in n)],
                'lr': self.config.learning_rate, 'weight_decay': 0
            },
        ]
        optimizer = AdamW(params=param_groups)
        schedule = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=batch_num * self.config.warmup_epoch,
                                                   num_training_steps=batch_num * self.config.max_epoch)

        best_scores = {"ac_f": 0.0}
        best_loss = float("inf")
        best_epoch = -1

        logger.info('================Start Training================')
        for epoch in range(self.config.max_epoch):
            logger.info('Epoch: {}'.format(epoch))
            progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
            optimizer.zero_grad()
            cummulate_loss = 0.
            for batch_idx, batch in enumerate(DataLoader(
                    train_set, batch_size=self.config.batch_size // self.config.accumulate_step,
                    shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
                loss = self.model(batch, epoch)
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
                state = dict(model=self.model.state_dict(), vocabs=self.vocabs, type_set=self.type_set, valid_patterns=self.valid_patterns)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                state = dict(tokenizer=self.tokenizer)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.tokenizer"))
                best_epoch = epoch

            logger.info(pprint.pformat({"epoch": epoch, "ac_overall_f1": dev_f}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))

    def predict(self, data, **kwargs):
        """E2E predict returning standard IE format.

        Falls back to self.config.processed_dev_amr when no AMR path provided,
        which is correct for dev eval during the AC hierarchical pipeline.
        """
        if "processed_test_amr" not in kwargs:
            kwargs["processed_test_amr"] = self.config.processed_dev_amr
        raw_preds = self.internal_predict(data, **kwargs)
        return [self._to_ie_format(p) for p in raw_preds]

    # predictAC inherited from E2EACMixin
