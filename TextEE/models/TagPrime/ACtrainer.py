import os, logging, tqdm
from collections import namedtuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup

from ..ACtrainer import EAEACMixin
from .ACmodel import TagPrimeACModel
from .EAEtrainer import TagPrimeEAETrainer, EAEBatch

logger = logging.getLogger(__name__)


# AC-specific batch: extends EAEBatch with `batch_oi_arguments`, the raw
# object_involved spans that bypass overlap removal. The dedicated oi BIO head
# (oi_label_ffn) in TagPrimeACModel consumes this so oi training signal survives
# even when oi spans are nested inside a longer working_circumstances span that
# would otherwise claim them in single-head BIO.
ACBatch_fields = list(EAEBatch._fields) + ['batch_oi_arguments']
ACBatch = namedtuple('ACBatch', field_names=ACBatch_fields,
                     defaults=[None] * len(ACBatch_fields))


def AC_collate_fn(batch):
    return ACBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_tokens=[instance["tokens"] for instance in batch],
        batch_pieces=[instance["pieces"] for instance in batch],
        batch_token_lens=[instance["token_lens"] for instance in batch],
        batch_token_num=[instance["token_num"] for instance in batch],
        batch_text=[instance["text"] for instance in batch],
        batch_trigger=[instance["trigger"] for instance in batch],
        batch_arguments=[instance["arguments"] for instance in batch],
        batch_oi_arguments=[instance.get("oi_arguments", []) for instance in batch],
    )


class TagPrimeACTrainer(EAEACMixin, TagPrimeEAETrainer):
    """AC trainer for TagPrime (EAE-based, two-stage hierarchical inference).

    Covers both TagPrime-C and TagPrime-CR configurations — differentiated via
    the model config JSON (e.g. global_data/configs/TagPrime/model.json).
    """

    def process_data(self, data):
        """Mirrors TagPrimeEAETrainer.process_data, but wraps each token in a
        single-element list when calling tokenizer.tokenize(..., is_split_into_words=True).

        The shared parent passes a bare string with is_split_into_words=True,
        which newer `transformers` versions reject (the kwarg requires a
        pre-tokenized List[str], not a single str). Wrapping in [t] is
        accepted by both slow and fast tokenizers and produces the same
        BPE-prefix behavior when add_prefix_space=True.
        """
        assert self.tokenizer, "Please load model and tokenizer before processing data!"

        # Demoted to debug: this method is called once per doc inside
        # predict_ac_eae_hierarchical, which would otherwise produce 53× spam.
        logger.debug("Removing overlapping arguments and over-length examples")

        n_total = 0
        new_data = []
        for dt in data:
            n_total += 1

            if len(dt["tokens"]) > self.config.max_length:
                continue

            # Capture raw object_involved args BEFORE overlap removal so the
            # dedicated oi head (in TagPrimeACModel) can train on them even
            # when they overlap with a longer working_circumstances span that
            # would otherwise erase them in single-head BIO.
            oi_arguments = [a for a in dt["arguments"] if a[2] == "object_involved"]

            no_overlap_flag = np.ones((len(dt["tokens"]),), dtype=bool)
            new_arguments = []
            for argument in sorted(dt["arguments"]):
                start, end = argument[0], argument[1]
                if np.all(no_overlap_flag[start:end]):
                    new_arguments.append(argument)
                    no_overlap_flag[start:end] = False

            pieces = [self.tokenizer.tokenize([t], is_split_into_words=True) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]

            new_data.append({
                "doc_id":       dt["doc_id"],
                "wnd_id":       dt["wnd_id"],
                "tokens":       dt["tokens"],
                "pieces":       [p for w in pieces for p in w],
                "token_lens":   token_lens,
                "token_num":    len(dt["tokens"]),
                "text":         dt["text"],
                "trigger":      dt["trigger"],
                "arguments":    new_arguments,
                "oi_arguments": oi_arguments,
            })

        logger.debug(
            f"There are {len(new_data)}/{n_total} EAE instances after removing "
            f"overlapping arguments and over-length examples"
        )
        return new_data

    def _filter_cr_zero_candidate_instances(self, instances, where=""):
        """Drop CR-mode instances whose trigger type has zero extraction
        candidates after `_cr_valid_roles`.

        After pattern regeneration, two Level-2 event types end up with K=0
        forwards in CR mode: `working_circumstances` (its only sub_factor
        construction_trade is a cls factor and is filtered out of patterns)
        and `object_involved` (no sub_factors in the schema). Random batching
        can produce all-K=0 batches, which crash the model's `process_data`
        on `max(enc_idxs)` over an empty sequence. Filtering upstream is
        cleaner than adding model-level guards.

        No-op in C mode and when the model isn't built yet (pre-load_model
        callers shouldn't hit this path; defensive `getattr` keeps it safe).
        """
        if getattr(self.config, "priming_type", None) != "condition+relation":
            return instances
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "_cr_valid_roles"):
            return instances
        kept = [dt for dt in instances if model._cr_valid_roles(dt["trigger"][2])]
        dropped = len(instances) - len(kept)
        if dropped > 0:
            tag = f" [{where}]" if where else ""
            logger.info(
                f"[CR filter]{tag} dropped {dropped}/{len(instances)} "
                f"instances with no extraction candidates "
                f"(trigger types whose patterns entry is empty after cls filtering)."
            )
        return kept

    def load_model(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            state = torch.load(os.path.join(checkpoint, "best_model.state"),
                               map_location=f'cuda:{self.config.gpu_device}',
                               weights_only=False)
            self.tokenizer = state["tokenizer"]
            self.type_set = state["type_set"]
            self.model = TagPrimeACModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            # EAEtrainer.process_data calls
            # `tokenizer.tokenize(token, is_split_into_words=True)` per single
            # token, which the fast tokenizer rejects. Some `transformers`
            # builds also ignore `use_fast=False` via AutoTokenizer, so for
            # RoBERTa we import the slow class directly to guarantee it.
            if os.path.basename(str(self.config.pretrained_model_name)).startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(
                    self.config.pretrained_model_name,
                    do_lower_case=False,
                    add_prefix_space=True,
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.pretrained_model_name,
                    do_lower_case=False,
                    use_fast=False,
                )
            logger.info(f"Tokenizer class: {type(self.tokenizer).__name__}")
            self.model = TagPrimeACModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        internal_train_data = self.process_data(train_data)
        internal_train_data = self._filter_cr_zero_candidate_instances(
            internal_train_data, where="train")

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

        # Save-best + early-stop state lives on the mixin so all EAE-AC
        # trainers (EEQA, RCEE, BartGen, …) can reuse the same logic.
        best_state = self.make_best_state()

        for epoch in range(1, self.config.max_epoch + 1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")

            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train {}'.format(epoch))
            self.model.train()
            optimizer.zero_grad()
            cummulate_loss = []
            nan_skips = 0
            for batch_idx, batch in enumerate(DataLoader(
                    internal_train_data,
                    batch_size=self.config.train_batch_size // self.config.accumulate_step,
                    shuffle=True, drop_last=False, collate_fn=AC_collate_fn)):

                loss = self.model(batch)
                # Guard against NaN/Inf — skip the batch entirely so it can't poison weights.
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    nan_skips += 1
                    if (batch_idx + 1) % self.config.accumulate_step == 0:
                        progress.update(1)
                    continue
                loss = loss * (1 / self.config.accumulate_step)
                cummulate_loss.append(loss.item())
                loss.backward()

                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    # clip_grad_norm_ returns the pre-clip total grad norm — surface it in the bar.
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           self.config.grad_clipping)
                    if not torch.isfinite(gnorm):
                        optimizer.zero_grad()
                        nan_skips += 1
                        progress.set_postfix(loss="grad-nan", nan=nan_skips)
                        continue
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if cummulate_loss:
                        progress.set_postfix(
                            loss=f"{np.mean(cummulate_loss[-100:]):.4f}",
                            gnorm=f"{gnorm:.2f}",
                            nan=nan_skips,
                        )

            progress.close()
            avg_loss = float(np.mean(cummulate_loss)) if cummulate_loss else float("nan")
            logger.info(f"[Epoch {epoch}] avg train loss = {avg_loss:.4f}  (nan_skips={nan_skips})")

            dev_f = self._ac_dev_eval(dev_data)
            should_save, should_stop = self._save_best_and_check_early_stop(
                dev_f, avg_loss, epoch, best_state)
            if should_save:
                logger.info("Saving best model")
                state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
            if should_stop:
                break

    def _predict_batch(self, eval_data, split="Test", show_progress=None):
        # No CR-filter here on purpose: predict_ac_eae_hierarchical zips
        # `level2_types` with `preds_level2` 1:1, so filtering would
        # misalign event_type → prediction. Batches with mixed K=0/K>0
        # instances are handled naturally (the K=0 ones simply contribute
        # no forwards but still occupy a slot in the predict_batch output).
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        # Default: show progress only when there's more than one batch.
        # During AC hierarchical inference each doc's 6 EAE instances fit in
        # one batch, so the per-doc 1/1 bar is pure noise — suppress it.
        if show_progress is None:
            show_progress = eval_batch_num > 1
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split, disable=not show_progress)

        predictions = []
        for batch_idx, batch in enumerate(DataLoader(
                eval_data, batch_size=self.config.eval_batch_size,
                shuffle=False, collate_fn=AC_collate_fn)):
            progress.update(1)
            pred_out = self.model.predict(batch)
            # `TagPrimeACModel.predict` returns (arguments, cls_predictions);
            # older variants may still return just arguments.
            if isinstance(pred_out, tuple) and len(pred_out) == 2:
                batch_pred_arguments, batch_cls_preds = pred_out
            else:
                batch_pred_arguments = pred_out
                batch_cls_preds = [{} for _ in batch_pred_arguments]
            for doc_id, wnd_id, tokens, text, trigger, pred_arguments, cls_preds in zip(
                    batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, batch.batch_text,
                    batch.batch_trigger, batch_pred_arguments, batch_cls_preds):
                predictions.append({
                    "doc_id":          doc_id,
                    "wnd_id":          wnd_id,
                    "tokens":          tokens,
                    "text":            text,
                    "trigger":         trigger,
                    "arguments":       pred_arguments,
                    # `{factor: label}` predicted by the shared cls head for
                    # this instance. Hierarchical predictor merges these into
                    # the final per-doc IE structure.
                    "cls_predictions": cls_preds,
                })
        progress.close()
        return predictions

    def internal_predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        internal_data = self.process_data(data)
        return self._predict_batch(internal_data, split="Test")


# TagPrimeE2ACTrainer (used by E2AC* / E2ACH* tasks) lives in
# E2ACtrainer.py — it inherits from TagPrimeACTrainer and only overrides
# predictAC to use the cls-probe pipeline. Imported via the package
# __init__ for the trainer registry lookup.
