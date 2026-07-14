import os, logging, tqdm, pprint
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ..ACtrainer import EAEACMixin
from .EAEtrainer import AmpereEAETrainer, EAE_collate_fn, get_span_idx
from .template_generate import event_template, eve_template_generator
from .pattern import patterns, ROLE_PH_MAP

logger = logging.getLogger(__name__)


class AmpereACTrainer(EAEACMixin, AmpereEAETrainer):
    """AC trainer for Ampere (EAE-based, two-stage hierarchical inference).

    AMR graphs are generated automatically via amrlib when no pre-computed
    AMR file is supplied.  A cached stog model is kept on self._stog so that
    the (expensive) load happens at most once per trainer instance.
    """

    # ------------------------------------------------------------------ #
    # AMR helpers                                                          #
    # ------------------------------------------------------------------ #

    def _get_stog(self):
        if not hasattr(self, "_stog"):
            import amrlib
            logger.info("Loading amrlib stog model for on-the-fly AMR generation…")
            self._stog = amrlib.load_stog_model()
        return self._stog

    def _build_amr_map(self, data):
        """Parse AMR for every unique (doc_id, wnd_id) window in *data*."""
        seen = {}
        for dt in data:
            key = (dt["doc_id"], dt["wnd_id"])
            if key not in seen:
                seen[key] = " ".join(dt["tokens"])

        keys = list(seen.keys())
        sentences = [seen[k] for k in keys]

        stog = self._get_stog()
        graphs = stog.parse_sents(sentences)

        amr_map = defaultdict(dict)
        for (doc_id, wnd_id), graph in zip(keys, graphs):
            amr_map[doc_id][wnd_id] = graph or ""
        return amr_map

    # ------------------------------------------------------------------ #
    # process_data override                                                #
    # ------------------------------------------------------------------ #

    def process_data(self, data, amr_file=None):
        """Process EAE instances, auto-generating AMR when amr_file is None."""
        if amr_file is not None:
            return super().process_data(data, amr_file)

        assert self.tokenizer, "Please load model and tokenizer before processing data!"

        amr_map = self._build_amr_map(data)

        n_total = 0
        new_data = []
        for dt in data:
            n_total += 1

            _trigger = (dt["trigger"][0], dt["trigger"][1], dt["trigger"][2])
            _arguments = [(_trigger, (r[0], r[1], r[2])) for r in dt["arguments"]]
            tmpl = eve_template_generator(
                self.config.dataset, dt["tokens"], [_trigger], _arguments,
                self.config.input_style, self.config.output_style, self.vocab, False,
            )
            event_training_data = tmpl.get_training_data()
            assert len(event_training_data) == 1

            data_ = event_training_data[0]
            if len(self.tokenizer.tokenize(data_[0])) > self.config.max_length:
                continue

            pieces = [self.tokenizer.tokenize(t) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]
            pieces = [p for piece in pieces for p in piece]
            piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
            assert sum(token_lens) == len(piece_idxs)
            token_start_idxs = (
                [sum(token_lens[:i]) for i in range(len(token_lens))] + [sum(token_lens)]
            )

            new_data.append({
                "doc_id":            dt["doc_id"],
                "wnd_id":            dt["wnd_id"],
                "tokens":            dt["tokens"],
                "text":              dt["text"],
                "piece_idxs":        piece_idxs,
                "token_start_idxs":  token_start_idxs,
                "trigger":           dt["trigger"],
                "arguments":         dt["arguments"],
                "input":             data_[0],
                "target":            data_[1],
                "info":              data_[2],
                "amrgraph":          amr_map[dt["doc_id"]][dt["wnd_id"]],
            })

        logger.info(
            f"Generated {len(new_data)} Ampere EAE instances from {n_total} "
            f"EAE instances (AMR auto-generated via amrlib)"
        )
        return new_data

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        # amr_file is optional — falls back to on-the-fly generation
        internal_train_data = self.process_data(train_data, kwargs.get("train_amr"))
        self._dev_amr = kwargs.get("dev_amr")  # None → auto-generate at eval time

        param_groups = [{'params': self.model.parameters(),
                         'lr': self.config.learning_rate,
                         'weight_decay': self.config.weight_decay}]

        train_batch_num = (len(internal_train_data) // self.config.train_batch_size
                           + (len(internal_train_data) % self.config.train_batch_size != 0))
        optimizer = AdamW(params=param_groups)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_batch_num * self.config.warmup_epoch,
            num_training_steps=train_batch_num * self.config.max_epoch,
        )

        best_scores = {"ac_f": 0.0}
        best_loss = float("inf")
        best_epoch = -1

        for epoch in range(1, self.config.max_epoch + 1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")

            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc=f"Train {epoch}")
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

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def _predict_batch(self, eval_data, split="Test"):
        eval_batch_num = (len(eval_data) // self.config.eval_batch_size
                          + (len(eval_data) % self.config.eval_batch_size != 0))
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)

        predictions = []
        for batch in DataLoader(eval_data, batch_size=self.config.eval_batch_size,
                                shuffle=False, collate_fn=EAE_collate_fn):
            progress.update(1)
            pred_texts = self.model.predict(
                batch, num_beams=self.config.beam_size, max_length=self.config.max_output_length,
            )
            for doc_id, wnd_id, tokens, text, piece_idxs, token_start_idxs, trigger, pred_text in zip(
                    batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, batch.batch_text,
                    batch.batch_piece_idxs, batch.batch_token_start_idxs,
                    batch.batch_trigger, pred_texts):

                tmpl = event_template(
                    trigger[2], patterns[self.config.dataset][trigger[2]],
                    self.config.input_style, self.config.output_style, tokens,
                    ROLE_PH_MAP[self.config.dataset],
                )
                pred_objects = tmpl.decode(pred_text)

                pred_arguments = []
                for span, role_type, _ in pred_objects:
                    sid, eid = get_span_idx(
                        piece_idxs, token_start_idxs, span, self.tokenizer,
                        trigger_span=trigger[:2],
                    )
                    if sid == -1:
                        continue
                    pred_arguments.append((sid, eid, role_type, span))

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

    def predictEAE(self, data, **kwargs):
        """EAE prediction; injects stored AMR path when none provided."""
        if "test_amr" not in kwargs:
            kwargs["test_amr"] = getattr(self, "_current_amr", getattr(self, "_dev_amr", None))
        return self.internal_predict(data, **kwargs)

    def internal_predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        amr_file = kwargs.get("test_amr", getattr(self, "_dev_amr", None))
        internal_data = self.process_data(data, amr_file)  # amr_file may be None → auto-gen
        return self._predict_batch(internal_data, split="Test")

    def _ac_dev_eval(self, dev_data):
        """Dev eval with AMR injection for the hierarchical AC pipeline."""
        self._current_amr = self._dev_amr
        try:
            return super()._ac_dev_eval(dev_data)
        finally:
            self._current_amr = None
