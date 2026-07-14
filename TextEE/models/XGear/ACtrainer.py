import os, logging, tqdm, pprint
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BartTokenizer, AutoTokenizer, get_linear_schedule_with_warmup

from ..ACtrainer import EAEACMixin
from .ACmodel import XGearACModel
from .EAEtrainer import (
    XGearEAETrainer,
    EAE_collate_fn,
    get_arg_objects,
    get_span_idx,
    SEP_STR,
    TRIGGER_SEP_STR,
    TEMPLATE_SEP_STR,
    NONE_STR,
    AND_STR,
)
from .pattern import patterns

logger = logging.getLogger(__name__)


class XGearACTrainer(EAEACMixin, XGearEAETrainer):
    """Triggerless AC trainer for XGear.

    Conditions generation on the event-type name instead of a trigger span,
    and removes the trigger-proximity bias when mapping generated spans back
    to token indices. Synthetic trigger from convert_AC_to_EAE is preserved
    in instance metadata for downstream evaluation only.
    """

    def load_model(self, checkpoint=None):
        """Mirror of XGearEAETrainer.load_model but instantiates XGearACModel
        so AC training uses the [None]-downweighted loss.
        """
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            state = torch.load(os.path.join(checkpoint, "best_model.state"),
                               map_location=f'cuda:{self.config.gpu_device}',
                               weights_only=False)
            self.tokenizer = state["tokenizer"]
            self.type_set = state["type_set"]
            self.model = XGearACModel(self.config, self.tokenizer, self.type_set)
            # strict=False because the new token_weights buffer won't exist in
            # checkpoints saved before this change.
            self.model.load_state_dict(state['model'], strict=False)
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if os.path.basename(str(self.config.pretrained_model_name)).startswith('facebook/bart-'):
                self.tokenizer = BartTokenizer.from_pretrained(
                    self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.pretrained_model_name, cache_dir=self.config.cache_dir, use_fast=False)

            special_tokens = [SEP_STR, TRIGGER_SEP_STR, TEMPLATE_SEP_STR, NONE_STR, AND_STR]
            role_set = set([r for e in patterns[self.config.dataset]
                            for r in patterns[self.config.dataset][e]])
            for r in sorted(role_set):
                special_tokens.append(f"<--{r}-->")
                special_tokens.append(f"</--{r}-->")
            logger.info(f"Add tokens {special_tokens}")
            self.tokenizer.add_tokens(special_tokens)
            self.model = XGearACModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)

        self.generate_vocab()

    @staticmethod
    def _build_prompt_template(role_candidates):
        """Build the [None]-padded prompt template from a (possibly partial)
        role list. Used when chunking a wide multi-role event into separate
        smaller prompts via `config.prompt_groups`.
        """
        return " ".join(
            f"<--{r}--> {NONE_STR} </--{r}-->" for r in sorted(role_candidates)
        )

    @staticmethod
    def _build_target_no_none(tokens, arguments, role_candidates):
        """AC-specific target: omit roles that have no arguments instead of
        emitting `<--Role--> [None] </--Role-->`. `role_candidates` is the
        explicit list of roles this prompt covers — a chunk subset when
        `prompt_groups` is configured, or the full pattern otherwise.
        """
        role_set = set(role_candidates)
        role_candidates = sorted(role_candidates)
        arg_list = defaultdict(list)
        for r in sorted(arguments, key=lambda x: (x[0], x[1])):
            if r[2] not in role_set:
                continue
            arg_list[r[2]].append(" ".join(tokens[r[0]:r[1]]))
        return " ".join(
            f"<--{r}--> " + f" {AND_STR} ".join(arg_list[r]) + f" </--{r}-->"
            for r in role_candidates if arg_list[r]
        )

    def _resolve_prompt_chunks(self, full_role_list):
        """Partition `full_role_list` into chunks based on `config.prompt_groups`.

        prompt_groups is a list of dicts with at least a `roles` field. For
        each group the chunk is `group["roles"] ∩ full_role_list` (so per-
        event-type filtering is automatic — a group entry that doesn't
        intersect this event_type's pattern is silently skipped). Roles in
        `full_role_list` not covered by any group go into a final fallback
        chunk so coverage is preserved.

        Returns a list of `(group_idx, sorted_role_list)` tuples. group_idx
        is -1 for the fallback chunk (and for the un-chunked case when
        prompt_groups is unset/empty), so `_predict_batch` can look up
        per-group decoding hyperparameters.
        """
        prompt_groups = list(getattr(self.config, "prompt_groups", []) or [])
        if not prompt_groups:
            return [(-1, sorted(full_role_list))]

        full_set = set(full_role_list)
        used = set()
        chunks = []
        for gi, group in enumerate(prompt_groups):
            roles = group.get("roles", []) if isinstance(group, dict) else group
            chunk = [r for r in roles if r in full_set and r not in used]
            if chunk:
                chunks.append((gi, sorted(chunk)))
                used.update(chunk)
        rest = sorted(r for r in full_role_list if r not in used)
        if rest:
            chunks.append((-1, rest))
        return chunks

    def _generate_target_list(self, tokens, trigger, arguments):
        """Generate the list of (group_idx, roles, prompt_template, target)
        tuples — one entry per prompt-group chunk this instance produces.

        Analogue of `generate_target_string` from EAEtrainer.py, but returns
        a LIST so prompt-group splitting (config.prompt_groups) can yield
        multiple training/inference items from one input. When prompt_groups
        is unset, the list has exactly one entry and behaviour matches the
        un-chunked path.
        """
        full_roles = sorted(patterns[self.config.dataset][trigger[2]])
        return [
            (gi, chunk_roles,
             self._build_prompt_template(chunk_roles),
             self._build_target_no_none(tokens, arguments, chunk_roles))
            for gi, chunk_roles in self._resolve_prompt_chunks(full_roles)
        ]

    def process_data(self, data):
        assert self.tokenizer, "Please load model and tokenizer before processing data!"

        n_total = 0
        n_drop_empty = 0
        n_drop_length = 0
        # Track which roles were lost when an instance is dropped for being
        # too long. If e.g. work_ctx shows up here disproportionately, that
        # would mean its 0 F1 is partly because its training instances are
        # silently disappearing (not being seen by the model at all).
        n_drop_role = defaultdict(int)
        n_drop_length_subwords = []
        new_data = []
        is_train = getattr(self, "_in_train", False)
        for dt in data:
            n_total += 1
            target_list = self._generate_target_list(
                dt["tokens"], dt["trigger"], dt["arguments"])
            chunked = len(target_list) > 1

            for chunk_idx, (group_idx, chunk_roles, prompt_template, target) in enumerate(target_list):
                # Triggerless: only the passage + role template; the role set in
                # the template implicitly encodes the event_type. With prompt_groups,
                # each chunk covers only a subset of roles, removing head-to-head
                # competition between hard and easy roles in the decoder.
                input_str = (
                    f"{' '.join(dt['tokens'])} {SEP_STR} {TEMPLATE_SEP_STR} {prompt_template}"
                )
                input_len = len(self.tokenizer.tokenize(input_str))
                if input_len > self.config.max_length:
                    n_drop_length += 1
                    n_drop_length_subwords.append(input_len)
                    chunk_set = set(chunk_roles)
                    for arg in dt["arguments"]:
                        if arg[2] in chunk_set:
                            n_drop_role[arg[2]] += 1
                    continue
                # Drop empty-target instances during training: with role-omission
                # targets, an empty string would teach the decoder to emit
                # immediate EOS. Covers (a) no arguments, (b) args whose roles
                # fall outside pattern_map[trigger[2]], and (c) chunks whose
                # role subset has no positive arguments.
                if is_train and not target.strip():
                    n_drop_empty += 1
                    continue

                pieces = [self.tokenizer.tokenize(t) for t in dt["tokens"]]
                token_lens = [len(p) for p in pieces]
                pieces = [p for piece in pieces for p in piece]
                piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
                assert sum(token_lens) == len(piece_idxs)
                token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]

                # Per-chunk wnd_id keeps DataLoader batching honest while
                # _orig_wnd_id lets internal_predict merge chunked predictions
                # back to one result per input instance.
                chunk_wnd_id = (f"{dt['wnd_id']}__chunk{chunk_idx}"
                                if chunked else dt["wnd_id"])
                chunk_set = set(chunk_roles)
                chunk_args = [a for a in dt["arguments"] if a[2] in chunk_set]
                new_data.append({
                    "doc_id":             dt["doc_id"],
                    "wnd_id":             chunk_wnd_id,
                    "_orig_wnd_id":       dt["wnd_id"],
                    "_prompt_group_idx":  group_idx,
                    "_chunk_roles":       tuple(chunk_roles),
                    "tokens":             dt["tokens"],
                    "text":               dt["text"],
                    "piece_idxs":         piece_idxs,
                    "token_start_idxs":   token_start_idxs,
                    "trigger":            dt["trigger"],
                    "arguments":          chunk_args,
                    "input":              input_str,
                    "target":             target,
                })

        logger.debug(f"Generate {len(new_data)} XGear AC (triggerless) instances from {n_total} EAE instances")
        if n_drop_empty:
            logger.info(f"[process_data] kept {len(new_data)}/{n_total} instances "
                        f"(dropped {n_drop_empty} for empty target during training)")
        if n_drop_length:
            avg_drop = float(np.mean(n_drop_length_subwords))
            max_drop = max(n_drop_length_subwords)
            logger.info(f"[process_data] DROPPED {n_drop_length}/{n_total} instances for "
                        f"input > max_length={self.config.max_length} "
                        f"(dropped lengths: avg={avg_drop:.0f} max={max_drop} subwords)")
            # Per-role count of arguments lost to length-drop.
            role_drop_summary = ", ".join(
                f"{r}={c}" for r, c in sorted(n_drop_role.items(), key=lambda x: -x[1])
            )
            logger.info(f"[process_data] roles lost in length-drops: {role_drop_summary}")
        return new_data

    def train(self, train_data, dev_data, **kwargs):
        self.load_model()
        self._in_train = True
        internal_train_data = self.process_data(train_data)
        self._in_train = False

        # Sanity check: every parameter must be finite right after model loading.
        for n, p in self.model.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f"Parameter {n} is non-finite immediately after load_model() — "
                    f"either the local checkpoint is corrupt or resize_token_embeddings produced NaN rows.")
        logger.info("[sanity] all model parameters finite at training start.")

        param_groups = [{'params': self.model.parameters(), 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay}]

        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        # foreach=False / fused=False: PyTorch 2.x's fused AdamW corrupts BART's tied
        # `shared.weight` (encoder.embed_tokens / decoder.embed_tokens / lm_head all alias it).
        # eps=1e-6: standard for BART fine-tuning; default 1e-8 underflows when bias-corrected
        # second-moment denominator is very small at warmup step 1.
        optimizer = AdamW(params=param_groups, eps=1e-6, foreach=False, fused=False)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_batch_num * self.config.warmup_epoch,
                                                    num_training_steps=train_batch_num * self.config.max_epoch)

        best_scores = {"ac_f": 0.0}
        best_loss = float("inf")
        best_epoch = -1

        # Precompute [And] usage on the (static) training set — log once per epoch
        # so it's anchored to each epoch boundary. If n_with_and is 0, span_mode=
        # individual produced no multi-arg roles, so [And] gets no training signal.
        and_str = f" {AND_STR} "
        n_with_and = sum(1 for d in internal_train_data if and_str in d["target"])
        n_and_total = sum(d["target"].count(and_str) for d in internal_train_data)
        and_usage_msg = (f"[And] usage: {n_with_and}/{len(internal_train_data)} train "
                         f"instances ({n_and_total} occurrences total)")

        # Per-role span-length diagnostic. For each role, count how often it appears
        # as a gold argument in train and compute span length stats (in word tokens
        # AND in tokenizer subwords — the latter is what the decoder actually has to
        # generate). Long-span roles are harder for seq2seq: if the spans avg 20+
        # subwords for one role and 3 for another, the model may struggle on the
        # former regardless of class-imbalance fixes.
        role_word_lens = defaultdict(list)
        role_subword_lens = defaultdict(list)
        for d in internal_train_data:
            tokens = d["tokens"]
            for arg in d["arguments"]:
                start, end, role = arg[0], arg[1], arg[2]
                if start >= end or end > len(tokens):
                    continue
                role_word_lens[role].append(end - start)
                span_text = " ".join(tokens[start:end])
                role_subword_lens[role].append(len(self.tokenizer.tokenize(span_text)))
        # Sort by gold count descending so the most frequent roles appear first.
        span_msg_lines = ["[Role span lengths] role: n=count words(avg/max) subwords(avg/max)"]
        for role in sorted(role_word_lens.keys(), key=lambda r: -len(role_word_lens[r])):
            ws = role_word_lens[role]
            ss = role_subword_lens[role]
            span_msg_lines.append(
                f"  {role}: n={len(ws)} words(avg={np.mean(ws):.1f} max={max(ws)}) "
                f"subwords(avg={np.mean(ss):.1f} max={max(ss)})"
            )
        span_lengths_msg = "\n".join(span_msg_lines)
        logger.info(span_lengths_msg)

        # Curriculum / bootcamp schedule. Each entry is a dict with at minimum
        # `epoch` (number of training epochs to spend on this stage) and
        # `roles` (the role set this stage should focus on). Stages run
        # sequentially; within a stage, only training chunks whose role
        # subset intersects `roles` are kept. After all stages are exhausted,
        # the remaining epochs train on the full unfiltered dataset (Stage
        # N+1: integration). Each entry can also carry per-bootcamp
        # inference constraints applied during that stage's dev eval:
        #   - "ban_none": bool — when true, ban [None] from generation.
        #   - "min_new_tokens": int — min number of generated tokens before
        #     EOS is allowed (forces the decoder past trivial empty output).
        bootcamp_cfg = list(getattr(self.config, "train_group", []) or [])
        def active_bootcamp_for_epoch(ep):
            """Return (entry_dict_or_None, stage_idx_or_None) for this epoch."""
            cumulative = 0
            for idx, entry in enumerate(bootcamp_cfg):
                if not isinstance(entry, dict):
                    continue
                budget = int(entry.get("epoch", 0) or 0)
                if budget <= 0:
                    continue
                if ep <= cumulative + budget:
                    return entry, idx
                cumulative += budget
            return None, None  # past all stages → integration phase

        for epoch in range(1, self.config.max_epoch + 1):
            active_entry, stage_idx = active_bootcamp_for_epoch(epoch)
            if active_entry is not None:
                focus_roles = set(active_entry.get("roles", []) or [])
                epoch_data = [d for d in internal_train_data
                              if any(r in focus_roles for r in d.get("_chunk_roles", ()))]
                stage_label = f"bootcamp-{stage_idx} roles={sorted(focus_roles)}"
            else:
                focus_roles = set()
                epoch_data = internal_train_data
                stage_label = "integration (no filter)"
            epoch_batch_num = (len(epoch_data) + (self.config.train_batch_size - 1)) // self.config.train_batch_size

            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}  [{stage_label}]  instances={len(epoch_data)}/{len(internal_train_data)}")
            logger.info(and_usage_msg)

            progress = tqdm.tqdm(total=epoch_batch_num, ncols=100, desc='Train {}'.format(epoch))
            self.model.train()
            optimizer.zero_grad()
            cummulate_loss = []
            nan_skips = 0
            for batch_idx, batch in enumerate(DataLoader(
                    epoch_data,
                    batch_size=self.config.train_batch_size // self.config.accumulate_step,
                    shuffle=True, drop_last=False, collate_fn=EAE_collate_fn)):

                loss = self.model(batch)
                # Guard against NaN/Inf — skip the batch entirely so it can't poison weights.
                if not torch.isfinite(loss):
                    if nan_skips == 0:
                        # One-shot diagnostic on the first NaN to identify root cause.
                        logger.warning(f"[NaN-loss] batch_idx={batch_idx}, loss={loss.item()}")
                        logger.warning(f"[NaN-loss] batch sizes: input={[len(s) for s in batch.batch_input]}, target={[len(s) for s in batch.batch_target]}")
                        logger.warning(f"[NaN-loss] sample input  : {batch.batch_input[0][:400]}")
                        logger.warning(f"[NaN-loss] sample target : {batch.batch_target[0][:400]}")
                        # Check if the model's weights are already NaN.
                        for n, p in self.model.named_parameters():
                            if not torch.isfinite(p).all():
                                logger.warning(f"[NaN-loss] non-finite param: {n}  (this is the root cause; the previous step poisoned weights)")
                                break
                        else:
                            logger.warning("[NaN-loss] all weights are finite — NaN comes from forward pass on this specific input")
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
                    # Compute grad norm; clip_grad_norm_ returns the pre-clip total norm.
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           self.config.grad_clipping)
                    # Reject the step if grad contains NaN/Inf — clipping doesn't fix non-finite values,
                    # and a single bad step poisons every weight permanently.
                    if not torch.isfinite(gnorm):
                        optimizer.zero_grad()
                        nan_skips += 1
                        progress.set_postfix(loss="grad-nan", nan=nan_skips)
                        continue
                    optimizer.step()
                    scheduler.step()
                    # Post-step weight check — done BEFORE zero_grad so grads are still inspectable.
                    bad_param = None
                    for n, p in self.model.named_parameters():
                        if not torch.isfinite(p).all():
                            bad_param = n
                            break
                    if bad_param is not None:
                        bad = dict(self.model.named_parameters())[bad_param]
                        if bad.dim() >= 1:
                            row_bad = ~torch.isfinite(bad).reshape(bad.shape[0], -1).all(dim=1)
                            n_bad   = int(row_bad.sum().item())
                            n_total = bad.shape[0]
                            bad_idx = row_bad.nonzero().flatten().tolist()[:30]
                            logger.error(f"[POST-STEP NaN] {bad_param}  shape={tuple(bad.shape)}")
                            logger.error(f"  non-finite rows: {n_bad}/{n_total}")
                            logger.error(f"  first bad row indices: {bad_idx}")
                            old_vocab = getattr(self.tokenizer, "vocab_size", None)
                            tok_len   = len(self.tokenizer)
                            if old_vocab is not None:
                                logger.error(f"  tokenizer.vocab_size={old_vocab}, len(tokenizer)={tok_len}")
                                if all(i >= old_vocab for i in bad_idx):
                                    logger.error("  ⇒ ALL bad rows are NEWLY-ADDED tokens — embedding init / first-step problem.")
                                elif all(i < old_vocab for i in bad_idx):
                                    logger.error("  ⇒ ALL bad rows are pretrained tokens — generic AdamW corruption.")
                                else:
                                    logger.error("  ⇒ Mix of pretrained and new — generic AdamW corruption (cascade).")
                        logger.error(f"  gnorm={gnorm:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                        # Also dump the gradient magnitudes for the new-token rows specifically.
                        if hasattr(self, "tokenizer") and bad.dim() >= 1:
                            ov = getattr(self.tokenizer, "vocab_size", 0)
                            grad = bad.grad if bad.grad is not None else None
                            if grad is not None and grad.shape[0] > ov:
                                new_rows_grad = grad[ov:].norm(dim=1)
                                old_rows_grad = grad[:ov].norm(dim=1)
                                logger.error(f"  grad norm — new rows: max={new_rows_grad.max():.4e}, mean={new_rows_grad.mean():.4e}")
                                logger.error(f"  grad norm — old rows: max={old_rows_grad.max():.4e}, mean={old_rows_grad.mean():.4e}")
                        progress.close()
                        raise RuntimeError(f"Optimizer step poisoned weights at batch {batch_idx}")
                    optimizer.zero_grad()
                    # Live running-mean loss in the tqdm bar.
                    if cummulate_loss:
                        progress.set_postfix(loss=f"{np.mean(cummulate_loss[-100:]):.4f}",
                                             gnorm=f"{gnorm:.2f}",
                                             nan=nan_skips)

            progress.close()
            avg_loss = float(np.mean(cummulate_loss)) if cummulate_loss else float("nan")
            logger.info(f"[Epoch {epoch}] avg train loss = {avg_loss:.4f}  (nan_skips={nan_skips})")

            # Reset the [span-unmatched] debug log budget for this eval. The
            # counter is shared across all per-document _predict_batch calls
            # made via predict_ac_eae_hierarchical so we get a fixed sample
            # of failures per epoch rather than per-doc spam.
            self._debug_log_remaining = int(getattr(self.config, "debug_limit", 30) or 30)

            # Per-bootcamp inference constraints — read from the active
            # train_group entry. Applied during dev evals only while this
            # stage is active. `ban_words` is a list of token strings (e.g.
            # ["[None]"]) to forbid in generation; `min_new_tokens` forces
            # the decoder past trivial empty output before EOS is allowed.
            # After all bootcamp stages exhaust, generation falls back to
            # vanilla beam search.
            self._bootcamp_constraints = None
            if active_entry is not None:
                ban_words = active_entry.get("ban_words", []) or []
                min_new = int(active_entry.get("min_new_tokens", 0) or 0)
                rep_pen = active_entry.get("repetition_penalty", None)
                constraints = {}
                if ban_words:
                    bad_words_ids = []
                    for w in ban_words:
                        wid = self.tokenizer.convert_tokens_to_ids(w)
                        if wid != self.tokenizer.unk_token_id:
                            bad_words_ids.append([wid])
                        else:
                            logger.warning(f"[bootcamp-{stage_idx}] ban_words entry "
                                           f"{w!r} not found in tokenizer; skipping")
                    if bad_words_ids:
                        constraints["bad_words_ids"] = bad_words_ids
                if min_new > 0:
                    constraints["min_new_tokens"] = min_new
                if rep_pen is not None and float(rep_pen) != 1.0:
                    constraints["repetition_penalty"] = float(rep_pen)
                if constraints:
                    self._bootcamp_constraints = constraints
                    logger.info(f"[bootcamp-{stage_idx}] inference constraints "
                                f"active for this eval: {constraints}")

            # Reset bootcamp prediction-sample log budget. While a bootcamp
            # is active and config.debug is set, the first `debug_limit`
            # UNIQUE pred_texts of the eval are dumped verbatim so we can
            # see the actual generations the constraints produced (matched
            # OR not). Identical generations across instances (common when
            # the model emits the same boilerplate) are deduplicated.
            self._bootcamp_pred_log_remaining = (
                int(getattr(self.config, "debug_limit", 30) or 30)
                if (active_entry is not None and getattr(self.config, "debug", False))
                else 0
            )
            self._bootcamp_pred_log_seen = set()

            dev_f = self._ac_dev_eval(dev_data)
            # Save-best policy:
            #   - first epoch always saves;
            #   - any strict F1 improvement saves;
            #   - on F1 tie (including both 0), prefer the lower train loss as a
            #     tiebreaker. The tie clause subsumes the older "both-zero only"
            #     guard while still preventing a regressed F1 from overwriting
            #     a positive-F1 best (observed: epoch 6 F1=4.03 lost to epoch 7
            #     F1=0 under the un-gated policy).
            should_save = (
                best_epoch < 0
                or dev_f > best_scores["ac_f"]
                or (dev_f == best_scores["ac_f"] and avg_loss < best_loss)
            )
            if should_save:
                best_scores["ac_f"] = dev_f
                best_loss = avg_loss
                logger.info("Saving best model")
                state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                best_epoch = epoch

            logger.info(pprint.pformat({"epoch": epoch, "ac_overall_f1": dev_f}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))

    @staticmethod
    def _auto_close_role_tags(text, valid_roles):
        """Append missing closing role tags to a generated text. If the model
        emitted `<--R-->` but ran out of tokens / drifted before closing,
        this appends `</--R-->` so `get_arg_objects` can still extract the
        partial span. Only closes tags whose role is valid for this event_type
        (passed in `valid_roles`); over-emitted tags from other event_types
        are left alone and will be silently dropped by get_arg_objects.
        """
        for r in valid_roles:
            open_tag = f"<--{r}-->"
            close_tag = f"</--{r}-->"
            n_open = text.count(open_tag)
            n_close = text.count(close_tag)
            if n_open > n_close:
                text = text + (" " + close_tag) * (n_open - n_close)
        return text

    def _predict_batch(self, eval_data, split="Test"):
        # Group by prompt-group so each batch shares decoding hyperparameters.
        # When prompt_groups is unset, every instance has _prompt_group_idx=-1
        # and we get a single group with default params — equivalent to the
        # un-chunked path.
        groups = defaultdict(list)
        for d in eval_data:
            groups[d.get("_prompt_group_idx", -1)].append(d)

        prompt_groups = list(getattr(self.config, "prompt_groups", []) or [])

        bs = self.config.eval_batch_size
        total_batches = sum(
            (len(g) + bs - 1) // bs for g in groups.values()
        )
        # Suppress the inner bar for per-document AC inference calls (the
        # outer `AC inference` loop in predict_ac_eae_hierarchical already
        # provides progress, and chunking makes per-doc total_batches
        # potentially > 1, defeating the old "<= 1" threshold). Heuristic:
        # if the entire eval slice fits in a few batches it's almost
        # certainly an inner call, not a full-set evaluation.
        progress = tqdm.tqdm(total=total_batches, ncols=100, desc=split,
                             disable=len(eval_data) <= bs * 8)

        predictions = []
        for gidx, gdata in groups.items():
            if 0 <= gidx < len(prompt_groups) and isinstance(prompt_groups[gidx], dict):
                group_cfg = prompt_groups[gidx]
                length_penalty = float(group_cfg.get("length_penalty", 1.0))
            else:
                length_penalty = 1.0

            # Bootcamp-time constraints (set in train() before each dev eval
            # while a bootcamp stage is active). None outside bootcamp.
            constraints = getattr(self, "_bootcamp_constraints", None) or {}

            for batch in DataLoader(gdata, batch_size=bs,
                                    shuffle=False, collate_fn=EAE_collate_fn):
                progress.update(1)
                pred_texts = self.model.predict(
                    batch,
                    num_beams=self.config.beam_size,
                    max_length=self.config.max_output_length,
                    length_penalty=length_penalty,
                    bad_words_ids=constraints.get("bad_words_ids"),
                    min_new_tokens=constraints.get("min_new_tokens"),
                    repetition_penalty=constraints.get("repetition_penalty"),
                )
                for doc_id, wnd_id, tokens, text, piece_idxs, token_start_idxs, trigger, pred_text in zip(
                        batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, batch.batch_text,
                        batch.batch_piece_idxs, batch.batch_token_start_idxs,
                        batch.batch_trigger, pred_texts):

                    # Bootcamp-time sample log: dump the first `debug_limit`
                    # UNIQUE raw pred_texts so we can see exactly what the
                    # model emits under the bootcamp's inference constraints.
                    # Gated on config.debug AND budget set in train()'s epoch
                    # loop (only non-zero during bootcamp epochs with debug on).
                    # Deduplicated on the printed snippet (not full pred_text)
                    # so generations that diverge only AFTER the snippet
                    # boundary still count as duplicates in the log.
                    if (getattr(self.config, "debug", False)
                            and getattr(self, "_bootcamp_pred_log_remaining", 0) > 0):
                        snippet = pred_text.replace("\n", " ")[:240]
                        if snippet not in getattr(self, "_bootcamp_pred_log_seen", set()):
                            logger.info(
                                f"[bootcamp-pred] event_type={trigger[2]} "
                                f"group={gidx} wnd_id={wnd_id} "
                                f"pred_text={snippet!r}")
                            self._bootcamp_pred_log_seen.add(snippet)
                            self._bootcamp_pred_log_remaining -= 1

                    # Auto-close any unclosed role tags before parsing so
                    # partial generations (open tag + content but no close)
                    # are still extractable. Only closes tags whose role is
                    # valid for this event_type's pattern.
                    valid_roles = patterns[self.config.dataset].get(trigger[2], [])
                    pred_text_closed = self._auto_close_role_tags(pred_text, valid_roles)
                    pred_objects = get_arg_objects(trigger[2], pred_text_closed, patterns[self.config.dataset])

                    # Span-mapping mode: when config.span_map is False, skip
                    # the get_span_idx grounding entirely and pass generated
                    # text through to scoring. The AC scorer is text-similarity
                    # based (ac_sim_threshold), so hallucinated tails / mild
                    # paraphrases still score above threshold without needing
                    # exact token-level matches. Sentinel (0, 1) keeps
                    # downstream `start >= end` filters from dropping the
                    # entity; the `span` text is what scoring will use.
                    span_map_enabled = bool(getattr(self.config, "span_map", True))
                    pred_arguments = []
                    for span, role_type in pred_objects:
                        if span_map_enabled:
                            # Triggerless: no trigger-proximity bias; first contiguous match wins.
                            sid, eid = get_span_idx(piece_idxs, token_start_idxs, span, self.tokenizer, trigger_span=None)
                            if sid == -1:
                                # Generated span couldn't be grounded in the input
                                # tokens (model hallucinated or paraphrased). Drop
                                # silently in normal runs; emit a sample log when
                                # config.debug is set, rate-limited by
                                # config.debug_limit (default 30 per dev eval,
                                # reset in train()'s epoch loop).
                                if (getattr(self.config, "debug", False)
                                        and getattr(self, "_debug_log_remaining", 0) > 0):
                                    snippet = span[:120].replace("\n", " ")
                                    logger.info(
                                        f"[span-unmatched] role={role_type} "
                                        f"event_type={trigger[2]} wnd_id={wnd_id} "
                                        f"span={snippet!r}")
                                    self._debug_log_remaining -= 1
                                    if self._debug_log_remaining == 0:
                                        logger.info("[span-unmatched] debug_limit reached, "
                                                    "suppressing further [span-unmatched] logs this eval")
                                continue
                        else:
                            # Span mapping disabled — keep the prediction with
                            # sentinel indices so it survives downstream
                            # `start < end` filters and gets scored via text.
                            sid, eid = 0, 1
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

    def internal_predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        internal_data = self.process_data(data)
        raw_predictions = self._predict_batch(internal_data, split="Test")
        # Build chunk_wnd_id -> orig_wnd_id mapping so chunked predictions can
        # be aggregated back to one result per input instance.
        wnd_to_orig = {d["wnd_id"]: d.get("_orig_wnd_id", d["wnd_id"]) for d in internal_data}
        merged = {}
        for pred in raw_predictions:
            orig = wnd_to_orig.get(pred["wnd_id"], pred["wnd_id"])
            if orig not in merged:
                mp = dict(pred)
                mp["wnd_id"] = orig
                mp["arguments"] = list(pred["arguments"])
                merged[orig] = mp
            else:
                # Different chunks of the same input cover disjoint role sets,
                # so concatenating their argument tuples is correct.
                merged[orig]["arguments"].extend(pred["arguments"])
        return list(merged.values())
