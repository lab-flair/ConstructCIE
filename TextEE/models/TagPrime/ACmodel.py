import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BertConfig, RobertaConfig, XLMRobertaConfig,
                          BertModel, RobertaModel, XLMRobertaModel)

from .EAEmodel import TagPrimeEAEModel, Linears, CRF
from .pattern import event_type_tags, role_type_tags, patterns
from ..ACtrainer import parse_group_weight

logger = logging.getLogger(__name__)


class TagPrimeACModel(TagPrimeEAEModel):
    """Triggerless TagPrime model variant for the AC (Accident Classification) task.

    The AC task has no token-level trigger anchor; trigger-position embedding
    and event-type embedding are not used here regardless of
    `use_trigger_feature` / `use_type_feature`.

    Prompt formats (no trigger_text slot — AC is triggerless):
        condition (TagPrime-C):
            single inference per (instance, trigger).
            <s> {input} </s> {event_type_label} </s>
            event_type_label = event_type_tags[dataset][trigger[2]] — at acc_type
                level this is the verbalized event type (e.g. "Fall");
                at main_factor level it is the verbalized FactorA
                (e.g. "Working Circumstances").
            Targets: role-specific BIO (B-task, I-object, …).

        condition+relation (TagPrime-CR):
            per-candidate iteration over `patterns[dataset][trigger[2]]` —
            one inference per candidate role, mirroring EAE-CR but triggerless.
            <s> {input} </s> {event_type_label} </s> {role_type_label} </s>
            role_type_label = role_type_tags[dataset][candidate]
                (FactorA at acc_type level, SubfactorB at main_factor level).
            Targets: unified B-Pred / I-Pred. Predictions are decomposed back
            to role types at predict time, identical to EAEmodel.predict's branch.
    """

    def __init__(self, config, tokenizer, type_set):
        # Skip parent __init__ — it builds trigger/type-feature modules and
        # sizes the role FFN to include those features. Triggerless AC needs
        # neither, so we set up the base encoder and tagging head directly.
        nn.Module.__init__(self)
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        # Build classification-vocab metadata BEFORE generate_tagging_vocab so
        # the BIO vocab can exclude classification factors. _build_cls_vocab
        # reads `config.key_map.task.classification.classes` and filters
        # `accident_type` when the task is AC (acc_type is given as input).
        self._build_cls_vocab()
        self.generate_tagging_vocab()

        name = os.path.basename(str(self.config.pretrained_model_name))
        if name.startswith('bert-'):
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.base_config = BertConfig.from_pretrained(
                self.config.pretrained_model_name)
            self.base_model = BertModel.from_pretrained(
                self.config.pretrained_model_name,
                output_hidden_states=True)
        elif name.startswith('roberta-'):
            self.base_config = RobertaConfig.from_pretrained(
                self.config.pretrained_model_name)
            self.base_model = RobertaModel.from_pretrained(
                self.config.pretrained_model_name,
                output_hidden_states=True)
        elif name.startswith('xlm-'):
            self.base_config = XLMRobertaConfig.from_pretrained(
                self.config.pretrained_model_name)
            self.base_model = XLMRobertaModel.from_pretrained(
                self.config.pretrained_model_name,
                output_hidden_states=True)
        else:
            raise ValueError(
                f"pretrained_model_name {self.config.pretrained_model_name!r} is not supported.")

        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.base_model_dim = self.base_config.hidden_size
        self.base_model_dropout = nn.Dropout(p=self.config.base_model_dropout)

        self.dropout = nn.Dropout(p=self.config.linear_dropout)
        self.role_label_ffn = Linears(
            [self.base_model_dim, self.config.linear_hidden_num, len(self.label_stoi["role"])],
            dropout_prob=self.config.linear_dropout,
            bias=self.config.linear_bias,
            activation=self.config.linear_activation,
        )
        if self.config.use_crf:
            self.role_crf = CRF(self.label_stoi["role"], bioes=False)

        # Dedicated parallel BIO head for `object_involved`. Trained on the raw
        # oi gold spans (bypassing the main head's overlap-removal pass) so a
        # short oi token nested inside a long working_circumstances narrative
        # still produces a B-object_involved training signal — which single-head
        # BIO can't represent (one label per token) and which the overlap-
        # removal pass silently dropped, producing ~0 oi predictions at
        # inference. `object_involved` is also excluded from the main head's
        # vocab (see generate_tagging_vocab) so the two heads don't double-
        # supervise the same role.
        if self.oi_label_stoi:
            self.oi_label_ffn = Linears(
                [self.base_model_dim, self.config.linear_hidden_num, len(self.oi_label_stoi)],
                dropout_prob=self.config.linear_dropout,
                bias=self.config.linear_bias,
                activation=self.config.linear_activation,
            )
            if self.config.use_crf:
                self.oi_crf = CRF(self.oi_label_stoi, bioes=False)
        else:
            self.oi_label_ffn = None

        # Shared classification head: one Linear over the union of class
        # labels across every classification factor. Per-factor allowed
        # subsets are enforced at decode (and during the cls loss) via the
        # masks registered by `_register_cls_masks`. Skipped entirely when
        # the task has no classification factors.
        if self.cls_factors:
            self.cls_head = nn.Linear(
                self.base_model_dim, len(self.class_stoi),
                bias=getattr(self.config, "linear_bias", True))
            self._register_cls_masks()
        else:
            self.cls_head = None

        # Per-role loss weights (XGear's group_weight schema). Parsing lives
        # in TextEE/models/ACtrainer.py so EAE-AC trainers and their models
        # use the same policy. Empty dict ⇒ uniform weighting; downstream
        # is a no-op in that case.
        self._role_weights = parse_group_weight(self.config)

    def _seq_weight(self, arguments, candidate=None):
        """Per-sequence loss weight (mirrors EAEACMixin._seq_weight).

        AC-CR (`candidate` given): weight = role_weights[candidate]
        AC-C  (`candidate` is None): weight = max role_weight across the
                                     sequence's gold arguments; sequences
                                     with no gold args fall back to 1.0.
        Roles not listed default to weight 1.0.
        """
        if not self._role_weights:
            return 1.0
        if candidate is not None:
            return self._role_weights.get(candidate, 1.0)
        if not arguments:
            return 1.0
        return max(self._role_weights.get(arg[2], 1.0) for arg in arguments)

    def _build_cls_vocab(self):
        """Build the classification head's vocab from `config.key_map`.

        Drives cls factors entirely off the global mapping:
          factors = key_map.task.classification.columns − config.skip_columns

        Effect:
          * AC / ACH: `accident_type` lives in skip_columns (it's given as
            input and shouldn't be scored), so it's stripped — head sizes
            to [construction_trade, severity] only.
          * E2AC / E2ACH: skip_columns omits `accident_type`, so the head
            also predicts it — same model class, different config.

        Sets:
          self.cls_factors  — sorted list of classification factor names
          self.class_stoi   — unified vocab over the union of all classes
                              across cls_factors (label → idx).
          self.class_itos   — inverse.
          self._cls_classes_source — per-factor class lists, used by
                              `_register_cls_masks` after __init__.
        """
        key_map = getattr(self.config, "key_map", None) or {}
        task_cfg = key_map.get("task") or {}
        cls_block = task_cfg.get("classification") or {}
        all_classes_map = cls_block.get("classes") or {}
        # Preferred source of truth: classification.columns. Fall back to
        # classes' own keys when columns is missing.
        declared_cols = cls_block.get("columns") or list(all_classes_map.keys())
        skip = set(getattr(self.config, "skip_columns", None) or [])
        active = [c for c in declared_cols
                  if c in all_classes_map and c not in skip]

        self.cls_factors = sorted(active)
        all_classes = sorted({c for f in active for c in all_classes_map[f]})
        self.class_stoi = {c: i for i, c in enumerate(all_classes)}
        self.class_itos = {i: c for c, i in self.class_stoi.items()}
        # Defer mask registration until the module is initialized — we keep
        # the source map on `self` so __init__ can register the buffers
        # after super().__init__().
        self._cls_classes_source = {f: all_classes_map[f] for f in active}

    def _register_cls_masks(self):
        """Register per-factor bool masks as non-persistent buffers.

        Each `_cls_mask_<factor>` is a BoolTensor over the unified
        `class_stoi` that is True iff the column is in that factor's
        declared class subset. Used to mask disallowed logits to -1e9 at
        loss + decode time. Must be called AFTER nn.Module.__init__.
        """
        self._cls_mask_names = {}
        for factor, classes in self._cls_classes_source.items():
            mask = torch.zeros(len(self.class_stoi), dtype=torch.bool)
            for c in classes:
                if c in self.class_stoi:
                    mask[self.class_stoi[c]] = True
            buf_name = f"_cls_mask_{factor}"
            self.register_buffer(buf_name, mask, persistent=False)
            self._cls_mask_names[factor] = buf_name

    def cls_factor_mask(self, factor):
        """Return the per-factor allowed-class mask buffer (lives on the
        model's device). Unknown factor → all-True (no constraint)."""
        buf_name = getattr(self, "_cls_mask_names", {}).get(factor)
        if buf_name is None:
            return torch.ones(len(self.class_stoi), dtype=torch.bool,
                              device=next(self.parameters()).device)
        return getattr(self, buf_name)

    def generate_tagging_vocab(self):
        prefix = ['B', 'I']
        trigger_label_stoi = {'O': 0}
        for t in sorted(self.type_set["trigger"]):
            for p in prefix:
                trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

        # Filter classification factors AND object_involved out of the main BIO
        # vocab. Classification factors (e.g. construction_trade, severity) are
        # handled by `cls_head`. object_involved is handled by `oi_label_ffn`,
        # a dedicated parallel BIO head — see __init__ for the rationale.
        cls_set = set(getattr(self, "cls_factors", []))
        oi_role = "object_involved"
        has_oi = oi_role in self.type_set.get("role", set())
        role_label_stoi = {'O': 0}
        for t in sorted(self.type_set["role"]):
            if t in cls_set or t == oi_role:
                continue
            for p in prefix:
                role_label_stoi['{}-{}'.format(p, t)] = len(role_label_stoi)

        # CR: per-candidate iteration → unified B-Pred / I-Pred BIO.
        # C : single inference          → role-specific BIO.
        if self.config.priming_type == "condition+relation":
            self.label_stoi = {"trigger": trigger_label_stoi,
                               "role":    {"O": 0, "B-Pred": 1, "I-Pred": 2}}
        else:
            self.label_stoi = {"trigger": trigger_label_stoi, "role": role_label_stoi}

        # Vocab for the dedicated oi head. Only built in C-mode when
        # object_involved is present in the training type_set. CR-mode
        # supervises oi through the main head via its own CR candidate
        # forward (see _cr_valid_roles); per-candidate `specify_role` in
        # get_role_seqlabels prevents the inside-narrative overlap that
        # motivated the parallel head in C-mode, so a separate oi head
        # would be redundant. Leaving self.oi_label_stoi=None here means
        # __init__ skips building self.oi_label_ffn / self.oi_crf, and
        # forward()/predict()'s `if self.oi_label_ffn is not None` guards
        # naturally no-op for CR.
        if has_oi and self.config.priming_type == "condition":
            self.oi_label_stoi = {"O": 0,
                                  f"B-{oi_role}": 1,
                                  f"I-{oi_role}": 2}
        else:
            self.oi_label_stoi = None

        trigger_type_stoi = {t: i for i, t in enumerate(sorted(self.type_set["trigger"]))}
        role_type_stoi    = {t: i for i, t in enumerate(sorted(self.type_set["role"]))}
        self.type_stoi    = {"trigger": trigger_type_stoi, "role": role_type_stoi}

    def _cr_valid_roles(self, trigger_type):
        """CR-mode candidate list — returns patterns[trigger_type] unfiltered.

        Each CR forward is independently supervised via `specify_role` (see
        get_role_seqlabels), which skips every role that isn't the candidate.
        So the inside-narrative overlap that motivated the C-mode oi head
        (working_circumstances span swallowing object_involved tokens during
        overlap removal) physically cannot happen in CR: the
        `</s> Fall </s> Object Involved` forward's BIO target only labels
        oi spans, with wc tokens left as O. oi is just another candidate here.

        The parallel oi head still exists (built in __init__) but is
        deliberately skipped during CR-mode forward()/predict() since the
        main head's per-candidate path already supervises oi.
        """
        return sorted(patterns.get(self.config.dataset, {}).get(trigger_type, []))

    def get_oi_seqlabels(self, oi_args, token_num):
        """Build BIO labels for the oi head from raw oi args (no overlap removal).

        Only object_involved spans are tagged; everything else is O. Multiple
        oi spans in the same instance that overlap each other (rare) skip the
        later one to keep BIO well-formed.
        """
        labels = ['O'] * token_num
        for arg in sorted(oi_args, key=lambda a: (a[0], a[1])):
            start, end = arg[0], arg[1]
            if start < 0 or end > token_num or start >= end:
                continue
            if any(labels[i] != 'O' for i in range(start, end)):
                continue
            labels[start] = 'B-object_involved'
            for i in range(start + 1, end):
                labels[i] = 'I-object_involved'
        return labels

    def process_data(self, batch):
        enc_idxs = []
        enc_attn = []
        role_seqidxs = []
        oi_seqidxs = []  # parallel oi-head targets; one row per forward
        token_lens = []
        token_nums = []
        seq_weights = []
        cls_targets_rows = []  # parallel to enc_idxs; one row per forward
        cr_skipped = 0          # K=0 instances dropped in CR mode (see below)
        cr_skipped_types = {}   # per-trigger-type counter for the log line
        max_token_num = max(batch.batch_token_num)

        bos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        sep    = self.tokenizer.sep_token
        pad_label = 0 if self.config.use_crf else -100

        is_cr = self.config.priming_type == "condition+relation"
        cls_set = set(getattr(self, "cls_factors", []))
        n_cls   = len(self.cls_factors)
        # Stable index lookup for the cls_targets columns.
        cls_factor_idx = {f: i for i, f in enumerate(self.cls_factors)}

        has_oi_head = self.oi_label_ffn is not None

        event_type_map = event_type_tags.get(self.config.dataset, {})
        role_type_map  = role_type_tags.get(self.config.dataset, {})

        # ACBatch always carries batch_oi_arguments (from AC_collate_fn); if a
        # caller somehow passes an EAEBatch (no oi field), fall back to [] per
        # instance so we degrade gracefully to "no oi supervision".
        batch_oi_args = getattr(batch, "batch_oi_arguments", None)
        if batch_oi_args is None:
            batch_oi_args = [[] for _ in batch.batch_arguments]

        for tokens, pieces, trigger, arguments, oi_args, token_len, token_num in zip(
                batch.batch_tokens, batch.batch_pieces, batch.batch_trigger,
                batch.batch_arguments, batch_oi_args,
                batch.batch_token_lens, batch.batch_token_num):

            # Split args into extraction (real token spans) vs classification
            # (sentinel-offset args carrying class labels). Classification args
            # → cls_head. Object_involved:
            #   • C-mode: stripped here, routed to the parallel oi head
            #     (option E) since single-forward BIO can't co-label oi +
            #     overlapping wc narrative on the same token.
            #   • CR-mode: kept. Per-candidate `specify_role` in get_role_seqlabels
            #     isolates each forward's BIO target to just its candidate's
            #     spans, so wc and oi never collide in the same target. oi
            #     becomes a regular CR candidate; oi head's loss/output is
            #     gated off in CR (see forward / predict).
            ext_arguments = [a for a in arguments
                             if a[2] not in cls_set
                             and (is_cr or a[2] != "object_involved")
                             and a[0] >= 0 and a[1] > a[0]]

            # Per-instance gold-class lookup: {factor: class_idx} pulled from
            # the sentinel-span cls args (their `text` field carries the
            # class label like "electrical" / "fatality").
            inst_cls_gold = {}
            for a in arguments:
                role = a[2]
                if role in cls_set and len(a) > 3:
                    label = a[3]
                    if label and label in self.class_stoi:
                        inst_cls_gold[role] = self.class_stoi[label]

            event_type_label = event_type_map.get(trigger[2], trigger[2])
            piece_id = self.tokenizer.convert_tokens_to_ids(pieces)

            # Per-instance oi BIO target — same across all forwards of this
            # instance (the oi head is prompt-agnostic). Padded to max length
            # using the same pad_label policy as the main head.
            if has_oi_head:
                oi_seq = self.get_oi_seqlabels(oi_args, len(tokens))
                oi_row = (
                    [self.oi_label_stoi[s] for s in oi_seq]
                    + [pad_label] * (max_token_num - len(tokens))
                )
            else:
                oi_row = None

            if is_cr:
                # Per-candidate iteration: one prompt per role in patterns[trigger_type].
                # Excludes object_involved — its training/decoding lives on the
                # oi head, not as a CR candidate prompt.
                # CR-mode K=0 instances (working_circumstances / object_involved
                # Level-2 — no extraction sub-roles after cls factors are
                # filtered) are dropped upstream by the trainer's CR filter for
                # training, but predict batches can still contain them (the
                # hierarchical predictor builds batches per main_factor type
                # without filtering); the skip counter below logs how often.
                valid_roles = self._cr_valid_roles(trigger[2])
                if not valid_roles:
                    cr_skipped += 1
                    cr_skipped_types[trigger[2]] = cr_skipped_types.get(trigger[2], 0) + 1
                    continue
                for candidate in valid_roles:
                    role_label = role_type_map.get(candidate, candidate)
                    prompt = "{} {} {} {}".format(
                        sep, event_type_label,
                        sep, role_label,
                    )
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    enc_idx = [bos_id] + piece_id + prompt_id + [eos_id]
                    enc_idxs.append(enc_idx)
                    enc_attn.append([1] * len(enc_idx))

                    role_seq = self.get_role_seqlabels(
                        ext_arguments, len(tokens),
                        specify_role=candidate, use_unified_label=True,
                    )
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    role_seqidxs.append(
                        [self.label_stoi["role"][s] for s in role_seq]
                        + [pad_label] * (max_token_num - len(tokens))
                    )
                    if has_oi_head:
                        oi_seqidxs.append(list(oi_row))
                    # CR: weight is keyed on the candidate role for this prompt.
                    seq_weights.append(self._seq_weight(ext_arguments, candidate=candidate))
                    # CR cls target: this forward is "primed" on `candidate`,
                    # so only the column for `candidate` (if it's a cls factor)
                    # is supervised; everything else is -100 (ignore).
                    row = [-100] * n_cls
                    if candidate in cls_set and candidate in inst_cls_gold:
                        row[cls_factor_idx[candidate]] = inst_cls_gold[candidate]
                    cls_targets_rows.append(row)
            else:
                # AC-C: single inference per instance — just the event type label.
                prompt = "{} {}".format(sep, event_type_label)
                prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                enc_idx = [bos_id] + piece_id + prompt_id + [eos_id]
                enc_idxs.append(enc_idx)
                enc_attn.append([1] * len(enc_idx))

                role_seq = self.get_role_seqlabels(ext_arguments, len(tokens))
                token_lens.append(token_len)
                token_nums.append(token_num)
                role_seqidxs.append(
                    [self.label_stoi["role"][s] for s in role_seq]
                    + [pad_label] * (max_token_num - len(tokens))
                )
                if has_oi_head:
                    oi_seqidxs.append(list(oi_row))
                # AC-C: weight is the max group weight of any role in the gold target.
                seq_weights.append(self._seq_weight(ext_arguments))
                # C cls target: one forward covers all cls factors.
                row = [-100] * n_cls
                for f, idx in cls_factor_idx.items():
                    if f in inst_cls_gold:
                        row[idx] = inst_cls_gold[f]
                cls_targets_rows.append(row)

        if not enc_idxs:
            # All instances in this batch had K=0 valid_roles in CR mode
            # (every instance was a working_circumstances / object_involved
            # Level-2 event whose only sub-factors are cls filtered out of
            # patterns, or has no subs at all). predict() handles None by
            # returning empty per-instance predictions. forward() shouldn't
            # reach here because the trainer's CR filter drops K=0 instances
            # upstream of the DataLoader for training.
            logger.warning(
                f"[process_data] empty batch — all {cr_skipped} instances "
                f"had K=0 valid_roles in CR mode. "
                f"Per-trigger-type drop counts: {cr_skipped_types}. "
                f"Returning None; caller will emit empty predictions."
            )
            return None
        if cr_skipped > 0:
            # Partial drop — some instances skipped, others kept.
            logger.debug(
                f"[process_data] CR-mode dropped {cr_skipped} K=0 instances "
                f"from batch of size {len(batch.batch_trigger)} "
                f"(per-type: {cr_skipped_types}); kept "
                f"{len(batch.batch_trigger) - cr_skipped} for forward."
            )
        max_len = max(len(enc_idx) for enc_idx in enc_idxs)
        enc_idxs = torch.LongTensor(
            [enc_idx + [pad_id] * (max_len - len(enc_idx)) for enc_idx in enc_idxs]
        )
        enc_attn = torch.LongTensor(
            [enc_att + [0] * (max_len - len(enc_att)) for enc_att in enc_attn]
        )
        enc_idxs     = enc_idxs.cuda()
        enc_attn     = enc_attn.cuda()
        role_seqidxs = torch.cuda.LongTensor(role_seqidxs)
        token_nums   = torch.cuda.LongTensor(token_nums)
        seq_weights  = torch.cuda.FloatTensor(seq_weights)
        if n_cls > 0 and cls_targets_rows:
            cls_targets = torch.cuda.LongTensor(cls_targets_rows)
        else:
            # Empty placeholder — no cls factors configured, or no forwards.
            cls_targets = torch.zeros((len(enc_idxs), max(n_cls, 0)),
                                      dtype=torch.long, device=enc_idxs.device)
        if has_oi_head and oi_seqidxs:
            oi_targets = torch.cuda.LongTensor(oi_seqidxs)
        else:
            oi_targets = None
        return enc_idxs, enc_attn, role_seqidxs, token_lens, token_nums, seq_weights, cls_targets, oi_targets

    def forward(self, batch):
        result = self.process_data(batch)
        if result is None:
            # Defensive — trainer-level CR filter should prevent empty
            # training batches, but if one slips through return a zero loss
            # anchored to a parameter so autograd has a valid graph.
            return next(self.parameters()).sum() * 0.0
        enc_idxs, enc_attn, role_seqidxs, token_lens, token_nums, seq_weights, cls_targets, oi_targets = result
        base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens)
        ext_loss = self._weighted_loss(base_model_outputs, role_seqidxs, token_nums, seq_weights)

        # oi loss from the parallel head. Trained on raw object_involved spans
        # (no overlap removal), prompt-agnostic — same target across all CR
        # forwards of an instance. Skipped when the oi head wasn't built (e.g.
        # the training type_set has no object_involved role).
        oi_loss = None
        if self.oi_label_ffn is not None and oi_targets is not None:
            oi_loss = self._oi_loss(base_model_outputs, oi_targets, token_nums)

        # The oi head's loss weight is now sourced from the unified
        # `group_weight` map (same place that drives BIO sequence weights and
        # cls factor weights). Lookup key is the role name `"object_involved"`;
        # roles not listed default to 1.0. Keeps weight configuration in one
        # block — no separate `oi_loss_wgt` config field needed.
        oi_wgt = float(self._role_weights.get("object_involved", 1.0))

        # When there are no cls factors (e.g. dataset has no classification
        # head defined, or `cls_head` was skipped) keep the legacy single-head
        # behavior and return just the BIO loss (+ oi loss if applicable).
        if self.cls_head is None or not self.cls_factors:
            if oi_loss is None:
                return ext_loss
            return ext_loss + oi_wgt * oi_loss

        cls_loss = self._cls_loss(base_model_outputs, token_nums, cls_targets)
        # Defaults form a 50/50 convex combination (ext_wgt + cls_wgt = 1).
        # Override either in config to bias toward one head, e.g.
        # `ext_loss_wgt: 0.7, cls_loss_wgt: 0.3` to upweight extraction.
        # oi term is additive on top, weighted by group_weight["object_involved"].
        ext_wgt  = float(getattr(self.config, "ext_loss_wgt", 0.5))
        cls_wgt  = float(getattr(self.config, "cls_loss_wgt", 0.5))
        total = ext_wgt * ext_loss + cls_wgt * cls_loss
        if oi_loss is not None:
            total = total + oi_wgt * oi_loss
        return total

    def _oi_loss(self, base_model_outputs, oi_targets, token_nums):
        """Loss for the parallel oi head — same CRF-or-CE policy as the main head."""
        scores = self.oi_label_ffn(base_model_outputs)
        if self.config.use_crf:
            scores_padded = self.oi_crf.pad_logits(scores)
            loglik = self.oi_crf.loglik(scores_padded, oi_targets, token_nums)
            return -loglik.mean()
        n_labels = scores.size(-1)
        tok_losses = F.cross_entropy(
            scores.view(-1, n_labels),
            oi_targets.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(oi_targets.size(0), -1)
        tok_mask = (oi_targets != -100).float()
        seq_loss = (tok_losses * tok_mask).sum(dim=1) / tok_mask.sum(dim=1).clamp(min=1.0)
        return seq_loss.mean()

    def _decode_oi(self, base_model_outputs, token_nums):
        """Decode oi BIO spans from the parallel oi head. Returns one list of
        [start, end, "object_involved"] per forward in the batch."""
        scores = self.oi_label_ffn(base_model_outputs)
        if self.config.use_crf:
            scores_padded = self.oi_crf.pad_logits(scores)
            _, label_preds = self.oi_crf.viterbi_decode(scores_padded, token_nums)
        else:
            label_preds = torch.argmax(scores, dim=-1)
        spans = self.tag_paths_to_spans(label_preds, token_nums, self.oi_label_stoi)
        # tag_paths_to_spans returns tag = "object_involved" already, since
        # oi_label_stoi uses the full role name in B-/I- labels. Each span is
        # [start, end, "object_involved"].
        return spans

    def _pool_doc(self, base_model_outputs, token_nums):
        """Mean-pool the per-token encoder output over real tokens.

        `base_model_outputs` is the post-multi-piece-collapse representation
        — one vector per doc token, padded to max_token_num. Masking by
        `token_nums` zeroes the padded positions before averaging.
        """
        max_len = base_model_outputs.size(1)
        mask = (torch.arange(max_len, device=base_model_outputs.device)[None, :]
                < token_nums[:, None]).float().unsqueeze(-1)
        summed = (base_model_outputs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _cls_loss(self, base_model_outputs, token_nums, cls_targets):
        """Joint per-factor cross-entropy loss over the shared cls head.

        For each cls factor, the head's logits are masked to that factor's
        allowed-class subset before CE so the model can't predict labels
        from another factor's vocab. Forwards with no supervision for a
        given factor (cls_target == -100) are ignored via `ignore_index`.

        Per-factor weights are pulled from `self._role_weights` (the same
        `group_weight` map that drives BIO extraction weighting via
        `_weighted_loss`), so the config can boost specific cls factors
        (e.g. raise `construction_trade` weight to give the cls head a
        bigger gradient on that hard 17-class label). Factors not listed
        default to weight 1.0.

        Aggregation is a WEIGHTED MEAN across participating factors —
        `Σ wᵢ·CEᵢ / Σ wᵢ` — so the overall cls term's magnitude stays
        roughly invariant to how many factors the task has and what the
        weight values are.
        """
        pooled = self._pool_doc(base_model_outputs, token_nums)
        logits = self.cls_head(pooled)  # (num_forwards, |class_stoi|)

        total = logits.new_zeros(())
        total_weight = 0.0
        for fi, factor in enumerate(self.cls_factors):
            target_col = cls_targets[:, fi]
            if not bool((target_col != -100).any()):
                continue  # no supervision for this factor in this batch
            mask = self.cls_factor_mask(factor)  # (|class_stoi|,)
            masked = logits.masked_fill(~mask, -1e9)
            ce = F.cross_entropy(masked, target_col,
                                 ignore_index=-100, reduction="mean")
            weight = float(self._role_weights.get(factor, 1.0))
            total = total + weight * ce
            total_weight += weight
        if total_weight == 0.0:
            return logits.new_zeros(())
        return total / total_weight

    def _weighted_loss(self, base_model_outputs, target, token_nums, seq_weights):
        """Per-sequence-weighted version of `span_id` (training only).

        seq_weights ∈ R^batch is the max role weight present in each
        sequence's gold target (AC-C) or the candidate role's weight (AC-CR).
        Weighted mean replaces the unweighted mean used by the parent's
        span_id; uniform weights reduce to the original behavior exactly.
        """
        scores = self.role_label_ffn(base_model_outputs)

        # Defensive: keep weights on the same device as scores and avoid
        # division-by-zero when an entire batch happens to have w=0.
        seq_weights = seq_weights.to(scores.device).float()
        denom       = seq_weights.sum().clamp(min=1e-6)

        if self.config.use_crf:
            scores_padded = self.role_crf.pad_logits(scores)
            loglik = self.role_crf.loglik(scores_padded, target, token_nums)  # (batch,)
            loss   = -(loglik * seq_weights).sum() / denom
        else:
            n_labels   = scores.size(-1)
            tok_losses = F.cross_entropy(
                scores.view(-1, n_labels),
                target.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(target.size(0), -1)
            tok_mask  = (target != -100).float()
            seq_loss  = (tok_losses * tok_mask).sum(dim=1) / tok_mask.sum(dim=1).clamp(min=1.0)
            loss      = (seq_loss * seq_weights).sum() / denom
        return loss

    @staticmethod
    def _correct_tags(spans, max_gap=1):
        """Bridge fragmented same-type spans into one.

        BIO + CRF tends to break long argument spans at uncertain interior
        tokens, producing fragments like
            [(5, 7, 'tool_cond'), (9, 12, 'tool_cond')]
        for what should be one span (5, 12). Merging same-type fragments
        separated by <= `max_gap` tokens recovers the long span. Operates
        on the (start, end, role) tuples emitted by tag_paths_to_spans.
        """
        if not spans:
            return list(spans)
        spans_sorted = sorted(spans, key=lambda s: (s[0], s[1]))
        merged = [list(spans_sorted[0])]
        for span in spans_sorted[1:]:
            last = merged[-1]
            if span[2] == last[2] and span[0] - last[1] <= max_gap:
                last[1] = max(last[1], span[1])
            else:
                merged.append(list(span))
        return merged

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            result = self.process_data(batch)
            if result is None:
                # All instances in this batch had K=0 valid_roles (CR mode,
                # batch composed entirely of working_circumstances /
                # object_involved Level-2 events). Return one empty argument
                # list and one empty cls dict per original instance so the
                # downstream zip() in predict_ac_eae_hierarchical stays
                # length-aligned with the input.
                self.train()
                n = len(batch.batch_trigger)
                return [[] for _ in range(n)], [{} for _ in range(n)]
            (enc_idxs, enc_attn, _, token_lens, token_nums,
             _seq_weights, _cls_targets, _oi_targets) = result
            base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens)
            _, arguments = self.span_id(base_model_outputs, token_nums, predict=True)

            # AC-CR: decompose unified Pred predictions back to candidate role types
            # — same logic as EAEmodel.predict's condition+relation branch.
            # Uses _cr_valid_roles (object_involved excluded) to stay aligned
            # with the forward-list produced by process_data.
            if self.config.priming_type == "condition+relation":
                cnt = 0
                new_arguments = []
                for trigger in batch.batch_trigger:
                    valid_roles = self._cr_valid_roles(trigger[2])
                    new_sub = []
                    for candidate in valid_roles:
                        new_sub.extend([[a[0], a[1], candidate] for a in arguments[cnt]])
                        cnt += 1
                    new_arguments.append(new_sub)
                assert cnt == enc_idxs.size(0)
                arguments = new_arguments

            # Optional post-tagging correction — bridges fragmented same-type
            # spans the BIO/CRF tagger broke at uncertain interior tokens.
            # `tag_correction_max_gap` controls how many O tokens may sit
            # between two same-type fragments before they're merged
            # (default 1 = bridge single-token holes only).
            if getattr(self.config, "tag_correction", False):
                max_gap = int(getattr(self.config, "tag_correction_max_gap", 1) or 0)
                arguments = [self._correct_tags(a, max_gap=max_gap) for a in arguments]

            # Parallel oi head → per-instance object_involved spans, merged
            # into the main arguments list.
            if self.oi_label_ffn is not None:
                oi_spans_per_forward = self._decode_oi(base_model_outputs, token_nums)
                if self.config.priming_type == "condition+relation":
                    # K forwards per instance — oi target was the same on each,
                    # so take the first forward's predictions per instance to
                    # avoid emitting duplicates.
                    cnt = 0
                    for i, trigger in enumerate(batch.batch_trigger):
                        valid_roles = self._cr_valid_roles(trigger[2])
                        if not valid_roles:
                            continue
                        first_idx = cnt
                        for span in oi_spans_per_forward[first_idx]:
                            arguments[i].append([span[0], span[1], "object_involved"])
                        cnt += len(valid_roles)
                else:
                    # AC-C: 1 forward per instance — straight zip.
                    for i, oi_spans in enumerate(oi_spans_per_forward):
                        for span in oi_spans:
                            arguments[i].append([span[0], span[1], "object_involved"])

            # Per-instance cls predictions from the shared cls head.
            cls_predictions = self._predict_cls(base_model_outputs, token_nums, batch)
        self.train()
        return arguments, cls_predictions

    def _predict_cls(self, base_model_outputs, token_nums, batch):
        """Decode per-instance {factor: label} from the cls head.

        Returns a list of dicts of length `len(batch.batch_trigger)`.

        Aggregation across the two priming modes:
          C  — one forward per instance; the same pooled vector predicts
                every cls factor.
          CR — K forwards per instance (one per candidate role); for each
                cls factor we read the forward whose `candidate` IS that
                factor (the prompt was primed on it).
        """
        batch_size = len(batch.batch_trigger)
        if self.cls_head is None or not self.cls_factors:
            return [{} for _ in range(batch_size)]

        pooled = self._pool_doc(base_model_outputs, token_nums)
        logits = self.cls_head(pooled)  # (num_forwards, |class_stoi|)

        results = [{} for _ in range(batch_size)]
        is_cr   = self.config.priming_type == "condition+relation"
        cls_set = set(self.cls_factors)

        if is_cr:
            cnt = 0
            for i, trigger in enumerate(batch.batch_trigger):
                # Use the same _cr_valid_roles filter as process_data so cnt
                # stays in lock-step with the forward list (object_involved
                # is not a CR candidate; it's handled by the oi head).
                valid_roles = self._cr_valid_roles(trigger[2])
                for candidate in valid_roles:
                    if candidate in cls_set:
                        mask = self.cls_factor_mask(candidate)
                        masked = logits[cnt].masked_fill(~mask, float("-inf"))
                        pred_idx = int(masked.argmax().item())
                        results[i][candidate] = self.class_itos[pred_idx]
                    cnt += 1
        else:
            for i in range(batch_size):
                for factor in self.cls_factors:
                    mask = self.cls_factor_mask(factor)
                    masked = logits[i].masked_fill(~mask, float("-inf"))
                    pred_idx = int(masked.argmax().item())
                    results[i][factor] = self.class_itos[pred_idx]
        return results
