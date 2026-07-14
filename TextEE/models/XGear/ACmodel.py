import torch
import torch.nn.functional as F

from .EAEmodel import XGearEAEModel


class XGearACModel(XGearEAEModel):
    """AC variant with per-group loss weighting.

    `config.group_weight` is a list of `{"roles": [...], "weight": float}`
    entries. Each role in a group has its loss multiplied by the group's
    weight. The literal entry "[None]" in a roles list refers to the
    `[None]` special token; all other entries are role names whose
    `<--R-->`/`</--R-->` tag pair plus content positions are weighted.

    Example:

        "group_weight": [
            {"roles": ["[None]"],                       "weight": 0.1},   # downweight
            {"roles": ["body_part", "object", "task"], "weight": 1.0},   # default
            {"roles": ["consq", "work_ctx"],           "weight": 3.0},   # boost moderate
            {"roles": ["mgmt_factor", "ppe_cond"],     "weight": 5.0}    # boost rare
        ]

    Mechanics, per role R with group weight w:
      (a) token_weights[`<--R-->`] = token_weights[`</--R-->`] = w —
          direct gradient pressure to EMIT the role tags. Without this,
          rare roles stay at 0 predictions because beam search never
          enters the "inside" state for them.
      (b) Position weight = w for tokens strictly between open/close tags
          — boosts content prediction once R is being emitted.
    Open + content + close are uniformly at w_R, so the entire R block
    is upweighted. For `[None]` only (a) applies — the token has no
    span structure.

    Roles not listed in any group get default weight 1.0.
    Set `group_weight: []` (or omit) to disable all weighting.
    """

    def __init__(self, config, tokenizer, type_set):
        super().__init__(config, tokenizer, type_set)

        none_id = tokenizer.convert_tokens_to_ids("[None]")
        assert none_id != tokenizer.unk_token_id, \
            "[None] not registered as a special token in the tokenizer"
        self.none_token_id = none_id

        # --- Parse group_weight: role -> weight (last group wins) -------
        group_weight_cfg = list(getattr(config, "group_weight", []) or [])
        role_weight_map = {}
        for group in group_weight_cfg:
            if not isinstance(group, dict):
                continue
            w = float(group.get("weight", 1.0))
            for r in group.get("roles", []) or []:
                role_weight_map[r] = w

        # --- Build token_weights and per-role position-weight specs ----
        # token_weights[t] is the per-token-id loss weight (used as the
        # `weight` arg to F.cross_entropy in the fast path, or applied
        # multiplicatively via gather in the manual path).
        # position_specs is the list of (open_id, close_id, weight) tuples
        # for roles whose CONTENT positions need a non-default weight.
        token_weights = torch.ones(len(tokenizer))
        position_specs = []  # plain Python list — small, no GPU sync
        for role, w in role_weight_map.items():
            if role == "[None]":
                # Special: just weight the [None] token; no span structure.
                token_weights[none_id] = w
                continue
            oid = tokenizer.convert_tokens_to_ids(f"<--{role}-->")
            cid = tokenizer.convert_tokens_to_ids(f"</--{role}-->")
            assert oid != tokenizer.unk_token_id, \
                f"<--{role}--> not registered as a special token (check group_weight vs pattern)"
            assert cid != tokenizer.unk_token_id, \
                f"</--{role}--> not registered as a special token (check group_weight vs pattern)"
            if w != 1.0:
                token_weights[oid] = w
                token_weights[cid] = w
                position_specs.append((oid, cid, w))
        self.register_buffer("token_weights", token_weights)
        self._role_position_specs = position_specs
        # Manual loss path is only needed when at least one role has a
        # non-1.0 *position* weight. [None] alone uses the fast path.
        self._upweight_active = bool(position_specs)

    def _position_weights(self, raw_lbl_idxs):
        """Return [B, L] per-position weights for content tokens inside
        each role's `<--R-->...</--R-->` span, using that role's group
        weight. Positions outside all weighted spans get 1.0.

        Assumes role tags are not nested (target builder guarantees this),
        so iterating roles and overwriting via torch.where is safe — at
        most one role's `inside` mask is True at any position.
        """
        weight_dtype = self.token_weights.dtype
        pos_w = torch.ones_like(raw_lbl_idxs, dtype=weight_dtype)
        if not self._role_position_specs:
            return pos_w
        for open_id, close_id, weight in self._role_position_specs:
            is_open = (raw_lbl_idxs == open_id).to(weight_dtype)
            is_close = (raw_lbl_idxs == close_id).to(weight_dtype)
            # Exclusive prefix sum of opens; inclusive prefix sum of closes.
            excl_cum_opens = torch.cat(
                [torch.zeros_like(is_open[:, :1]),
                 torch.cumsum(is_open, dim=1)[:, :-1]], dim=1,
            )
            incl_cum_closes = torch.cumsum(is_close, dim=1)
            inside = excl_cum_opens > incl_cum_closes
            pos_w = torch.where(inside,
                                torch.full_like(pos_w, weight),
                                pos_w)
        return pos_w

    def forward(self, batch):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        outputs = self.model(input_ids=enc_idxs,
                             attention_mask=enc_attn,
                             decoder_input_ids=dec_idxs,
                             decoder_attention_mask=dec_attn,
                             return_dict=True)
        logits = outputs.logits

        if not self._upweight_active:
            # Fast path: only [None] downweighting via class weights.
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                lbl_idxs.reshape(-1),
                weight=self.token_weights,
                ignore_index=-100,
            )

        # Manual weighted CE so we can combine token-id weights (for [None])
        # with per-position weights (for infrequent role content).
        log_probs = F.log_softmax(logits, dim=-1)
        valid = (lbl_idxs != -100).to(log_probs.dtype)
        safe_lbl = lbl_idxs.clamp_min(0)
        nll = -log_probs.gather(2, safe_lbl.unsqueeze(-1)).squeeze(-1)  # [B, L]
        token_w = self.token_weights[safe_lbl]                          # [B, L]
        pos_w = self._position_weights(raw_lbl_idxs)                    # [B, L]
        total_w = token_w * pos_w * valid
        return (nll * total_w).sum() / total_w.sum().clamp_min(1.0)

    def predict(self, batch, num_beams=4, max_length=50, length_penalty=1.0,
                bad_words_ids=None, min_new_tokens=None, repetition_penalty=None):
        """Override to plumb decoding constraints through to HF generate. The
        parent's generate() ignores kwargs, so for prompt-group decoding and
        bootcamp-time inference constraints we invoke `self.model.generate`
        directly with the per-call params:
          - length_penalty: per prompt-group decoding bias.
          - bad_words_ids: forbidden token sequences (e.g. [[none_id]] to
            forbid emitting `[None]` during bootcamp evals).
          - min_new_tokens: minimum number of tokens before EOS is allowed.
          - repetition_penalty: >1.0 penalizes any previously-generated
            token. Discourages `[And] [And]` and degenerate "the the the"
            loops during long-span generation. None or 1.0 = no penalty.
        """
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        self.eval()
        gen_kwargs = dict(
            input_ids=enc_idxs,
            attention_mask=enc_attn,
            num_beams=num_beams,
            max_length=max_length,
            length_penalty=float(length_penalty),
        )
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids
        if min_new_tokens is not None and int(min_new_tokens) > 0:
            gen_kwargs["min_new_tokens"] = int(min_new_tokens)
        if repetition_penalty is not None and float(repetition_penalty) != 1.0:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)
        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)
        final_output = []
        for bid in range(enc_idxs.size(0)):
            final_output.append(self.tokenizer.decode(
                outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        self.train()
        return final_output
