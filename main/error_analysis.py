"""LLM error analysis.

Five error categories evaluated per (model, dataset, task, k_shot), pooled
across splits:

  1. Over-extraction   — a pred span "swallows" exactly one gold span
                         (substring or token-overlap >= sim_threshold relative
                         to gold tokens) AND has extra material beyond it.
                         "I extracted the right thing, plus extras."
  2. Corrupted Output  — pred span contains special control tokens
                         (`<|thought|>`, `<|im_start|>`, `<|endoftext|>`, …),
                         partial-token leaks, runaway-loop signatures, or
                         duplicate list entries.
  3. Hallucination     — pred span (non-special tokens) is not a substring of
                         the source `acc_txt` or any same-factor gold at any
                         of three levels: strict, case-insensitive, normalized
                         (lowercase + collapsed whitespace + stripped punct).
                         Flagged when any level fails.
  4. Joined Span       — one pred span swallows TWO OR MORE distinct gold
                         spans (same swallow rule as over-extraction). A
                         strict superset case of over-extraction; the two
                         flags are mutually exclusive per pred span.
  5. Mislabeling — pred span fuzzy-matches (>= sim_threshold) a gold
                         span belonging to a SIBLING factor — i.e. another
                         child of the same parent in key_mapping.structure.

Output: one errors_<model>_<dataset>_<task>_k<K>.json per k_shot in the log
directory (defaults to the prediction directory), plus rendered text tables.
"""

import argparse
import datetime
import glob
import json
import logging
import math
import os
import re
from collections import defaultdict
from functools import lru_cache

from types import SimpleNamespace

# Reuse scorer's F1 primitives so error-analysis matching decisions are aligned
# with the score we're explaining: parse_ac_texts (split on ';' + lowercase),
# calculate_overlap (SequenceMatcher.ratio, no extra normalization), and
# is_factor_match (Hungarian-style strict 1-to-1 bijection at >= sim_threshold).
from TextEE.scorer import parse_ac_texts, calculate_overlap, is_factor_match

logger = logging.getLogger(__name__)

_RUN_ID = None

def make_run_id(prefix="log"):
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = f"{prefix}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _RUN_ID


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

# Structural columns (`accident_report`, `id`) are excluded via the global
# config's `skip_columns` (base_skip), and classification columns
# (`accident_type`, ...) via the key map's classification classes — both
# resolved in analyze_combination. Only IE-format structural leftovers +
# `doc_id` are hardcoded here so they're never treated as factors.
IGNORE_KEYS = {"doc_id", "tokens", "event_mentions"}


def _parse_texts(value):
    """Mirror scorer.parse_ac_texts: split list/strings on ';', strip, drop empties."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for item in items:
        try:
            if isinstance(item, float) and math.isnan(item):
                continue
        except (TypeError, ValueError):
            pass
        for part in str(item).split(";"):
            p = part.strip()
            if p:
                out.append(p)
    return out


# _normalize and _tokens are called many times per pred (per swallowed-gold
# scan, per substring-level check, per pred_within_gold). The same source
# string and the same gold spans get re-normalized for every pred in a record.
# lru_cache turns the repeats into O(1) dict lookups. Strings are hashable so
# the cache key is the input string itself. _tokens returns a tuple instead of
# a list so the cached value is immutable — every call site treats it as
# read-only (set(...), len(...), iteration) so tuple is a safe drop-in.
@lru_cache(maxsize=8192)
def _normalize(s):
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


@lru_cache(maxsize=8192)
def _tokens(s):
    return tuple(t for t in _normalize(s).split() if t)


def _substring_levels(span, references):
    """Three-level substring check of `span` against each reference string.

    Returns a dict {strict, case_ins, normalized}. A level passes if ANY
    reference contains the span at that level. References typically include
    the source `acc_txt` plus same-factor gold spans, so classification labels
    that match a gold but not the source are not falsely flagged.
    """
    s_low = span.lower()
    s_norm = _normalize(span)
    strict = case_ins = normalized = False
    for ref in references:
        if not ref:
            continue
        if not strict and span in ref:
            strict = True
        if not case_ins and s_low in ref.lower():
            case_ins = True
        if not normalized and s_norm in _normalize(ref):
            normalized = True
        if strict and case_ins and normalized:
            break
    return {"strict": strict, "case_ins": case_ins, "normalized": normalized}


def _best_fuzzy(query, candidates, threshold):
    """Return (idx, score) of best fuzzy match >= threshold, else (-1, 0.0).

    Uses scorer.calculate_overlap directly (lowercased on both sides to match
    parse_ac_texts' canonical form). The query is lowercased once outside the
    loop; empty strings are short-circuited because calculate_overlap's own
    empty-check still pays a function-call cost we don't need here.

    Length-ratio short-circuit: SequenceMatcher.ratio is upper-bounded by
    `2·min(len_a, len_b) / (len_a + len_b)` (== `real_quick_ratio`). For
    threshold T, the minimum acceptable length ratio is `T/(2−T)`. Pairs
    failing that cheap test can never reach `threshold`, so we skip the
    SequenceMatcher call entirely — that's the dominant cost of the loop.
    """
    if not query:
        return -1, 0.0
    q = query.lower()
    q_len = len(q)
    min_len_ratio = threshold / (2.0 - threshold) if threshold < 2.0 else 0.0
    best_idx, best = -1, -1.0
    for i, c in enumerate(candidates):
        if not c:
            continue
        c_low = c.lower()
        c_len = len(c_low)
        if min(q_len, c_len) < min_len_ratio * max(q_len, c_len):
            continue
        s = calculate_overlap(q, c_low)
        if s >= threshold and s > best:
            best, best_idx = s, i
    return best_idx, max(best, 0.0)


def _swallowed_gold_spans(pred_span, gold_spans, threshold):
    """Indices of gold spans contained inside pred_span — i.e. a SUBSET of
    pred matches the gold at >= threshold.

    A gold is considered swallowed when EITHER:
      - the gold's normalized form is a literal substring of the pred, OR
      - the share of gold tokens present in pred is >= `threshold`.

    This mirrors the F1 fuzzy-match notion: "some subset of pred matches the
    gold above the similarity threshold".
    """
    p_norm = _normalize(pred_span)
    p_toks = set(_tokens(pred_span))
    swallowed = []
    for i, g in enumerate(gold_spans):
        g_norm = _normalize(g)
        if g_norm and g_norm in p_norm:
            swallowed.append(i)
            continue
        g_toks = set(_tokens(g))
        if g_toks and len(p_toks & g_toks) / len(g_toks) >= threshold:
            swallowed.append(i)
    return swallowed


def _extra_tokens(pred_span, gold_spans):
    """Tokens present in pred but not in any of the given gold spans.
    Used to surface "what got over-extracted" beyond the matched golds."""
    if not gold_spans:
        return _tokens(pred_span)
    gold_tok_union = set()
    for g in gold_spans:
        gold_tok_union.update(_tokens(g))
    return [t for t in _tokens(pred_span) if t not in gold_tok_union]


def _pred_within_gold(pred_span, gold_span):
    """True when `pred_span` is contained in `gold_span` — i.e. gold "covers"
    pred. Mirror of `_swallowed_gold_spans` (which checks the opposite direction).

    Containment is asserted if EITHER:
      - normalized pred is a literal substring of normalized gold, OR
      - the pred token set is a subset of the gold token set.

    Used by the Under-extraction and Omission triggers, where the semantic
    question is "did the pred fit inside an annotated gold?".
    """
    p_norm = _normalize(pred_span)
    g_norm = _normalize(gold_span)
    if p_norm and g_norm and p_norm in g_norm:
        return True
    p_toks = set(_tokens(pred_span))
    g_toks = set(_tokens(gold_span))
    return bool(p_toks) and p_toks <= g_toks


def _build_siblings(structure):
    """Map each child factor -> list of siblings under the same parent."""
    siblings = {}
    for children in (structure or {}).values():
        for c in children:
            siblings[c] = [x for x in children if x != c]
    return siblings


# ---------------------------------------------------------------------------
# Corruption markers
# ---------------------------------------------------------------------------

_CORR_MARKERS = {
    "leak_thought": ("<|thought|>",),
    "leak_special": ("<|im_start|>", "<|im_end|>", "<|endoftext|>",
                     "<|eot_id|>", "<end_of_turn>"),
    "leak_partial": ("|thought", "Nonethought", "thoughttoken",
                     "tokentoken", "Nonetoken"),
}
_CORR_LONG_CHARS = 400
_CORR_REPEAT_CHUNK = 20
_CORR_REPEAT_MIN = 5
_CORR_FLAGS = (
    "leak_thought", "leak_special", "leak_partial",
    "very_long", "repeating", "duplicate_list",
)


def _detect_corruption(item):
    """Return a set of corruption flags for one pred string."""
    flags = set()
    if not isinstance(item, str):
        return flags
    for cat, markers in _CORR_MARKERS.items():
        if any(m in item for m in markers):
            flags.add(cat)
    if len(item) > _CORR_LONG_CHARS:
        flags.add("very_long")
    if len(item) >= _CORR_REPEAT_CHUNK * _CORR_REPEAT_MIN:
        chunk = item[:_CORR_REPEAT_CHUNK]
        if item.count(chunk) >= _CORR_REPEAT_MIN:
            flags.add("repeating")
    return flags



# ---------------------------------------------------------------------------
# Per-record analysis
# ---------------------------------------------------------------------------

def analyze_record(pred, gold, structure, sim_threshold=0.8, skip_factors=None, norm=False):
    """`skip_factors`, when provided, is a collection of factor names whose
    predictions and golds are excluded from the report entirely — same
    semantics as the LLM trainer's skip vector (factors with too few few-shot
    examples). Skipped factors don't contribute to any error category or to
    `pred_count` / `gold_count`."""
    source = " ".join(_parse_texts(gold.get("accident_report", "")))
    siblings = _build_siblings(structure)
    skip = set(skip_factors or ())

    gold_factors, pred_factors = {}, {}
    for k, v in gold.items():
        if k in IGNORE_KEYS or k in skip:
            continue
        if isinstance(k, str) and k.endswith("_keywords"):
            continue  # parallel keyword list — not a factor
        spans = _parse_texts(v)
        if spans:
            gold_factors[k] = spans
    for k, v in pred.items():
        if k in IGNORE_KEYS or k in skip:
            continue
        if isinstance(k, str) and k.endswith("_keywords"):
            continue
        spans = _parse_texts(v)
        if spans:
            pred_factors[k] = spans

    report = {
        "id": gold.get("id", gold.get("doc_id")),
        "accident_type": (gold.get("accident_type") or [None])[0],
        "factors": {},
    }

    for factor in sorted(set(gold_factors) | set(pred_factors)):
        g_list = gold_factors.get(factor, [])
        p_list = pred_factors.get(factor, [])

        # Scorer-aligned (lowercased, ';'-split) projection used to decide
        # whether this factor bundle would count as a match under the same
        # F1 rule that scorer.compute_AC_scores applies. Boolean only —
        # the per-span error categories below still operate on case-preserved
        # spans where needed (e.g. hallucination's strict substring check).
        g_scorer = parse_ac_texts(gold.get(factor), norm=norm)
        p_scorer = parse_ac_texts(pred.get(factor), norm=norm)
        scorer_match = is_factor_match(p_scorer, g_scorer, overlap_pct=sim_threshold)

        f_report = {
            "pred_count": len(p_list),
            "gold_count": len(g_list),
            "scorer_match": scorer_match,
            "over_extraction": [],
            "under_extraction": [],
            "corruption": [],
            "hallucination": [],
            "joined_span": [],
            "mislabeling": [],
            "omission": [],
            "mis_extraction": [],
            "unclassified": [],
        }

        # Diagnostic corruption scan runs on the RAW pred field value so leaked
        # tokens and duplicates show up before any normalization.
        raw_pred_value = pred.get(factor)
        if raw_pred_value is not None:
            raw_items = (raw_pred_value if isinstance(raw_pred_value, list)
                         else [raw_pred_value])
            seen = []
            for item in raw_items:
                if not isinstance(item, str):
                    continue
                flags = _detect_corruption(item)
                if item in seen:
                    flags.add("duplicate_list")
                else:
                    seen.append(item)
                if flags:
                    f_report["corruption"].append({
                        "pred": item if len(item) <= 200 else item[:200] + "…",
                        "flags": sorted(flags),
                    })

        # Per-pred checks
        for p in p_list:
            swallowed = _swallowed_gold_spans(p, g_list, sim_threshold)

            # F1-aligned per-pred gate: if some same-factor gold matches this
            # pred at calculate_overlap >= sim_threshold, scorer counts the
            # pred as correct — flagging it as over-extracted / hallucinated /
            # mislabeled would be noise. Use _best_fuzzy (delegates to
            # scorer.calculate_overlap) so the gate matches the F1 rule.
            # Joined Span and Corruption are NOT gated:
            #   - joined-span preds always have |P|<|G| inside is_factor_match,
            #     so they never F1-match by definition;
            #   - corruption is output-quality, independent of correctness.
            pred_f1_match = _best_fuzzy(p, g_list, sim_threshold)[0] >= 0
            # Tracks whether ANY pred-side error category fires for this pred.
            # Preds that fail F1 but don't fall into any bucket are recorded
            # under `unclassified` — a taxonomy-coverage diagnostic.
            classified = False

            # Joined Span vs Over-extraction are mutually exclusive per pred:
            # joined = swallows >= 2 distinct golds; over-extraction = swallows
            # exactly 1 gold AND pred has extra tokens beyond it.
            if len(swallowed) >= 2:
                f_report["joined_span"].append({
                    "pred": p,
                    "gold_spans": [g_list[i] for i in swallowed],
                })
                classified = True
            elif len(swallowed) == 1 and not pred_f1_match:
                g = g_list[swallowed[0]]
                extras = _extra_tokens(p, [g])
                if extras:
                    f_report["over_extraction"].append({
                        "pred": p,
                        "gold": g,
                        "extra_tokens": extras,
                    })
                    classified = True

            # Hallucination: pred is an F1-miss whose content is not present
            # as a substring of source or any same-factor gold. Substring
            # presence is checked at three strictness levels (strict /
            # case-insensitive / normalized); failing any level flags the
            # pred. Gated only by F1-miss — special-token residues like
            # "Nonethought" are now treated as hallucinated content too
            # (model invented them; they're orthogonally also flagged under
            # the corruption diagnostic).
            if not pred_f1_match:
                sub = _substring_levels(p, [source, *g_list])
                if not (sub["strict"] and sub["case_ins"] and sub["normalized"]):
                    # Severity tag: `special_tok` if the pred contains any of
                    # the corruption markers (chat-template leak fragments,
                    # special-token strings) — model "hallucinated" by
                    # spitting back template residue. `organic` otherwise —
                    # genuine invented content. Surfaced in the per-run
                    # severity table to distinguish easy-to-strip corruption
                    # noise from true content failures.
                    corr_markers = _detect_corruption(p) & {
                        "leak_thought", "leak_special", "leak_partial"
                    }
                    f_report["hallucination"].append({
                        "pred": p,
                        "special_tok": bool(corr_markers),
                        **sub,
                    })
                    classified = True

            # Mislabeling: pred fuzzy-matches a sibling factor's gold,
            # AND fails to match its own factor's gold. The same_factor gate
            # is what distinguishes "wrong factor" from "the same string
            # legitimately appears in two factors".
            if not pred_f1_match:
                xref_hits = []
                for sib in siblings.get(factor, []):
                    sib_g = gold_factors.get(sib, [])
                    if not sib_g:
                        continue
                    idx, score = _best_fuzzy(p, sib_g, sim_threshold)
                    if idx >= 0:
                        xref_hits.append({
                            "sibling_factor": sib,
                            "matched_gold": sib_g[idx],
                            "fuzzy": round(score, 4),
                        })
                if xref_hits:
                    f_report["mislabeling"].append({
                        "pred": p,
                        "matches": xref_hits,
                    })
                    classified = True

            # Under-extraction: pred sits inside a same-factor gold and is
            # strictly shorter ("arm" vs "left arm", "earth" vs "earthwork").
            # Different from over-extraction — model picked the right span but
            # cut it short. "Shorter" is asserted at the token level OR the
            # normalized-character level so cases where pred/gold collapse to
            # one token after punctuation stripping (e.g. "earth-excavation")
            # still get caught. Only meaningful when the pred doesn't already
            # F1-match — otherwise "shorter than the canonical gold" is noise.
            under_match = None
            if not pred_f1_match:
                p_toks = _tokens(p)
                p_norm = _normalize(p)
                for g in g_list:
                    if not _pred_within_gold(p, g):
                        continue
                    g_toks = _tokens(g)
                    g_norm = _normalize(g)
                    if (len(p_toks) < len(g_toks)
                            or len(p_norm) < len(g_norm)):
                        under_match = g
                        break
                if under_match is not None:
                    f_report["under_extraction"].append({
                        "pred": p,
                        "gold": under_match,
                        "missing_tokens": [
                            t for t in _tokens(under_match) if t not in set(p_toks)
                        ],
                    })
                    classified = True

            # Mis-extraction: F1-miss pred that's grounded in source text but
            # doesn't fit any of the above categories. The model picked up a
            # real phrase from `acc_txt` that doesn't correspond to any
            # annotated factor at this level — typically a different
            # instance, different scope, or wrong-field copy of source text
            # (e.g. picked "shovel" when annotator picked "lifeline").
            # Distinguished from hallucination, which fires only when pred is
            # NOT in source.
            if not pred_f1_match and not classified:
                p_norm = _normalize(p)
                if p_norm and p_norm in _normalize(source):
                    f_report["mis_extraction"].append({"pred": p})
                    classified = True

            # Unclassified: residual F1-miss preds that no category caught —
            # not even mis_extraction. With mis_extraction enabled this should
            # be near zero; non-zero values surface real taxonomy gaps worth
            # examining. Corruption is intentionally NOT counted here
            # (output-quality flag, orthogonal to F1 attribution).
            if not pred_f1_match and not classified:
                f_report["unclassified"].append({"pred": p})

        # Omission (gold-driven, per-factor): for each gold span, check whether
        # any pred — same factor OR any sibling factor — fuzzy-matches it at
        # >= sim_threshold. If none does, the gold was missed: a recall-side
        # failure that the pred-driven categories above can't surface.
        #
        # Uses _best_fuzzy → calculate_overlap as the per-pair primitive, the
        # same comparator scorer uses internally. This is a relaxed greedy
        # check, NOT scorer's strict 1-to-1 bijection (which is bundle-level
        # only and reported separately via the f1_match column). Recording
        # per-gold attribution is necessarily relaxed — scorer's strict rule
        # has no per-span notion of "this gold was missed".
        for g in g_list:
            if _best_fuzzy(g, p_list, sim_threshold)[0] >= 0:
                continue  # same-factor pred matched this gold
            rescued = False
            for sib in siblings.get(factor, []):
                sib_p = pred_factors.get(sib, [])
                if sib_p and _best_fuzzy(g, sib_p, sim_threshold)[0] >= 0:
                    rescued = True
                    break
            if rescued:
                continue  # extracted into a sibling — counted as mislabeling
            f_report["omission"].append({"gold": g})

        report["factors"][factor] = f_report

    return report


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_KINDS = ("over_extraction", "under_extraction", "corruption",
          "hallucination", "joined_span", "mislabeling", "omission",
          "mis_extraction", "unclassified")


def aggregate(reports):
    summary = {
        "n_records": len(reports),
        "totals": defaultdict(int),
        "by_factor": defaultdict(lambda: defaultdict(int)),
        "by_acc_type": defaultdict(lambda: defaultdict(int)),
    }

    for r in reports:
        atype = r.get("accident_type") or "unknown"
        for fac, fr in r["factors"].items():
            summary["totals"]["pred_count"] += fr["pred_count"]
            summary["totals"]["gold_count"] += fr["gold_count"]
            summary["by_factor"][fac]["pred_count"] += fr["pred_count"]
            summary["by_factor"][fac]["gold_count"] += fr["gold_count"]

            # Scorer-aligned F1 bundle status: each (record, factor) is one
            # "bundle"; count how many would pass scorer's is_factor_match.
            # `f1_bundles` is the denominator, `f1_matches` the numerator —
            # ratio = factor-bundle-level F1 retention.
            summary["totals"]["f1_bundles"] += 1
            summary["by_factor"][fac]["f1_bundles"] += 1
            if fr.get("scorer_match"):
                summary["totals"]["f1_matches"] += 1
                summary["by_factor"][fac]["f1_matches"] += 1

            for kind in _KINDS:
                n = len(fr.get(kind, []))
                summary["totals"][kind] += n
                summary["by_factor"][fac][kind] += n
                summary["by_acc_type"][atype][kind] += n

            # Hallucination severity (nested: strict ⊆ case_ins ⊆ normalized).
            #   true   — normalized fails (model invented content not in src/gold)
            #   punct  — normalized passes but case_ins fails (whitespace/punct)
            #   case   — case_ins passes but strict fails (just casing)
            # Orthogonal cut: special_tok vs organic — does the pred contain
            # chat-template/special-token markers? Lets us split "low-severity
            # corruption residue" from "high-severity invented content".
            for h in fr.get("hallucination", []):
                if not h.get("normalized"):
                    bucket = "true"
                elif not h.get("case_ins"):
                    bucket = "punct"
                else:
                    bucket = "case"
                summary["totals"]["halluc_" + bucket] += 1
                summary["by_factor"][fac]["halluc_" + bucket] += 1
                origin = "special_tok" if h.get("special_tok") else "organic"
                summary["totals"]["halluc_" + origin] += 1
                summary["by_factor"][fac]["halluc_" + origin] += 1

            # Corruption flag counts across the run.
            for corr in fr.get("corruption", []):
                for flag in corr.get("flags", []):
                    summary["totals"]["corr_" + flag] += 1
                    summary["by_factor"][fac]["corr_" + flag] += 1

    summary["totals"] = dict(summary["totals"])
    summary["by_factor"] = {k: dict(v) for k, v in summary["by_factor"].items()}
    summary["by_acc_type"] = {k: dict(v) for k, v in summary["by_acc_type"].items()}
    return summary


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _analyze_split_worker(args):
    """Process-pool worker: analyze every pred record in ONE (k-shot, split)
    and return its list of report dicts. Defined at module level so it
    pickles cleanly under `spawn` on Windows / macOS.

    Each chunk is independent (loads its own preds + golds, runs
    `analyze_record` over the pred records), so the (k × split) grid
    parallelizes naturally."""
    (k, split, pred_path, gold_path, structure, classes_map,
     sim_threshold, norm, skip_factors) = args
    try:
        from TextEE.utils import convert_predictions
    except ImportError:
        from utils import convert_predictions

    convert_config = SimpleNamespace(key_map={
        "structure": structure or {},
        "task": {"classification": {"classes": classes_map or {}}},
    })

    preds = _load_jsonl(pred_path)
    if preds and ("entity_mentions" in preds[0] or "event_mentions" in preds[0]):
        src_fmt = "IE"
    else:
        src_fmt = "ACH"
    preds = convert_predictions(preds, src_fmt, "AC", convert_config)

    golds = _load_jsonl(gold_path)
    golds = convert_predictions(golds, "ACH", "AC", convert_config)
    gold_by_id = {str(g.get("id", g.get("doc_id"))): g for g in golds}

    reports = []
    for p in preds:
        pid = str(p.get("id", p.get("doc_id")))
        g = gold_by_id.get(pid)
        if g is None:
            continue
        rep = analyze_record(p, g, structure,
                              sim_threshold=sim_threshold,
                              skip_factors=skip_factors,
                              norm=norm)
        rep["split"] = split
        reports.append(rep)
    return k, reports


def analyze_combination(out_path, model, dataset, task, gold_root, structure,
                        sim_threshold=0.8, splits=None, shots=None, log_dir=None,
                        classes_map=None, auto_subdir=False, write_compare=False,
                        skip_factors_by_shot=None, base_skip=None,
                        norm=False, variant="", max_workers=None):
    """Run error analysis for one (model, dataset, task) across all splits per k.

    `classes_map`, when provided, supplies the acc_type enum used to detect
    supervised IE-format predictions and convert them to flat AC dicts via
    `utils.convert_predictions`. Without it, supervised pred files (which
    carry `entity_mentions` / `event_mentions` instead of factor keys) would
    look "empty" to analyze_record and produce noise-only reports.

    `log_dir` controls where outputs land:
      * unset                  → next to the predictions (`out_path`).
      * set, auto_subdir=False → `log_dir` used verbatim.
      * set, auto_subdir=True  → wrapped to
            `<log_dir>/<run_id>/<task>/<dataset>/<model>/`
        where `<run_id>` comes from `make_run_id("error_analysis")` (shared
        per-process). Use this from sweep callers (run.py -a error) so every
        combination lands under one timestamped folder.

    `write_compare`, when True and ≥2 k-shots produced output, also writes a
    cross-k comparison log (compare_<model>_<dataset>_<task>.log) into the
    same log_dir.

    `skip_factors_by_shot`, when provided, is a `{shot_int: [factor, ...]}`
    map of underfilled factors to exclude from the per-record analysis at
    each shot count — matches the skip vector LLM evaluation applies via
    `skip_threshold`. Recorded into the top-level output JSON as
    `skip_factors` for traceability.

    `base_skip` is the task-level `skip_columns` list from the global
    config (structural columns like `accident_report` / `id`). Together
    with the key map's classification columns (`classes_map` keys) it is
    excluded uniformly for every model and shot, keeping the error
    taxonomy extraction-only.

    Pred files are matched by their canonical `pred_*_<dataset>_<task>_
    split{N}_k{K}.json` name. If a (model, dataset, task) combination
    matches no pred files at all, `analyze_combination` raises rather than
    silently returning an empty result.

    Returns the list of written errors_*.json paths.
    """
    # Synthetic config for convert_predictions: it reads `config.key_map` to
    # get structure + acc_types. Reconstructed from the params analyze_combination
    # already accepts so callers don't have to pass the full key_map dict.
    convert_config = SimpleNamespace(key_map={
        "structure": structure or {},
        "task": {"classification": {"classes": classes_map or {}}},
    })

    if log_dir and auto_subdir:
        log_dir = os.path.join(
            log_dir, make_run_id("error_analysis"), task, dataset, model)
    log_dir = log_dir or out_path
    os.makedirs(log_dir, exist_ok=True)
    task_lc = task.lower()

    if splits is None:
        split_dirs = sorted(glob.glob(os.path.join(gold_root, dataset, "split*")))
        splits = []
        for d in split_dirs:
            m = re.match(r"split(\d+)$", os.path.basename(d))
            if m:
                splits.append(int(m.group(1)))

    if shots is None:
        # Match any pred_<anything>_<dataset>_<task>_split{N}_k{K}.json — the
        # `<anything>` is the trainer-side model name (often model_alts[model])
        # which we don't have here. out_path is already model-scoped so we
        # don't need to over-constrain on the file's model prefix.
        shot_pat = re.compile(
            rf"^pred_.+_{re.escape(dataset)}_{re.escape(task_lc)}"
            rf"_split\d+_k(\d+)\.json$"
        )
        shot_set = set()
        for f in os.listdir(out_path):
            m = shot_pat.match(f)
            if m:
                shot_set.add(int(m.group(1)))
        shots = sorted(shot_set)

    skip_map = skip_factors_by_shot or {}
    # Uniform skip set, identical for every model family:
    #   * `base_skip` — task-level `skip_columns` from the global config
    #     (structural columns like `accident_report` / `id`).
    #   * classification columns from the key map (`classes_map` keys) —
    #     the error taxonomy is extraction-only. Classification factors are
    #     closed-vocabulary labels, so their misses would all land in
    #     `hallucination`/`omission` as an artifact of the substring checks.
    #     Every trainer emits them nowadays (TagPrime included), hence no
    #     per-format sniffing or special-casing.
    always_skip = set(base_skip or ()) | set(classes_map or ())
    # Build the (k, split, pred_path, gold_path, ...) work items in the main
    # process — skip-factor resolution is cheap and lets workers stay
    # narrowly scoped to "analyze every record in one chunk".
    shot_skip = {}      # k -> resolved skip_factors list
    work_items = []
    pred_matches_found = 0
    for k in shots:
        skip_factors = list(skip_map.get(k, ()) or skip_map.get(str(k), ()) or ())
        skip_factors = sorted(set(skip_factors) | always_skip)
        shot_skip[k] = skip_factors

        for split in splits:
            matches = glob.glob(os.path.join(
                out_path, f"pred_*_{dataset}_{task_lc}_split{split}_k{k}.json"))
            if not matches:
                # Expected for shots a model didn't produce (e.g. supervised
                # models only ever have k=0) — not an error by itself. The
                # check below after the loop catches the case that matters:
                # the canonical name matching NOTHING anywhere.
                continue
            pred_matches_found += 1
            pred_path = matches[0]
            gold_path = os.path.join(gold_root, dataset, f"split{split}", "test.json")
            if not os.path.exists(gold_path):
                logger.warning("missing gold: %s", gold_path)
                continue
            work_items.append((k, split, pred_path, gold_path,
                                structure, classes_map, sim_threshold, norm,
                                skip_factors))

    if shots and splits and not pred_matches_found:
        raise FileNotFoundError(
            f"analyze_combination: no pred_*.json files matched for "
            f"model={model!r} dataset={dataset!r} task={task!r} under "
            f"canonical filename tokens dataset={dataset!r} "
            f"task={task_lc!r} in {out_path!r} "
            f"(tried shots={shots}, splits={splits})."
        )

    # Run the per-(shot, split) chunks in parallel, then group reports by
    # shot. Falls back to serial when there's nothing to parallelize.
    reports_by_shot = defaultdict(list)
    if work_items:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        if max_workers is None:
            max_workers = min(mp.cpu_count() or 1, len(work_items), 8)
        if max_workers <= 1 or len(work_items) == 1:
            for item in work_items:
                k, reps = _analyze_split_worker(item)
                reports_by_shot[k].extend(reps)
        else:
            logger.info("\U0001f9ee Analyzing %d (shot, split) chunk(s) on %d workers...",
                         len(work_items), max_workers)
            chunksize = max(1, len(work_items) // (max_workers * 4))
            pool = ProcessPoolExecutor(max_workers=max_workers)
            try:
                for k, reps in pool.map(_analyze_split_worker, work_items,
                                          chunksize=chunksize):
                    reports_by_shot[k].extend(reps)
            finally:
                pool.shutdown(wait=True)

    written = []
    for k in shots:
        all_reports = reports_by_shot.get(k) or []
        if not all_reports:
            continue
        summary = aggregate(all_reports)
        out = {
            "model": model,
            "dataset": dataset,
            "task": task,
            "k_shot": k,
            "sim_threshold": sim_threshold,
            "norm": bool(norm),
            "skip_factors": sorted(shot_skip.get(k) or ()),
            "summary": summary,
            "records": all_reports,
        }
        # Suffix the variant so multi-variant -a error sweeps under a shared
        # --log_path don't overwrite each other's per-k JSON / log files.
        # Empty variant → unchanged filenames (backwards-compatible).
        v_suf = f"_{variant}" if variant else ""
        out_name = f"errors_{model}{v_suf}_{dataset}_{task_lc}_k{k}.json"
        out_path_full = os.path.join(log_dir, out_name)
        with open(out_path_full, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        written.append(out_path_full)

    if write_compare and len(written) >= 2:
        v_suf = f"_{variant}" if variant else ""
        compare_path = os.path.join(
            log_dir, f"compare_{model}{v_suf}_{dataset}_{task_lc}.log"
        )
        write_comparison_log(written, compare_path)

    return written


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _format_table(title, headers, rows, footer=None, total_row=None):
    """Render a fixed-width text table.

    `total_row`, when supplied, is rendered after a separator using the same
    column widths as the data rows — visually a "TOTAL" footer line that still
    aligns with the columns above. `footer` is a free-form string appended
    after another separator (legacy mechanism, kept for callers that pass a
    summary sentence rather than a row).
    """
    if not rows and total_row is None:
        return f"\n=== {title} ===\n  (no data)\n"
    str_rows = [[("" if c is None else str(c)) for c in r] for r in (rows or [])]
    str_total = (
        [("" if c is None else str(c)) for c in total_row]
        if total_row is not None else None
    )
    all_str_rows = str_rows + ([str_total] if str_total is not None else [])
    widths = [
        max(len(str(headers[i])),
            max((len(r[i]) for r in all_str_rows), default=0))
        for i in range(len(headers))
    ]
    def fmt(row):
        return " | ".join(str(c).ljust(w) for c, w in zip(row, widths))
    sep = "-+-".join("-" * w for w in widths)
    lines = [
        "",
        f"=== {title} ===",
        fmt(headers),
        sep,
    ]
    lines.extend(fmt(r) for r in str_rows)
    if str_total is not None:
        lines.append(sep)
        lines.append(fmt(str_total))
    if footer:
        lines.append(sep)
        lines.append(footer)
    return "\n".join(lines)


def _pct(num, denom):
    return f"{(100.0 * num / denom):.1f}%" if denom else "—"


def _count_pct(n, denom):
    """Render a cell as 'N (P.P%)' for nonzero N, empty string for zero.
    Used everywhere the count+pct convention is unified so wide tables don't
    drown in zero placeholders, while informative cells carry their share."""
    if not n:
        return ""
    if not denom:
        return str(n)
    return f"{n} ({_pct(n, denom)})"


def _render_overall(data):
    """Three OVERALL tables: pred-side F1 attribution (share of pred),
    gold-side F1 miss (share of gold), and corruption (diagnostic, share of
    pred). Format mirrors the per-error-type tables: model/kshot/task as
    leading columns, each data cell as 'count (pct%)' inline."""
    s = data["summary"]
    t = s["totals"]
    pred = t.get("pred_count", 0)
    gold = t.get("gold_count", 0)
    model = data.get("model", "")
    task  = data.get("task", "")
    kshot = str(data.get("k_shot", ""))
    over_ext  = t.get("over_extraction", 0)
    under_ext = t.get("under_extraction", 0)
    halluc    = t.get("hallucination", 0)
    joined    = t.get("joined_span", 0)
    mislabel  = t.get("mislabeling", 0)
    mis_ext   = t.get("mis_extraction", 0)
    unclass   = t.get("unclassified", 0)
    omit      = t.get("omission", 0)
    corrupt   = sum(t.get("corr_" + f, 0) for f in _CORR_FLAGS)
    f1_pct    = _pct(t.get("f1_matches", 0), t.get("f1_bundles", 0))

    pred_headers = ["model", "kshot", "task", "records", "pred",
                    "over_ext", "under_ext", "halluc", "joined", "mislabel",
                    "mis_ext", "unclass", "f1_match"]
    pred_row = [
        model, kshot, task, s["n_records"], pred,
        _count_pct(over_ext, pred), _count_pct(under_ext, pred),
        _count_pct(halluc, pred), _count_pct(joined, pred),
        _count_pct(mislabel, pred), _count_pct(mis_ext, pred),
        _count_pct(unclass, pred), f1_pct,
    ]
    table_pred = _format_table(
        "OVERALL — pred-side F1 misses (share of pred)",
        pred_headers, [pred_row],
    )

    gold_headers = ["model", "kshot", "task", "records", "gold", "omit"]
    gold_row = [
        model, kshot, task, s["n_records"], gold,
        _count_pct(omit, gold),
    ]
    table_gold = _format_table(
        "OVERALL — gold-side F1 miss (share of gold)",
        gold_headers, [gold_row],
    )

    corr_headers = ["model", "kshot", "task", "records", "pred", "corruption"]
    corr_row = [
        model, kshot, task, s["n_records"], pred,
        _count_pct(corrupt, pred),
    ]
    table_corr = _format_table(
        "OVERALL — corruption (diagnostic, share of pred)",
        corr_headers, [corr_row],
    )

    return "\n".join([table_pred, table_gold, table_corr])


def _render_over_extraction(data):
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "pred", "gold", "over_ext", "% of pred"]
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        n = f.get("over_extraction", 0)
        if not n:
            continue
        rows.append([fac, f.get("pred_count", 0), f.get("gold_count", 0),
                     n, _pct(n, f.get("pred_count", 0))])
    total_row = [
        "TOTAL",
        t.get("pred_count", 0), t.get("gold_count", 0),
        t.get("over_extraction", 0),
        _pct(t.get("over_extraction", 0), t.get("pred_count", 0)),
    ]
    return _format_table(
        "OVER-EXTRACTION (pred swallows 1 gold + has extras)",
        headers, rows, total_row=total_row,
    )


def _render_corruption(data):
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "pred"] + list(_CORR_FLAGS)
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        per_factor_total = sum(f.get("corr_" + flag, 0) for flag in _CORR_FLAGS)
        if not per_factor_total:
            continue
        rows.append(
            [fac, f.get("pred_count", 0)]
            + [f.get("corr_" + flag, 0) for flag in _CORR_FLAGS]
        )
    total_corr = sum(t.get("corr_" + flag, 0) for flag in _CORR_FLAGS)
    if not rows and not total_corr:
        return ""
    total_row = (
        ["TOTAL", t.get("pred_count", 0)]
        + [t.get("corr_" + flag, 0) for flag in _CORR_FLAGS]
    )
    return _format_table(
        "CORRUPTED OUTPUT (special tokens, runaway loops, duplicates)",
        headers, rows, total_row=total_row,
    )


def _render_hallucination(data):
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "pred", "halluc", "true", "punct", "case", "% of pred"]
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        h = f.get("hallucination", 0)
        if not h:
            continue
        rows.append([
            fac, f.get("pred_count", 0), h,
            f.get("halluc_true", 0), f.get("halluc_punct", 0), f.get("halluc_case", 0),
            _pct(h, f.get("pred_count", 0)),
        ])
    total_row = [
        "TOTAL", t.get("pred_count", 0),
        t.get("hallucination", 0),
        t.get("halluc_true", 0), t.get("halluc_punct", 0), t.get("halluc_case", 0),
        _pct(t.get("hallucination", 0), t.get("pred_count", 0)),
    ]
    return _format_table(
        "HALLUCINATION (substring levels: true=normalized-fail, punct=case-ins-fail, case=strict-fail)",
        headers, rows, total_row=total_row,
    )


def _render_joined_span(data):
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "pred", "joined_pred_spans", "% of pred"]
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        j = f.get("joined_span", 0)
        if not j:
            continue
        rows.append([fac, f.get("pred_count", 0), j, _pct(j, f.get("pred_count", 0))])
    total_row = [
        "TOTAL", t.get("pred_count", 0),
        t.get("joined_span", 0),
        _pct(t.get("joined_span", 0), t.get("pred_count", 0)),
    ]
    return _format_table(
        "JOINED SPAN (one pred swallows ≥2 gold spans)",
        headers, rows, total_row=total_row,
    )


def _render_mislabeling(data):
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "pred", "mislabel", "% of pred"]
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        x = f.get("mislabeling", 0)
        if not x:
            continue
        rows.append([fac, f.get("pred_count", 0), x, _pct(x, f.get("pred_count", 0))])
    total_row = [
        "TOTAL", t.get("pred_count", 0),
        t.get("mislabeling", 0),
        _pct(t.get("mislabeling", 0), t.get("pred_count", 0)),
    ]
    return _format_table(
        "MISLABELING (pred matches a sibling factor's gold)",
        headers, rows, total_row=total_row,
    )


def _render_under_extraction(data):
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "pred", "gold", "under_ext", "% of pred"]
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        n = f.get("under_extraction", 0)
        if not n:
            continue
        rows.append([fac, f.get("pred_count", 0), f.get("gold_count", 0),
                     n, _pct(n, f.get("pred_count", 0))])
    total_row = [
        "TOTAL",
        t.get("pred_count", 0), t.get("gold_count", 0),
        t.get("under_extraction", 0),
        _pct(t.get("under_extraction", 0), t.get("pred_count", 0)),
    ]
    return _format_table(
        "UNDER-EXTRACTION (pred is a strict token-subset of a same-factor gold)",
        headers, rows, total_row=total_row,
    )


def _render_omission(data):
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "gold", "omit", "% of gold"]
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        n = f.get("omission", 0)
        if not n:
            continue
        rows.append([fac, f.get("gold_count", 0), n, _pct(n, f.get("gold_count", 0))])
    total_row = [
        "TOTAL", t.get("gold_count", 0),
        t.get("omission", 0),
        _pct(t.get("omission", 0), t.get("gold_count", 0)),
    ]
    return _format_table(
        "OMISSION (gold span missed by all preds at the same level)",
        headers, rows, total_row=total_row,
    )


def _render_mis_extraction(data):
    """F1-miss preds grounded in source but not matching any annotated factor
    at this level — wrong instance, wrong scope, or wrong-field copy."""
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "pred", "mis_ext", "% of pred"]
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        n = f.get("mis_extraction", 0)
        if not n:
            continue
        rows.append([fac, f.get("pred_count", 0), n, _pct(n, f.get("pred_count", 0))])
    total_row = [
        "TOTAL", t.get("pred_count", 0),
        t.get("mis_extraction", 0),
        _pct(t.get("mis_extraction", 0), t.get("pred_count", 0)),
    ]
    return _format_table(
        "MIS-EXTRACTION (F1-miss pred is in source but unmatched at this level)",
        headers, rows, total_row=total_row,
    )


def _render_unclassified(data):
    """Diagnostic table: preds that failed F1 but didn't fall into any of the
    pred-side error categories. High counts signal taxonomy gaps."""
    s = data["summary"]
    by_fac = s["by_factor"]
    t = s["totals"]
    headers = ["factor", "pred", "unclass", "% of pred"]
    rows = []
    for fac in sorted(by_fac):
        f = by_fac[fac]
        n = f.get("unclassified", 0)
        if not n:
            continue
        rows.append([fac, f.get("pred_count", 0), n, _pct(n, f.get("pred_count", 0))])
    total_row = [
        "TOTAL", t.get("pred_count", 0),
        t.get("unclassified", 0),
        _pct(t.get("unclassified", 0), t.get("pred_count", 0)),
    ]
    return _format_table(
        "UNCLASSIFIED (F1-miss pred caught by no pred-side category)",
        headers, rows, total_row=total_row,
    )


def _build_tables_text(data):
    norm_tag = "on" if data.get("norm") else "off"
    header = (f"\n{'#' * 78}\n"
              f"# {data.get('_basename', '')}  "
              f"(model={data['model']}, dataset={data['dataset']}, "
              f"task={data['task']}, k={data['k_shot']}, "
              f"sim={data['sim_threshold']}, norm={norm_tag})\n"
              f"{'#' * 78}")
    sections = [
        header,
        _render_overall(data),
        _render_over_extraction(data),
        _render_under_extraction(data),
        _render_corruption(data),
        _render_hallucination(data),
        _render_joined_span(data),
        _render_mislabeling(data),
        _render_omission(data),
        _render_mis_extraction(data),
        _render_unclassified(data),
    ]
    return "\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Cross-run comparison tables
# ---------------------------------------------------------------------------

def _run_label(data):
    return f"{data['model']}/{data['dataset']}/{data['task']}/k{data['k_shot']}"


# Variant tag stripped from any model name before display (matches what
# data_analyzer does). `_c<N>` is the supervised-variant suffix run.py
# tacks onto pool keys to keep multi-variant runs apart.
_ERROR_VARIANT_RE = re.compile(r"_c\d+$")


def _err_disp_model(m, model_aliases=None):
    if model_aliases and m in model_aliases:
        return model_aliases[m]
    return _ERROR_VARIANT_RE.sub("", m or "")


def _esc_latex_err(s):
    return (str(s)
            .replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("$", r"\$"))


def _count_pct_latex(n, denom):
    """LaTeX version of `_count_pct`. Empty cell for zero; bare integer when
    denom is zero; otherwise `N (P.P\%)` with the percent sign escaped."""
    if not n:
        return ""
    if not denom:
        return str(n)
    pct = (100.0 * n) / denom
    return f"{n} ({pct:.1f}\\%)"


def _pct_latex(num, denom):
    return f"{(100.0 * num / denom):.1f}\\%" if denom else "--"


def _pct_only(n, denom):
    """Cell showing JUST the percentage as a number (no count, no '%' sign).
    Used by the by-model error-composition table where every row sums to
    100 so the bare percent is unambiguous."""
    if not denom:
        return "--"
    return f"{(100.0 * n / denom):.1f}"


def _aggregate_per_model(runs):
    """Roll the run list up into one bucket per model — shared by the text
    and LaTeX renderers so both views agree on the same numbers."""
    per_model = {}
    for d in runs:
        m = d.get("model", "")
        t = d["summary"]["totals"]
        n_records = d["summary"].get("n_records", 0)
        b = per_model.setdefault(m, {
            "n_runs": 0, "n_records": 0,
            "pred": 0, "gold": 0,
            "over_ext": 0, "under_ext": 0, "halluc": 0, "joined": 0,
            "mislabel": 0, "mis_ext": 0, "unclass": 0, "omit": 0,
            "corrupt": 0, "f1_matches": 0, "f1_bundles": 0,
        })
        b["n_runs"]      += 1
        b["n_records"]   += n_records
        b["pred"]        += t.get("pred_count", 0)
        b["gold"]        += t.get("gold_count", 0)
        b["over_ext"]    += t.get("over_extraction", 0)
        b["under_ext"]   += t.get("under_extraction", 0)
        b["halluc"]      += t.get("hallucination", 0)
        b["joined"]      += t.get("joined_span", 0)
        b["mislabel"]    += t.get("mislabeling", 0)
        b["mis_ext"]     += t.get("mis_extraction", 0)
        b["unclass"]     += t.get("unclassified", 0)
        b["omit"]        += t.get("omission", 0)
        b["corrupt"]     += sum(t.get("corr_" + flag, 0) for flag in _CORR_FLAGS)
        b["f1_matches"]  += t.get("f1_matches", 0)
        b["f1_bundles"]  += t.get("f1_bundles", 0)
    return per_model


def _ordered_present(model_order, present):
    """Order `present` model keys by their position in `model_order` (e.g.
    `list(model_alts.keys())` from the global config), appending any not
    found there in ascending name order. Keeps row order sourced from the
    same config the short display names come from, instead of a second
    hardcoded ordering."""
    order = list(model_order or ())
    present = set(present)
    ordered = [m for m in order if m in present]
    ordered += sorted(m for m in present if m not in order)
    return ordered


def _by_model_row_order(per_model, supervised_models=None):
    """Row order for the by-model tables: supervised models first, then
    LLMs, each group in ascending name order. Group membership comes from
    the global mapping's `model_type.supervised` list (threaded down from
    run.py); when it's absent the order degrades to plain alphabetical."""
    sup = set(supervised_models or ())
    return sorted(per_model.keys(), key=lambda m: (0 if m in sup else 1, m.lower()))


def _render_by_model_summary_latex(runs, model_aliases=None,
                                   supervised_models=None):
    """LaTeX twin of `_render_by_model_summary`. Same aggregation, same
    column layout, but rendered as a booktabs table* block suitable for a
    paper. Returns the empty string when there's nothing to show.

    `model_aliases` is the same {full_name -> short_alias} map data_analyzer
    uses — pass it through from run.py to keep labels short."""
    if not runs:
        return ""
    per_model = _aggregate_per_model(runs)
    if not per_model:
        return ""

    # --- TABLE A: totals (runs / pred / gold / F1) ---
    totals_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rrr r}",
        r"\toprule",
        r"\textbf{Model} & \textbf{runs} & \textbf{pred} & \textbf{gold} "
        r"& \textbf{F1} \\",
        r"\midrule",
    ]
    for m in _by_model_row_order(per_model, supervised_models):
        b = per_model[m]
        totals_lines.append(" & ".join([
            _esc_latex_err(_err_disp_model(m, model_aliases)),
            str(b["n_runs"]), str(b["pred"]), str(b["gold"]),
            _pct_latex(b["f1_matches"], b["f1_bundles"]),
        ]) + r" \\")
    sums_t = {k: sum(b[k] for b in per_model.values())
              for k in ("n_runs", "pred", "gold", "f1_matches", "f1_bundles")}
    totals_lines.append(r"\midrule")
    totals_lines.append(" & ".join([
        r"\textbf{TOTAL}", str(sums_t["n_runs"]),
        str(sums_t["pred"]), str(sums_t["gold"]),
        _pct_latex(sums_t["f1_matches"], sums_t["f1_bundles"]),
    ]) + r" \\")
    totals_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-model run counts, pred/gold denominators, and the "
        r"pooled span match rate (F1). Supervised models contribute their "
        r"$k$=0 runs only.}",
        r"\label{tab:error-by-model-totals}",
        r"\end{table}",
    ]

    # --- TABLE B: error-type composition (rows sum to 100%) ---
    # Cells are bare percentages (no count, no '%' sign) — the caption /
    # column headers make the unit explicit and the row sums are 100. The
    # `_ext` suffixes are dropped for compactness.
    comp_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        # 1 model col + 6 error-type cells = 7 cols
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"\textbf{Model} "
        r"& \textbf{over} & \textbf{under} & \textbf{halluc} "
        r"& \textbf{joined} & \textbf{mislabel} & \textbf{mis} \\",
        r"\midrule",
    ]
    for m in _by_model_row_order(per_model, supervised_models):
        b = per_model[m]
        # Denominator = THIS model's total error count over the six main
        # pred-side categories. Each cell shows its share of that model's
        # errors → row reads as the composition / dominant failure mode
        # and the six percentages sum to 100.
        d = _row_error_total(b)
        comp_lines.append(" & ".join([
            _esc_latex_err(_err_disp_model(m, model_aliases)),
            _pct_only(b["over_ext"],  d),
            _pct_only(b["under_ext"], d),
            _pct_only(b["halluc"],    d),
            _pct_only(b["joined"],    d),
            _pct_only(b["mislabel"],  d),
            _pct_only(b["mis_ext"],   d),
        ]) + r" \\")
    comp_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-model error-type composition, rolled up across "
        r"$k$-shots, tasks, and datasets. Cells are percentages of "
        r"\emph{this model's} total errors across the six main pred-side "
        r"categories (over-extraction, under-extraction, hallucination, "
        r"joined span, mislabeling, mis-extraction); rows sum to 100.}",
        r"\label{tab:error-by-model}",
        r"\end{table*}",
    ]
    return "\n".join(totals_lines + [""] + comp_lines)


def _render_grouped_composition_latex(runs, model_order=None, model_aliases=None,
                                      supervised_models=None, caption=None,
                                      label="tab:error"):
    """Single error-composition table with three row blocks: Supervised
    Models, then LLMs run under JHE, then LLMs run under IHE.

    Unlike `_render_by_model_summary_latex` (which pools every task
    together per model), this keeps JHE and IHE numbers separate per model
    — the same model can have very different failure profiles on the two
    tasks, so merging them would hide that.

    `model_order`, when provided, is the row/display order — pass
    `list(model_alts.keys())` from the global config so ordering comes
    from the same source of truth as the short names (`model_aliases`),
    instead of a second hardcoded list here.
    """
    if not runs:
        return ""
    sup = set(supervised_models or ())

    sup_runs, jhe_runs, ihe_runs = [], [], []
    for d in runs:
        m = d.get("model", "")
        task_lc = str(d.get("task", "")).lower()
        if m in sup:
            sup_runs.append(d)
        elif task_lc == "jhe":
            jhe_runs.append(d)
        elif task_lc == "ihe":
            ihe_runs.append(d)

    sections = [
        ("Supervised Models", _aggregate_per_model(sup_runs)),
        ("Joint Hierarchical Extraction (JHE)", _aggregate_per_model(jhe_runs)),
        ("Individual Hierarchical Extraction (IHE)", _aggregate_per_model(ihe_runs)),
    ]
    sections = [(title, per_model) for title, per_model in sections if per_model]
    if not sections:
        return ""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Model}",
        r"& \textbf{Over}",
        r"& \textbf{Under}",
        r"& \textbf{Halluc.}",
        r"& \textbf{Joined}",
        r"& \textbf{Mis-la}",
        r"& \textbf{Mis-ex} \\",
    ]
    for title, per_model in sections:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{title}}}}} \\")
        for m in _ordered_present(model_order, per_model.keys()):
            b = per_model[m]
            d = _row_error_total(b)
            lines.append(_esc_latex_err(_err_disp_model(m, model_aliases)))
            lines.append("& " + " & ".join([
                _pct_only(b["over_ext"],  d),
                _pct_only(b["under_ext"], d),
                _pct_only(b["halluc"],    d),
                _pct_only(b["joined"],    d),
                _pct_only(b["mislabel"],  d),
                _pct_only(b["mis_ext"],   d),
            ]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{" + (caption or "Exact-match error distributions by "
                         "extraction strategy.") + "}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# Error-type columns shown in the by-model summary AND summed as the
# per-row denominator. Only the six "main" pred-side error categories are
# kept so the row percentages add up to 100% — answering "among the model's
# errors, which one is the most common?". `unclass` (near-zero residual),
# `omit` (gold-side; different denominator), and `corrupt` (diagnostic)
# are intentionally excluded.
_BY_MODEL_ERROR_KEYS = (
    "over_ext", "under_ext", "halluc", "joined", "mislabel", "mis_ext",
)


def _row_error_total(b):
    """Sum of every error-category count in one model's bucket — used as the
    per-row denominator for the by-model summary. Answers 'among THIS
    model's errors, what share does each category have?'."""
    return sum(b.get(k, 0) for k in _BY_MODEL_ERROR_KEYS)


def _render_by_model_summary(runs, supervised_models=None):
    """Two-table aggregate:

    1. BY-MODEL SUMMARY — rows = unique models, cols = the six main
       pred-side error categories + F1. Each cell shows the category's
       share of THIS model's total errors across those six categories,
       so each row's percentages sum to 100%.

    2. BY-MODEL TOTALS — companion table with run counts and the
       pred / gold denominators. Pulled out of the main table so the
       error-composition view stays focused on the percentages."""
    if not runs:
        return ""

    per_model = _aggregate_per_model(runs)

    # --- 1. BY-MODEL TOTALS — counts only ---
    totals_headers = ["model", "runs", "pred", "gold", "f1_match"]
    totals_rows = []
    for m in _by_model_row_order(per_model, supervised_models):
        b = per_model[m]
        totals_rows.append([
            m, b["n_runs"], b["pred"], b["gold"],
            _pct(b["f1_matches"], b["f1_bundles"]),
        ])
    sums_t = {k: sum(b[k] for b in per_model.values())
              for k in ("n_runs", "pred", "gold", "f1_matches", "f1_bundles")}
    totals_total = [
        "TOTAL", sums_t["n_runs"], sums_t["pred"], sums_t["gold"],
        _pct(sums_t["f1_matches"], sums_t["f1_bundles"]),
    ]
    totals_table = _format_table(
        "BY-MODEL TOTALS — record counts + pooled F1",
        totals_headers, totals_rows, total_row=totals_total,
    )

    # --- 2. BY-MODEL SUMMARY — error-type composition (rows sum to 100%) ---
    # Cells are bare percentages; the column headers + caption make it
    # clear the unit is "% of this model's errors". `_ext` suffixes
    # dropped for compactness.
    headers = ["model", "over", "under", "halluc", "joined",
               "mislabel", "mis"]
    rows = []
    for m in _by_model_row_order(per_model, supervised_models):
        b = per_model[m]
        d = _row_error_total(b)
        rows.append([
            m,
            _pct_only(b["over_ext"],  d),
            _pct_only(b["under_ext"], d),
            _pct_only(b["halluc"],    d),
            _pct_only(b["joined"],    d),
            _pct_only(b["mislabel"],  d),
            _pct_only(b["mis_ext"],   d),
        ])
    composition_table = _format_table(
        "BY-MODEL SUMMARY — error-type composition (% of each model's errors; rows sum to 100)",
        headers, rows,
    )

    return totals_table + "\n" + composition_table


def aggregate_tables(json_paths, label_fn=_run_label, supervised_models=None):
    """Build comparison tables across multiple errors_*.json bundles.

    Returns a single multi-line text block. Natural use is "across k for one
    (model, dataset, task)" via write_comparison_log; also handles
    cross-model / cross-dataset comparisons when given heterogeneous paths.
    """
    runs = []
    for p in json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception as exc:
            logger.warning("could not read %s: %s", p, exc)
    if not runs:
        return ""

    # Sort by (task, k_shot, model) so the per-error-type tables list every
    # model side-by-side at each (task, kshot) experimental point. That makes
    # model comparison a single visual sweep: pick a kshot+task, scan the
    # adjacent rows. dataset is kept as a tertiary key so multi-dataset
    # sweeps still group cleanly within a task.
    runs.sort(key=lambda d: (
        str(d.get("task", "")),
        int(d.get("k_shot", 0)),
        str(d.get("model", "")),
        str(d.get("dataset", "")),
    ))

    sections = [
        f"\n{'#' * 78}\n"
        f"# COMPARISON across {len(runs)} runs\n"
        f"{'#' * 78}"
    ]

    # 0. BY-MODEL SUMMARY — single table, rows = unique models, columns =
    # error-type categories. Counts are rolled up across every run for that
    # model (i.e. summed across kshot × task × dataset). Pred-side categories
    # use the model's total pred_count as their denominator; omission uses
    # gold_count; corruption uses pred_count (diagnostic). Lives at the top
    # because most readers want the model-vs-model headline before drilling
    # into per-(kshot, task) breakdowns.
    sections.append(_render_by_model_summary(runs, supervised_models))

    # 1. OVERALL — three tables to keep denominators coherent:
    #    a) pred-side F1 attribution + f1_match  (share of pred)
    #    b) gold-side F1 miss / omission         (share of gold)
    #    c) corruption diagnostic                 (share of pred, non-gated)
    # All three use the standard layout: model/kshot/task as leading cols,
    # data cells as 'count (pct%)' inline. TOTAL row sums across runs.
    pred_headers = ["model", "kshot", "task", "records", "pred",
                    "over_ext", "under_ext", "halluc", "joined", "mislabel",
                    "mis_ext", "unclass", "f1_match"]
    gold_headers = ["model", "kshot", "task", "records", "gold", "omit"]
    corr_headers = ["model", "kshot", "task", "records", "pred", "corruption"]
    pred_rows, gold_rows, corr_rows = [], [], []
    sum_records = sum_pred = sum_gold = 0
    sum_over_ext = sum_under_ext = sum_halluc = sum_joined = 0
    sum_mislabel = sum_mis_ext = sum_unclass = sum_omit = sum_corrupt = 0
    sum_f1_matches = sum_f1_bundles = 0
    for d in runs:
        s, t = d["summary"], d["summary"]["totals"]
        pred = t.get("pred_count", 0)
        gold = t.get("gold_count", 0)
        over_ext  = t.get("over_extraction", 0)
        under_ext = t.get("under_extraction", 0)
        corrupt   = sum(t.get("corr_" + flag, 0) for flag in _CORR_FLAGS)
        halluc    = t.get("hallucination", 0)
        joined    = t.get("joined_span", 0)
        mislabel  = t.get("mislabeling", 0)
        omit      = t.get("omission", 0)
        mis_ext   = t.get("mis_extraction", 0)
        unclass   = t.get("unclassified", 0)
        f1_pct    = _pct(t.get("f1_matches", 0), t.get("f1_bundles", 0))
        m, k, tk = d.get("model", ""), str(d.get("k_shot", "")), d.get("task", "")

        pred_rows.append([
            m, k, tk, s["n_records"], pred,
            _count_pct(over_ext, pred), _count_pct(under_ext, pred),
            _count_pct(halluc, pred), _count_pct(joined, pred),
            _count_pct(mislabel, pred), _count_pct(mis_ext, pred),
            _count_pct(unclass, pred), f1_pct,
        ])
        gold_rows.append([m, k, tk, s["n_records"], gold,
                          _count_pct(omit, gold)])
        corr_rows.append([m, k, tk, s["n_records"], pred,
                          _count_pct(corrupt, pred)])

        sum_records += s["n_records"]
        sum_pred += pred;       sum_gold += gold
        sum_over_ext += over_ext; sum_under_ext += under_ext
        sum_halluc += halluc;   sum_joined += joined
        sum_mislabel += mislabel; sum_mis_ext += mis_ext
        sum_unclass += unclass; sum_omit += omit
        sum_corrupt += corrupt
        sum_f1_matches += t.get("f1_matches", 0)
        sum_f1_bundles += t.get("f1_bundles", 0)

    pred_total_row = [
        "TOTAL", "", "", sum_records, sum_pred,
        _count_pct(sum_over_ext, sum_pred), _count_pct(sum_under_ext, sum_pred),
        _count_pct(sum_halluc, sum_pred),   _count_pct(sum_joined, sum_pred),
        _count_pct(sum_mislabel, sum_pred), _count_pct(sum_mis_ext, sum_pred),
        _count_pct(sum_unclass, sum_pred),  _pct(sum_f1_matches, sum_f1_bundles),
    ]
    gold_total_row = ["TOTAL", "", "", sum_records, sum_gold,
                      _count_pct(sum_omit, sum_gold)]
    corr_total_row = ["TOTAL", "", "", sum_records, sum_pred,
                      _count_pct(sum_corrupt, sum_pred)]

    sections.append(_format_table(
        "OVERALL — pred-side F1 misses per run (share of pred)",
        pred_headers, pred_rows, total_row=pred_total_row,
    ))
    sections.append(_format_table(
        "OVERALL — gold-side F1 miss per run (share of gold)",
        gold_headers, gold_rows, total_row=gold_total_row,
    ))
    sections.append(_format_table(
        "OVERALL — corruption per run (diagnostic, share of pred)",
        corr_headers, corr_rows, total_row=corr_total_row,
    ))

    # 2. HALLUCINATION severity per run. Subcategories true/punct/case are
    # disjoint slices of `halluc`; show each as a share of that run's halluc
    # so the dominant level is obvious at a glance.
    # Built here as a string; injected next to the HALLUCINATION per-factor
    # summary in the category loop below so the two views sit together.
    # Two orthogonal cuts of `halluc`:
    #   strictness — true / punct / case (nested; mutually exclusive bins)
    #   origin     — special_tok / organic (whether pred contains chat-
    #                template markers; lets you separate corruption residue
    #                from genuine invented content).
    sev_headers = ["model", "kshot", "task", "halluc",
                   "true", "punct", "case", "special_tok", "organic"]
    sev_rows = []
    sum_h = sum_h_true = sum_h_punct = sum_h_case = 0
    sum_h_special = sum_h_organic = 0
    for d in runs:
        t = d["summary"]["totals"]
        h     = t.get("hallucination", 0)
        h_t   = t.get("halluc_true", 0)
        h_p   = t.get("halluc_punct", 0)
        h_c   = t.get("halluc_case", 0)
        h_s   = t.get("halluc_special_tok", 0)
        h_o   = t.get("halluc_organic", 0)
        m, k, tk = d.get("model", ""), str(d.get("k_shot", "")), d.get("task", "")
        sev_rows.append([m, k, tk, h,
                         _count_pct(h_t, h), _count_pct(h_p, h), _count_pct(h_c, h),
                         _count_pct(h_s, h), _count_pct(h_o, h)])
        sum_h += h;       sum_h_true += h_t
        sum_h_punct += h_p; sum_h_case += h_c
        sum_h_special += h_s; sum_h_organic += h_o
    sev_total = ["TOTAL", "", "", sum_h,
                 _count_pct(sum_h_true, sum_h),
                 _count_pct(sum_h_punct, sum_h),
                 _count_pct(sum_h_case, sum_h),
                 _count_pct(sum_h_special, sum_h),
                 _count_pct(sum_h_organic, sum_h)]
    sev_text = _format_table(
        "HALLUCINATION — severity per run (share of halluc)",
        sev_headers, sev_rows, total_row=sev_total,
    )

    # 3. CORRUPTION flag breakdown per run. Each flag's denominator is
    # pred_count (a leaked-token rate). Suppress the whole table if no run
    # had any corruption.
    # Built as a string; injected next to the CORRUPTED OUTPUT per-factor
    # summary in the category loop below.
    flag_headers = (["model", "kshot", "task", "pred"]
                    + list(_CORR_FLAGS) + ["overall"])
    flag_rows = []
    flag_totals = {flag: 0 for flag in _CORR_FLAGS}
    sum_pred_for_corr = 0
    sum_corr_overall = 0
    any_corr = False
    for d in runs:
        t = d["summary"]["totals"]
        pred = t.get("pred_count", 0)
        per_flag = [t.get("corr_" + flag, 0) for flag in _CORR_FLAGS]
        row_total = sum(per_flag)
        if row_total:
            any_corr = True
        m, k, tk = d.get("model", ""), str(d.get("k_shot", "")), d.get("task", "")
        flag_rows.append(
            [m, k, tk, pred]
            + [_count_pct(v, pred) for v in per_flag]
            + [_count_pct(row_total, pred)]
        )
        sum_pred_for_corr += pred
        sum_corr_overall += row_total
        for flag, v in zip(_CORR_FLAGS, per_flag):
            flag_totals[flag] += v
    flag_text = ""
    if any_corr:
        flag_total_row = (
            ["TOTAL", "", "", sum_pred_for_corr]
            + [_count_pct(flag_totals[f], sum_pred_for_corr) for f in _CORR_FLAGS]
            + [_count_pct(sum_corr_overall, sum_pred_for_corr)]
        )
        flag_text = _format_table(
            "CORRUPTED OUTPUT — flag counts per run (share of pred)",
            flag_headers, flag_rows, total_row=flag_total_row,
        )

    # 4. PER-ERROR-TYPE SUMMARY — one table per category. Rows = runs ordered
    # by (model, kshot, task); columns = factors (filtered to those with any
    # nonzero count, so empty columns don't pad the table). Right-most column
    # is the per-run overall; the bottom row sums down each column across
    # runs and gives the cross-sweep overall — i.e. the rolled-up category
    # total. With unclassified at 0 across the sweep, the "overall" column
    # for each category is the partition of F1 misses attributable to it.
    all_factors = set()
    for d in runs:
        all_factors.update((d["summary"].get("by_factor") or {}).keys())
    all_factors_sorted = sorted(all_factors)

    # Cell value for (run d, factor fac, category kind). Corruption is
    # stored per-flag (corr_<flag>) in by_factor; collapse to one number.
    def _cell(d, fac, kind):
        bf = (d["summary"].get("by_factor") or {}).get(fac, {})
        if kind == "corruption":
            return sum(bf.get("corr_" + flag, 0) for flag in _CORR_FLAGS)
        return bf.get(kind, 0)

    # Per-factor denominator for run d depends on the category: omission is
    # gold-side, everything else is pred-side.
    def _fac_denom(d, fac, kind):
        bf = (d["summary"].get("by_factor") or {}).get(fac, {})
        key = "gold_count" if kind == "omission" else "pred_count"
        return bf.get(key, 0)

    def _run_denom(d, kind):
        t = d["summary"]["totals"]
        return t.get("gold_count" if kind == "omission" else "pred_count", 0)

    # Per-category diagnostic blocks (built above) that should appear right
    # before the matching per-factor summary. Empty values are skipped so
    # categories with no diagnostic table (e.g. omission) still emit their
    # summary unchanged.
    diagnostic_blocks = {
        "hallucination": sev_text,
        "corruption":    flag_text,
    }

    for kind, label in [
        ("over_extraction",   "OVER-EXTRACTION"),
        ("under_extraction",  "UNDER-EXTRACTION"),
        ("corruption",        "CORRUPTED OUTPUT"),
        ("hallucination",     "HALLUCINATION"),
        ("joined_span",       "JOINED SPAN"),
        ("mislabeling", "MISLABELING"),
        ("omission",          "OMISSION"),
        ("mis_extraction",    "MIS-EXTRACTION"),
        ("unclassified",      "UNCLASSIFIED"),
    ]:
        col_totals = {fac: sum(_cell(d, fac, kind) for d in runs)
                      for fac in all_factors_sorted}
        # Corruption is a diagnostic category — keep every factor column even
        # when zero so the table shape stays comparable to the other category
        # tables and "absent flag" is visibly distinct from "factor not in the
        # space". Other categories collapse empty columns to reduce width.
        if kind == "corruption":
            factors_for_table = list(all_factors_sorted)
        else:
            factors_for_table = [fac for fac in all_factors_sorted if col_totals[fac]]
        grand_total = sum(col_totals.values())
        if not factors_for_table and not grand_total:
            continue

        # Per-factor denominator summed across all runs — used for the bottom
        # TOTAL row's inline pct so it reflects the cross-sweep share.
        col_denoms = {fac: sum(_fac_denom(d, fac, kind) for d in runs)
                      for fac in factors_for_table}
        sweep_denom = sum(_run_denom(d, kind) for d in runs)

        headers = ["model", "kshot", "task"] + factors_for_table + ["overall"]
        rows = []
        for d in runs:
            # Factors the trainer dropped at this shot count — render as
            # `NaN` to distinguish from "factor was evaluated and had 0
            # errors of this category" (which renders as blank via _count_pct).
            run_skip = set(d.get("skip_factors") or ())
            row = [d.get("model", ""), str(d.get("k_shot", "")), d.get("task", "")]
            row_total = 0
            for fac in factors_for_table:
                if fac in run_skip:
                    row.append("NaN")
                    continue
                v = _cell(d, fac, kind)
                row.append(_count_pct(v, _fac_denom(d, fac, kind)))
                row_total += v
            row.append(_count_pct(row_total, _run_denom(d, kind)))
            rows.append(row)
        total_row = (
            ["overall", "", ""]
            + [_count_pct(col_totals[fac], col_denoms[fac]) for fac in factors_for_table]
            + [_count_pct(grand_total, sweep_denom)]
        )
        diag = diagnostic_blocks.get(kind)
        if diag:
            sections.append(diag)
        sections.append(_format_table(
            f"{label} — summary across runs",
            headers, rows, total_row=total_row,
        ))

    return "\n".join(s for s in sections if s)


def write_comparison_log(json_paths, log_path, model_aliases=None,
                         supervised_models=None):
    """Split `json_paths` by (dataset, task) and write one independent
    comparison log (+ sibling `.tex`) per pair, to `<log_path>` with the
    dataset and task inserted before the extension (e.g.
    `compare_Llama3.2-3B_constructcie_jhe.log`).

    This is the single-combo variant `analyze_combination` calls for its
    own `write_compare` (cross-k-shot log for one model/dataset/task) — for
    that caller `json_paths` only ever spans one (dataset, task) pair, so
    the split is a no-op there and this just names the file. For a
    multi-model, multi-dataset sweep, use `write_aggregate_comparison_log`
    instead, which nests output per dataset the way `-a analysis` does
    rather than suffixing one flat filename.

    Splitting matters in the general case because `aggregate_tables`' BY-
    MODEL SUMMARY and TOTAL rows roll counts up across every run given to
    them, with no awareness of dataset or task — without the split, a model
    run under both JHE and IHE would have its error counts silently summed
    together into one misleading row/total instead of kept independent.

    `model_aliases` is the same map data_analyzer uses ({full_name ->
    short_alias}); passing it from `run.py` keeps the LaTeX labels short
    and consistent across artifacts.

    Returns the list of log_paths written (empty when nothing was written).
    """
    paths_by_key = {}
    for p in json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                dataset = data.get("dataset", "")
                task = str(data.get("task", "")).lower()
        except Exception as exc:
            logger.warning("could not read %s: %s", p, exc)
            continue
        paths_by_key.setdefault((dataset, task), []).append(p)

    written = []
    root, ext = os.path.splitext(log_path)

    for dataset_key, task_key in sorted(paths_by_key):
        sub_paths = paths_by_key[(dataset_key, task_key)]
        text = aggregate_tables(sub_paths, supervised_models=supervised_models)
        if not text:
            continue
        suffix = "_".join(s for s in (dataset_key, task_key) if s)
        ds_log_path = f"{root}_{suffix}{ext}" if suffix else log_path
        os.makedirs(os.path.dirname(ds_log_path) or ".", exist_ok=True)
        with open(ds_log_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        logger.warning("wrote %s", ds_log_path)

        runs = []
        for p in sub_paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception as exc:
                logger.warning("could not read %s: %s", p, exc)
        if runs:
            tex_body = _render_by_model_summary_latex(
                runs, model_aliases=model_aliases,
                supervised_models=supervised_models)
            if tex_body:
                tex_path = os.path.splitext(ds_log_path)[0] + ".tex"
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(tex_body + "\n")
                logger.warning("wrote %s", tex_path)
        written.append(ds_log_path)
    return written


def write_aggregate_comparison_log(json_paths, log_dir, model_aliases=None,
                                   supervised_models=None, model_order=None):
    """Write the cross-model, cross-task `-a error --aggregate` summary,
    one dataset per subfolder — same layout `-a analysis` uses for its own
    output (see `progress_analyzer.per_dataset_log_path`): each dataset
    gets `<log_dir>/<dataset>/`, so two datasets never collide on the same
    filename and every dataset's artifacts sit together.

    Inside each dataset folder:

    1. Plain-text `.log` — one per task (e.g. `compare_all_jhe.log`,
       `compare_all_ihe.log`). Splitting by task here matters because
       `aggregate_tables`' BY-MODEL SUMMARY and TOTAL rows roll counts up
       across every run it's given, with no awareness of task — without
       the split, a model run under both JHE and IHE would have its error
       counts silently summed together into one misleading row/total
       instead of kept independent.

    2. LaTeX `.tex` — one `compare_all.tex` per dataset (both tasks
       pooled into one table, via `_render_grouped_composition_latex`).
       This is the paper-ready artifact: Supervised / JHE / IHE row
       blocks in a single table, still keeping each model's JHE and IHE
       numbers separate within it.

    `model_aliases` is the same {full_name -> short_alias} map
    data_analyzer uses; `model_order` is the row/display order (pass
    `list(model_alts.keys())` from the global config) — both threaded from
    run.py so the LaTeX output's naming and ordering come from the same
    config source of truth instead of being hardcoded here.

    Returns the list of paths written (empty when nothing was written).
    """
    from main.progress_analyzer import per_dataset_log_path

    paths_by_key = {}
    paths_by_dataset = {}
    for p in json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                dataset = data.get("dataset", "")
                task = str(data.get("task", "")).lower()
        except Exception as exc:
            logger.warning("could not read %s: %s", p, exc)
            continue
        paths_by_key.setdefault((dataset, task), []).append(p)
        paths_by_dataset.setdefault(dataset, []).append(p)

    written = []

    for dataset_key, task_key in sorted(paths_by_key):
        sub_paths = paths_by_key[(dataset_key, task_key)]
        text = aggregate_tables(sub_paths, supervised_models=supervised_models)
        if not text:
            continue
        ds_dir = per_dataset_log_path(log_dir, dataset_key) if dataset_key else log_dir
        os.makedirs(ds_dir, exist_ok=True)
        fname = f"compare_all_{task_key}.log" if task_key else "compare_all.log"
        ds_log_path = os.path.join(ds_dir, fname)
        with open(ds_log_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        logger.warning("wrote %s", ds_log_path)
        written.append(ds_log_path)

    for dataset_key in sorted(paths_by_dataset):
        runs = []
        for p in paths_by_dataset[dataset_key]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception as exc:
                logger.warning("could not read %s: %s", p, exc)
        tex_body = _render_grouped_composition_latex(
            runs, model_order=model_order, model_aliases=model_aliases,
            supervised_models=supervised_models,
            caption="Exact-match error distributions by extraction "
                    f"strategy{' on ' + dataset_key if dataset_key else ''}.",
            label=f"tab:error-{dataset_key}" if dataset_key else "tab:error",
        )
        if not tex_body:
            continue
        ds_dir = per_dataset_log_path(log_dir, dataset_key) if dataset_key else log_dir
        os.makedirs(ds_dir, exist_ok=True)
        tex_path = os.path.join(ds_dir, "compare_all.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_body + "\n")
        logger.warning("wrote %s", tex_path)
        written.append(tex_path)

    return written


def print_summary_tables(out_path, write_log=True):
    """Render the table block and emit it via logger.

    When `write_log` is True (default), also persist the rendered text to a
    sibling `.log` file next to the source JSON.
    """
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("could not read %s: %s", out_path, exc)
        return None

    data["_basename"] = os.path.basename(out_path)
    text = _build_tables_text(data)
    logger.error("%s", text)

    if write_log:
        log_path = os.path.splitext(out_path)[0] + ".log"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            logger.warning("wrote %s", log_path)
        except OSError as exc:
            logger.warning("could not write %s: %s", log_path, exc)
        return log_path
    return None


# Backwards-compatible alias: callers still importing print_summary_line work.
print_summary_line = print_summary_tables


# ---------------------------------------------------------------------------
# Standalone CLI: analyze a single pred dir
# ---------------------------------------------------------------------------

_PRED_RE = re.compile(r"^pred_(.+)_split(\d+)_k(\d+)\.json$")


def _scan_pred_dir(pred_dir):
    """Group pred_*.json files by (model, dataset, task)."""
    groups = defaultdict(set)
    for f in os.listdir(pred_dir):
        m = _PRED_RE.match(f)
        if not m:
            continue
        head = m.group(1)
        parts = head.rsplit("_", 2)
        if len(parts) != 3:
            continue
        model, dataset, task = parts
        groups[(model, dataset, task)].add(int(m.group(3)))
    return groups


def main():
    p = argparse.ArgumentParser(description="LLM AC error analysis (standalone)")
    p.add_argument("--pred_dir", required=True,
                   help="Directory containing pred_*.json files for one (task/dataset/model)")
    p.add_argument("--gold_root", required=True,
                   help="Path to processed_data root containing <dataset>/split{N}/test.json")
    p.add_argument("--key_map", required=True,
                   help="Path to global_data/key_mapping.json")
    p.add_argument("--sim_threshold", type=float, default=0.8)
    p.add_argument("--skip_columns", nargs="*", default=None,
                   help="Task-level skip_columns from the global mapping "
                        "(e.g. accident_report id); run.py passes these "
                        "automatically via get_global_config")
    p.add_argument("--log_dir", default=None,
                   help="Where to write errors_*.json (default: --pred_dir)")
    p.add_argument("--log_level", default="WARNING",
                   help="Logging level for the standalone CLI (default: WARNING)")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(message)s",
    )

    with open(args.key_map, "r", encoding="utf-8") as f:
        keymap = json.load(f)
    structure = (keymap or {}).get("structure", {})
    # classes_map enables IE→AC conversion for supervised pred files; harmless
    # for LLM preds since convert_predictions short-circuits same-format input.
    classes_map = (
        ((keymap or {}).get("task", {}) or {})
        .get("classification", {})
        .get("classes", {})
    ) or None

    written_all = []
    for (model, dataset, task), shots in _scan_pred_dir(args.pred_dir).items():
        written = analyze_combination(
            out_path=args.pred_dir,
            model=model, dataset=dataset, task=task.upper(),
            gold_root=args.gold_root, structure=structure,
            sim_threshold=args.sim_threshold,
            shots=sorted(shots),
            log_dir=args.log_dir,
            classes_map=classes_map,
            write_compare=True,
            base_skip=args.skip_columns,
        )
        written_all.extend(written)

    for path in written_all:
        print_summary_tables(path)


if __name__ == "__main__":
    main()
