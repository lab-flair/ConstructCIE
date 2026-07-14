import numpy as np
import difflib
import re
from scipy.optimize import linear_sum_assignment

def exclude_factor_from_overall(score, factor):
    """Recompute `score.factors.overall` (and `factors.per_acc_type`) by
    subtracting `factor`'s match / pred / gold counts. The per_factor entry
    is left intact so the standalone factor F1 is still recoverable.

    Used so that E2*-family tasks (which PREDICT accident_type) report an
    overall F1 that's directly comparable to non-E2 tasks (which receive
    accident_type as gold input via skip_columns). Calling this on an E2*
    score with `factor="accident_type"` puts both task families on equal
    footing — accident_type contributes to neither overall.

    Modifies `score` in place. No-op when per_factor[factor] is missing or
    its f1 is the literal "NaN" string set by skip_columns (e.g. ACH/ACH1
    where accident_type is already excluded at score time).
    """
    factors = score.get("factors", {})
    per_factor = factors.get("per_factor") or {}
    at = per_factor.get(factor) or {}
    at_overall = at.get("overall") or {}
    v = at_overall.get("f1")
    # `bool` is a subclass of int — exclude it; `f1` is never True/False.
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return

    overall = factors.get("overall") or {}
    pn = overall.get("pred_num", 0) - at_overall.get("pred_num", 0)
    gn = overall.get("gold_num", 0) - at_overall.get("gold_num", 0)
    mn = overall.get("match_num", 0) - at_overall.get("match_num", 0)
    if pn < 0 or gn < 0 or mn < 0:
        return  # defensive — leave alone if counts are inconsistent
    p, r, f = compute_f1(pn, gn, mn)
    factors["overall"] = {
        **overall, "pred_num": pn, "gold_num": gn, "match_num": mn,
        "precision": p, "recall": r, "f1": f,
    }

    # Mirror the subtraction in each per_acc_type bucket so the per-
    # acctype summary stays consistent with the rewritten overall.
    at_per_atype = at.get("per_acc_type") or {}
    for atype, pa_node in list((factors.get("per_acc_type") or {}).items()):
        at_pa = at_per_atype.get(atype)
        if not at_pa:
            continue
        v2 = at_pa.get("f1")
        if not isinstance(v2, (int, float)) or isinstance(v2, bool):
            continue
        pn2 = pa_node.get("pred_num", 0) - at_pa.get("pred_num", 0)
        gn2 = pa_node.get("gold_num", 0) - at_pa.get("gold_num", 0)
        mn2 = pa_node.get("match_num", 0) - at_pa.get("match_num", 0)
        if pn2 < 0 or gn2 < 0 or mn2 < 0:
            continue
        p2, r2, f2 = compute_f1(pn2, gn2, mn2)
        factors["per_acc_type"][atype] = {
            **pa_node, "pred_num": pn2, "gold_num": gn2, "match_num": mn2,
            "precision": p2, "recall": r2, "f1": f2,
        }


def strict_keys_from_keymap(key_map):
    """Classification factor keys from a key_map dict.

    Returns the list of keys under `task.classification.classes` (e.g.
    `accident_type`, `severity`, `construction_trade`). Pass to
    `compute_AC_scores` / `compute_scores` as `strict_keys` so those
    factors require exact match instead of sim_threshold-fuzzy match
    (e.g. so "fatality" can't match "fatalities" at ratio 0.89).
    """
    classes = ((key_map or {}).get("task", {}) or {}).get("classification", {}).get("classes") or {}
    return list(classes.keys())


def compute_scores(preds, golds, task, sim_threshold=0.8, skip_columns=None,
                   norm=True, strict_keys=None):
    if task == "ED":
        return compute_ED_scores(preds, golds, metrics={"trigger_id", "trigger_cls"})
    elif task == "EAE":
        return compute_EAE_scores(preds, golds, metrics={"argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"})
    elif task == "EARL":
        return compute_EARL_scores(preds, golds, metrics={"argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"})
    elif task == "E2E":
        return compute_E2E_scores(preds, golds, metrics={"trigger_id", "trigger_cls", "argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"})
    elif task in ("AC", "AC1", "ACH", "ACH1", "JHE", "IHE"):
        return compute_AC_scores(preds, golds, metrics={"factors"},
                                 sim_threshold=sim_threshold,
                                 skip_columns=skip_columns, norm=norm,
                                 strict_keys=strict_keys)

def print_scores(scores, split=None, shot=None, stage=None):
    """Pretty-print a scores dict.

    Optional `split`, `shot`, and `stage` are shown in the header. `stage`
    disambiguates contexts that otherwise share the same `[split, shot]` label —
    e.g. per-epoch `val` evals during training vs the final `test` eval, which
    would otherwise both print under `[split1] k=0`. `split=None` is treated
    as the global aggregate.
    """
    label_split = split if split else "GLOBAL"
    label_shot  = f"k={shot}" if shot is not None else "k=?"
    label_stage = f"  ({stage})" if stage else ""
    print("------------------------------------------------------------------------------")
    print(f"  [{label_split}]  {label_shot}{label_stage}")
    print("------------------------------------------------------------------------------")
    if "trigger_id" in scores:
        print('Tri-I            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["trigger_id"]["precision"], scores["trigger_id"]["match_num"], scores["trigger_id"]["pred_num"], 
            scores["trigger_id"]["recall"], scores["trigger_id"]["match_num"], scores["trigger_id"]["gold_num"], scores["trigger_id"]["f1"]))
    if "trigger_cls" in scores:
        print('Tri-C            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["trigger_cls"]["precision"], scores["trigger_cls"]["match_num"], scores["trigger_cls"]["pred_num"], 
            scores["trigger_cls"]["recall"], scores["trigger_cls"]["match_num"], scores["trigger_cls"]["gold_num"], scores["trigger_cls"]["f1"]))
    if "argument_id" in scores:
        print('Arg-I            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["argument_id"]["precision"], scores["argument_id"]["match_num"], scores["argument_id"]["pred_num"], 
            scores["argument_id"]["recall"], scores["argument_id"]["match_num"], scores["argument_id"]["gold_num"], scores["argument_id"]["f1"]))
    if "argument_cls" in scores:
        print('Arg-C            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["argument_cls"]["precision"], scores["argument_cls"]["match_num"], scores["argument_cls"]["pred_num"], 
            scores["argument_cls"]["recall"], scores["argument_cls"]["match_num"], scores["argument_cls"]["gold_num"], scores["argument_cls"]["f1"]))
    if "argument_attached_id" in scores:
        print('Arg-I (attached) - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["argument_attached_id"]["precision"], scores["argument_attached_id"]["match_num"], scores["argument_attached_id"]["pred_num"], 
            scores["argument_attached_id"]["recall"], scores["argument_attached_id"]["match_num"], scores["argument_attached_id"]["gold_num"], scores["argument_attached_id"]["f1"]))
    if "argument_attached_cls" in scores:
        print('Arg-C (attached) - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["argument_attached_cls"]["precision"], scores["argument_attached_cls"]["match_num"], scores["argument_attached_cls"]["pred_num"], 
            scores["argument_attached_cls"]["recall"], scores["argument_attached_cls"]["match_num"], scores["argument_attached_cls"]["gold_num"], scores["argument_attached_cls"]["f1"]))
    
    # AC Specific Print Statements
    if "factors" in scores:
        # --- 1. OVERALL ---
        ov = scores["factors"]["overall"]
        print('AC OVERALL       - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            ov["precision"], ov["match_num"], ov["pred_num"], 
            ov["recall"], ov["match_num"], ov["gold_num"], ov["f1"]))
        
        # --- 2. OVERALL BY ACCIDENT TYPE ---
        for atype, a_scores in scores["factors"]["per_acc_type"].items():
            label = f"  [{atype}]"[:16].ljust(16)
            print('{} - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
                label,
                a_scores["precision"], a_scores["match_num"], a_scores["pred_num"], 
                a_scores["recall"], a_scores["match_num"], a_scores["gold_num"], a_scores["f1"]))

        print("------------------------------------------------------------------------------")
        print(" SCORES PER FACTOR".center(78))
        # Single header line listing the skipped factors; the per-factor block
        # below omits their rows entirely.
        skipped_factors = sorted(
            k for k, f_data in scores["factors"]["per_factor"].items()
            if isinstance(f_data.get("overall", {}).get("f1"), str)
        )
        if skipped_factors:
            print(f" Skipped: {', '.join(skipped_factors)}")
        print("------------------------------------------------------------------------------")

        def _fmt_row(label, s):
            # Skipped columns (f1 is the string "NaN") are dropped from the
            # human-readable output. The JSON record still carries them.
            if isinstance(s.get("f1"), str):
                return None
            return ('{} - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'
                    .format(label,
                            s["precision"], s["match_num"], s["pred_num"],
                            s["recall"],    s["match_num"], s["gold_num"], s["f1"]))

        # --- 3 & 4. BY FACTOR & BY FACTOR PER ACCIDENT TYPE ---
        for key, f_data in scores["factors"]["per_factor"].items():
            ov_f = f_data["overall"]
            label = f"{key.upper()}"[:16].ljust(16)
            row = _fmt_row(label, ov_f)
            if row is None:
                continue  # skipped factor — hide the whole block including sub-rows
            print(row)

            for atype, ka_scores in f_data["per_acc_type"].items():
                sub_label = f"  -> [{atype}]"[:16].ljust(16)
                sub_row = _fmt_row(sub_label, ka_scores)
                if sub_row is not None:
                    print(sub_row)
                
    print("------------------------------------------------------------------------------\n")

def print_keyword_scores(scores, split=None, shot=None, stage=None):
    """Pretty-print a keyword-match scores dict.

    Same layout as `print_scores`'s AC section but headers say KW so a
    reader can't confuse the table with the similarity-overlap scores
    printed above it. Skipped factors (f1=="NaN") are listed in the header
    line and omitted from the per-factor block, matching `print_scores`.
    """
    if "factors" not in scores:
        return
    label_split = split if split else "GLOBAL"
    label_shot  = f"k={shot}" if shot is not None else "k=?"
    label_stage = f"  ({stage})" if stage else ""
    print("------------------------------------------------------------------------------")
    print(f"  [{label_split}]  {label_shot}{label_stage}   (KEYWORD MATCH)")
    print("------------------------------------------------------------------------------")

    ov = scores["factors"]["overall"]
    print('KW OVERALL       - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
        ov["precision"], ov["match_num"], ov["pred_num"],
        ov["recall"],    ov["match_num"], ov["gold_num"], ov["f1"]))

    for atype, a_scores in scores["factors"]["per_acc_type"].items():
        label = f"  [{atype}]"[:16].ljust(16)
        print('{} - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            label,
            a_scores["precision"], a_scores["match_num"], a_scores["pred_num"],
            a_scores["recall"],    a_scores["match_num"], a_scores["gold_num"], a_scores["f1"]))

    print("------------------------------------------------------------------------------")
    print(" KEYWORD SCORES PER FACTOR".center(78))
    skipped_factors = sorted(
        k for k, f_data in scores["factors"]["per_factor"].items()
        if isinstance(f_data.get("overall", {}).get("f1"), str)
    )
    if skipped_factors:
        print(f" Skipped: {', '.join(skipped_factors)}")
    print("------------------------------------------------------------------------------")

    def _fmt_row(label, s):
        if isinstance(s.get("f1"), str):
            return None
        return ('{} - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'
                .format(label,
                        s["precision"], s["match_num"], s["pred_num"],
                        s["recall"],    s["match_num"], s["gold_num"], s["f1"]))

    for key, f_data in scores["factors"]["per_factor"].items():
        ov_f = f_data["overall"]
        label = f"{key.upper()}"[:16].ljust(16)
        row = _fmt_row(label, ov_f)
        if row is None:
            continue
        print(row)
        for atype, ka_scores in f_data["per_acc_type"].items():
            sub_label = f"  -> [{atype}]"[:16].ljust(16)
            sub_row = _fmt_row(sub_label, ka_scores)
            if sub_row is not None:
                print(sub_row)
    print("------------------------------------------------------------------------------\n")


def print_keyword_score_comparison(scores_by_mk, coverage=None, structure=None,
                                   is_supervised_map=None):
    """Cross-(model, kshot) keyword-F1 comparison.

    Delegates to the same renderer as `print_score_comparison` after retitling
    the banner. The shape of the keyword `scores` dict matches the overlap
    scorer's (overall / per_acc_type / per_factor.overall / per_factor.per_acc_type),
    so the renderer's tables and DIFF blocks render unchanged — only the
    semantics of the F1 cells change (keyword containment vs span overlap).
    """
    if not scores_by_mk:
        return
    print("\n" + "=" * 78)
    print(" CROSS-MODEL KEYWORD-F1 COMPARISON ".center(78, "="))
    print("=" * 78)
    print_score_comparison(scores_by_mk, coverage=coverage, structure=structure,
                           is_supervised_map=is_supervised_map,
                           _suppress_banner=True)


class _LatexComparisonFormatter:
    """Emits a LaTeX-friendly version of the AC cross-model F1 comparison.

    Mirrors `print_score_comparison`'s factor-grouping logic but renders into
    one or more `\\begin{table*}` blocks per task. Each table column-group
    corresponds to a main factor (with its subs) or a standalone leaf.

    Splits across multiple tables when the cumulative column count would
    overflow page width; the first table carries the TOTAL block, subsequent
    tables repeat only `k` / `Model` plus their slice of factor groups.
    """

    # Soft cap on LaTeX columns per table. Used to derive chunk count
    # (ceil(total/max)) and then groups are distributed across that many
    # chunks for roughly equal width. 40 puts a full E2ACH (62 cols) into
    # 2 balanced halves rather than 3 lopsided ones.
    DEFAULT_MAX_COLS = 40

    def __init__(self, scores_by_mk, structure=None,
                 caption="Cross-model F1 by factor group.",
                 label="tab:cross-model-f1",
                 max_cols_per_table=None):
        self.scores_by_mk = scores_by_mk
        self.structure = structure or {}
        self.caption = caption
        self.label = label
        self.max_cols = max_cols_per_table or self.DEFAULT_MAX_COLS

    @staticmethod
    def _esc(s):
        """Escape LaTeX special chars in a free-text string (model names)."""
        return (str(s)
                .replace("\\", r"\textbackslash{}")
                .replace("_", r"\_")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("#", r"\#")
                .replace("$", r"\$"))

    @staticmethod
    def _title(s):
        """factor_key → human title (working_circumstances → "Working Circumstances")."""
        return " ".join(p.capitalize() for p in str(s).split("_"))

    @staticmethod
    def _cell(node, key):
        """Format one P/R/F1 cell. `--` for missing / explicit-skip nodes."""
        if not node:
            return "--"
        v = node.get(key)
        if isinstance(v, str) or v is None:
            return "--"
        return f"{v:.2f}"

    def _collect(self):
        """Mirror `print_score_comparison`'s factor-collection + ordering."""
        sbk = self.scores_by_mk
        tasks = sorted({t for _, t, _ in sbk})
        all_factors = {
            f for sc in sbk.values()
            for f in (sc.get("factors", {}).get("per_factor") or {})
        }

        def _has_real_score(factor):
            for sc in sbk.values():
                node = ((sc.get("factors", {}).get("per_factor") or {}).get(factor) or {}).get("overall", {})
                v = node.get("f1")
                if v is not None and not isinstance(v, str):
                    return True
            return False
        all_factors = {f for f in all_factors if _has_real_score(f)}

        main_to_subs = {
            m: list(self.structure.get(m, []))
            for m in self.structure.get("accident_report", [])
        }
        is_main = set(main_to_subs)
        is_sub = {s for subs in main_to_subs.values() for s in subs}

        factors = []
        for main, subs in main_to_subs.items():
            if main in all_factors:
                factors.append(main)
            for sub in subs:
                if sub in all_factors:
                    factors.append(sub)
        leftovers = sorted(f for f in all_factors if f not in is_main and f not in is_sub)
        factors.extend(leftovers)
        if "accident_type" in factors:
            factors.remove("accident_type")
            factors.insert(0, "accident_type")
        return tasks, factors, main_to_subs, is_main, is_sub

    def _build_groups(self, factors, main_to_subs, is_main, is_sub):
        """Return [(group_label, [(subcol_label, factor_key), ...]), ...].

        A main factor becomes a group with "All" + each present sub. A
        leftover becomes a single-subcol group. Subs already consumed by
        their main are not emitted again.
        """
        groups = []
        consumed = set()
        for f in factors:
            if f in consumed:
                continue
            if f in is_main:
                subs = [s for s in main_to_subs.get(f, []) if s in factors]
                subcols = [("All", f)] + [(self._title(s), s) for s in subs]
                consumed.update(subs)
                groups.append((self._title(f), subcols))
            else:
                groups.append((self._title(f), [("All", f)]))
        return groups

    def _split_groups(self, splittable_groups, fixed_per_chunk):
        """Balanced split of `splittable_groups` into N chunks.

        N is chosen so that `fixed_per_chunk + max(chunk_cols) <= max_cols`,
        i.e. N = ceil((fixed + total_splittable) / max_cols). Splittable
        cols are then distributed greedily so each chunk lands near
        `total_splittable / N` cols, avoiding lopsided splits.

        `fixed_per_chunk` is the per-chunk overhead (k, Model, TOTAL, plus
        any anchor groups that appear in every chunk)."""
        import math
        splittable_total = sum(3 * len(g[1]) for g in splittable_groups)
        total_cols = fixed_per_chunk + splittable_total

        n_chunks = max(1, math.ceil(total_cols / self.max_cols))
        if n_chunks == 1 or not splittable_groups:
            return [splittable_groups]

        target_splittable = splittable_total / n_chunks
        chunks = []
        cur, cur_cols = [], 0
        for grp in splittable_groups:
            grp_cols = 3 * len(grp[1])
            # Decide BEFORE adding: would flushing now leave the current
            # chunk closer to target than adding this group would? Picks
            # the better of the two split points and avoids the lopsided
            # "always-fill-then-overshoot" pattern (e.g. 33/21 → 24/30).
            if cur and len(chunks) < n_chunks - 1:
                dist_now = abs(cur_cols - target_splittable)
                dist_after = abs(cur_cols + grp_cols - target_splittable)
                if dist_now <= dist_after:
                    chunks.append(cur)
                    cur, cur_cols = [], 0
            cur.append(grp)
            cur_cols += grp_cols
        if cur:
            chunks.append(cur)
        return chunks

    def _header(self, chunk_groups, include_total):
        """Build the 3-row header for one table chunk."""
        n_subcols = sum(len(g[1]) for g in chunk_groups)
        n_subcols_total = n_subcols + (1 if include_total else 0)
        col_spec = "ll*{" + str(n_subcols_total) + "}{ccc}"

        out = []
        out.append(r"\resizebox{\textwidth}{!}{%")
        out.append(r"\begin{tabular}{" + col_spec + "}")
        out.append(r"\toprule")

        # Row 1: \multirow{3} for k/Model; \multicolumn for each group.
        row1 = [r"\multirow{3}{*}{\textbf{k}}",
                r"\multirow{3}{*}{\textbf{Model}}"]
        if include_total:
            row1.append(r"\multicolumn{3}{c}{\multirow{2}{*}{\textbf{Total}}}")
        for label, subcols in chunk_groups:
            row1.append(r"\multicolumn{" + str(3 * len(subcols)) + r"}{c}{\textbf{" + label + "}}")
        out.append(" &\n".join(row1) + r" \\")

        # cmidrules between row 1 and row 2: under each group except Total
        # (which is multirow'd over rows 1+2).
        col = 3 + (3 if include_total else 0)  # first factor-group col
        midrules1 = []
        for label, subcols in chunk_groups:
            w = 3 * len(subcols)
            midrules1.append(r"\cmidrule(lr){" + f"{col}-{col + w - 1}" + "}")
            col += w
        if midrules1:
            out.append(" ".join(midrules1))

        # Row 2: empty under k/Model (multirow'd) and Total (multirow'd);
        # then "All" / sub labels under each group's subcols.
        row2 = ["", ""]
        if include_total:
            row2.append(r"\multicolumn{3}{c}{}")
        for label, subcols in chunk_groups:
            for sub_label, _ in subcols:
                # Wrap long sub-labels in \shortstack at the first space if
                # the label exceeds a width threshold; keeps header compact.
                if len(sub_label) > 18 and " " in sub_label:
                    mid = sub_label.rfind(" ", 0, len(sub_label) // 2 + 4)
                    if mid > 0:
                        sub_label = (r"\shortstack{" + sub_label[:mid]
                                     + r"\\" + sub_label[mid + 1:] + "}")
                row2.append(r"\multicolumn{3}{c}{" + sub_label + "}")
        out.append(" &\n".join(row2) + r" \\")

        # cmidrules between row 2 and row 3: under EVERY 3-wide sub-column
        # (including Total now that we're entering the metric header row).
        col = 3
        midrules2 = []
        if include_total:
            midrules2.append(r"\cmidrule(lr){" + f"{col}-{col + 2}" + "}")
            col += 3
        for label, subcols in chunk_groups:
            for _ in subcols:
                midrules2.append(r"\cmidrule(lr){" + f"{col}-{col + 2}" + "}")
                col += 3
        if midrules2:
            out.append(" ".join(midrules2))

        # Row 3: P / R / F1 under every triple.
        row3 = ["", ""]
        n_triples = (1 if include_total else 0) + n_subcols
        row3.extend(["P & R & F1"] * n_triples)
        out.append(" & ".join(row3) + r" \\")
        out.append(r"\midrule")
        return out

    def _row_cells(self, sc, chunk_groups, include_total):
        """Render the data cells for one (model, task, kshot) row."""
        cells = []
        if include_total:
            tn = sc.get("factors", {}).get("overall")
            cells.extend([self._cell(tn, "precision"),
                          self._cell(tn, "recall"),
                          self._cell(tn, "f1")])
        per_factor = sc.get("factors", {}).get("per_factor") or {}
        for label, subcols in chunk_groups:
            for _, fk in subcols:
                node = (per_factor.get(fk) or {}).get("overall")
                cells.extend([self._cell(node, "precision"),
                              self._cell(node, "recall"),
                              self._cell(node, "f1")])
        return cells

    def _emit_task(self, task, task_rows, groups, suffix=""):
        """Emit ONE `\\begin{table*}` per task; chunks become stacked
        `\\resizebox{\\textwidth}{!}{\\begin{tabular}}` blocks inside it.

        The first chunk carries TOTAL + Accident Type as orientation
        columns; later chunks hold only their slice of factor groups
        (still preceded by `k` and `Model` so each row is identifiable).
        Caption + label render once at the end of the joined table*."""
        sbk = self.scores_by_mk

        # Per-task gating: only show accident_type when THIS task actually
        # scores it (E2AC / E2AC1 / E2ACH / E2ACH1). Non-E2 variants list
        # accident_type in skip_columns so its per_factor cells are "NaN"
        # strings — rendering them would produce a column of "--" stripes.
        def _has_real_acc_type(rk):
            node = (((sbk[rk].get("factors", {}).get("per_factor") or {})
                     .get("accident_type") or {}).get("overall", {}))
            v = node.get("f1")
            return v is not None and not isinstance(v, str)
        task_scores_acc_type = any(_has_real_acc_type(r) for r in task_rows)

        # Anchor groups — appended to the FIRST chunk only (alongside TOTAL).
        # accident_type lives here when in scope so it sits next to Total on
        # the first tabular and isn't repeated in subsequent ones.
        anchor_groups = []
        splittable_groups = []
        for grp in groups:
            is_acc_type = any(fk == "accident_type" for _, fk in grp[1])
            if is_acc_type:
                if task_scores_acc_type:
                    anchor_groups.append(grp)
                # else: drop entirely for this task's table
            else:
                splittable_groups.append(grp)
        anchor_cols = sum(3 * len(g[1]) for g in anchor_groups)
        # First-chunk overhead: k+Model + TOTAL + anchor groups.
        # Subsequent chunks: just k+Model. Pass the *larger* overhead so the
        # split balances around what the page can fit (the first chunk is
        # the widest, so it drives the chunk count).
        fixed_first = 2 + 3 + anchor_cols
        chunks = self._split_groups(splittable_groups, fixed_first)

        # Sort rows by (kshot, model) within the task.
        def _natural_key(s):
            return tuple(int(t) if t.isdigit() else t.lower()
                         for t in re.split(r'(\d+)', s))
        task_rows = sorted(task_rows,
                           key=lambda mtk: (int(mtk[2]), _natural_key(mtk[0])))

        # Open the task's table* (once).
        print()
        print(f"% Task: {task}")
        print(r"\begin{table*}[t]")
        print(r"\centering")
        print(r"\tiny")
        print(r"\setlength{\tabcolsep}{1.8pt}")
        print(r"\renewcommand{\arraystretch}{1.08}")

        for chunk_idx, chunk in enumerate(chunks):
            is_first = (chunk_idx == 0)
            full_chunk = (anchor_groups + chunk) if is_first else chunk
            include_total = is_first

            print()
            if not is_first:
                # Visual gap between joined tabulars within the same table*.
                print(r"\vspace{0.35em}")
                print()
            for line in self._header(full_chunk, include_total):
                print(line)

            # Data rows, grouped by kshot with \midrule between groups.
            prev_k = None
            for (m, t, k) in task_rows:
                if prev_k is not None and k != prev_k:
                    print(r"\midrule")
                cells = [k, self._esc(m)]
                cells.extend(self._row_cells(sbk[(m, t, k)], full_chunk, include_total))
                print(" & ".join(cells) + r" \\")
                prev_k = k

            print(r"\bottomrule")
            print(r"\end{tabular}%")
            print(r"}")

        # Caption + label once for the whole joined table*.
        cap = self.caption
        if suffix:
            cap = f"{cap} ({suffix})"
        print(r"\caption{" + cap + "}")
        lbl = self.label
        if task and task not in lbl:
            lbl = f"{lbl}-{task.lower()}"
        print(r"\label{" + lbl + "}")
        print(r"\end{table*}")

    def render(self):
        tasks, factors, main_to_subs, is_main, is_sub = self._collect()
        if not factors:
            return
        groups = self._build_groups(factors, main_to_subs, is_main, is_sub)
        rows = list(self.scores_by_mk.keys())
        for task in tasks:
            task_rows = [r for r in rows if r[1] == task]
            if not task_rows:
                continue
            self._emit_task(task, task_rows, groups)


def print_score_comparison_latex(scores_by_mk, structure=None,
                                  caption="Cross-model F1 by factor group with span-overlap matching.",
                                  label="tab:cross-model-f1",
                                  max_cols_per_table=None):
    """LaTeX renderer for the AC cross-model F1 comparison — sibling of
    `print_score_comparison`. Emits one or more `\\begin{table*}` blocks per
    task to stdout; caller is responsible for redirecting to a `.tex` file.
    """
    if not scores_by_mk:
        return
    _LatexComparisonFormatter(
        scores_by_mk, structure=structure,
        caption=caption, label=label,
        max_cols_per_table=max_cols_per_table,
    ).render()


def print_keyword_score_comparison_latex(scores_by_mk, structure=None,
                                          caption="Cross-model keyword-F1 by factor group with keyword string matching.",
                                          label="tab:cross-model-kw-f1",
                                          max_cols_per_table=None):
    """LaTeX renderer for the keyword-F1 comparison — same shape as
    `print_score_comparison_latex` with retitled caption/label."""
    print_score_comparison_latex(
        scores_by_mk, structure=structure,
        caption=caption, label=label,
        max_cols_per_table=max_cols_per_table,
    )


def print_score_comparison(scores_by_mk, coverage=None, structure=None,
                           is_supervised_map=None, _suppress_banner=False):
    """Render the AC cross-model F1 comparison.

    Layout: one table per acctype-scope. Each table has:
      col 1 = kshot, col 2 = model, col 3 = task, col 4 = TOTAL (scope F1),
      and one column per factor (factor's F1 within that scope).
    Rows are sorted by (kshot, model, task).

    Sections emitted:
      - "AC OVERALL"        — TOTAL=factors.overall,        factor cols=per_factor[f].overall
      - "[<acctype>]" each  — TOTAL=per_acc_type[atype],    factor cols=per_factor[f].per_acc_type[atype]

    Diff sections appended after the main tables:
      - "TASK DIFF (AC - AC1)" — per (model, kshot), F1 delta where both
        tasks exist for that model+kshot. NaN when only one task ran.
      - "CLASS DIFF (supervised - LLM)" — per (task, kshot), F1 delta of the
        mean-supervised F1 vs the mean-LLM F1. Requires `is_supervised_map`
        with at least one of each class; otherwise skipped.

    Cell rendering: numeric F1 → "  72.34"; explicit skip ("f1":"NaN") OR missing → "--".
    Diff cells render with explicit sign ("+5.23" / "-3.18") so direction is
    unambiguous; "--" when either side is missing.

    `scores_by_mk` maps (model, task, k_str) -> the dict returned by compute_AC_scores.
    `coverage` (optional) maps (model, task, k_str) -> [str, ...] listing which
    (task/dataset) pairs contributed; printed as a footer.
    `is_supervised_map` (optional) maps model_key -> bool. Required for the
    CLASS DIFF section; absent entries default to False (treated as LLM).
    """
    if not scores_by_mk:
        return

    models  = sorted({m for m, _, _ in scores_by_mk})
    tasks   = sorted({t for _, t, _ in scores_by_mk})
    shots   = sorted({k for _, _, k in scores_by_mk}, key=int)
    atypes  = sorted({
        a for sc in scores_by_mk.values()
        for a in (sc.get("factors", {}).get("per_acc_type") or {})
    })
    all_factors = {
        f for sc in scores_by_mk.values()
        for f in (sc.get("factors", {}).get("per_factor") or {})
    }
    # Drop factors that are skipped in EVERY (model, task, shot) entry —
    # i.e. their per_factor.overall.f1 is the literal "NaN" string (set by
    # compute_AC_flat_score when the factor is in skip_columns). This
    # hides always-skipped columns like accident_report / id / accident_type
    # so the table stays focused on actually-evaluated factors. A factor
    # that's skipped for SOME models but evaluated for others stays
    # (useful for comparison).
    def _has_real_score(factor):
        for sc in scores_by_mk.values():
            node = ((sc.get("factors", {}).get("per_factor") or {}).get(factor) or {}).get("overall", {})
            v = node.get("f1")
            if v is not None and not isinstance(v, str):
                return True
        return False
    all_factors = {f for f in all_factors if _has_real_score(f)}
    # Order: main factor, then its subs (per `structure`), then any unknown factors.
    # `structure` shape: {"accident_report": [main, ...], main: [sub, ...], ...}
    main_to_subs = {m: list(structure.get(m, [])) for m in (structure or {}).get("accident_report", [])} if structure else {}
    is_main = set(main_to_subs.keys())
    is_sub  = {s for subs in main_to_subs.values() for s in subs}
    factors = []
    for main, subs in main_to_subs.items():
        if main in all_factors:
            factors.append(main)
        for sub in subs:
            if sub in all_factors:
                factors.append(sub)
    # Append anything not categorised by structure (alphabetical for stability).
    leftovers = sorted(f for f in all_factors if f not in is_main and f not in is_sub)
    factors.extend(leftovers)
    # Hoist accident_type to the front so it sits immediately after TOTAL.
    if "accident_type" in factors:
        factors.remove("accident_type")
        factors.insert(0, "accident_type")

    # Sort by (task, kshot, model). task alphabetical, kshot numeric, model
    # uses natural sort (so "Qwen3.5-9B" < "Qwen3.5-27B").
    def _natural_key(s):
        return tuple(int(t) if t.isdigit() else t.lower()
                     for t in re.split(r'(\d+)', s))
    rows = sorted(scores_by_mk.keys(),
                  key=lambda mtk: (mtk[1], int(mtk[2]), _natural_key(mtk[0])))
    if not rows:
        return

    def cell_text(node):
        # --   no entry OR explicitly-skipped — compute_AC_scores either didn't
        #      produce a stat for this (factor, acc_type) at all (missing) or
        #      it landed in skip_columns (f1 == "NaN" string). Merged visually
        #      so the table stays uncluttered; the underlying scores JSON still
        #      preserves the distinction.
        # 0.00 entry exists with f1 == 0 (gold or predictions present, none correct).
        if not node:
            return "--"
        v = node.get("f1")
        if isinstance(v, str) or v is None:
            return "--"
        return f"{v:6.2f}"

    def cell_pr(node, key):
        """Precision/recall cell with `(matched/denom)` audit fraction —
        same format as the SUMMARY blocks. e.g. "29.27 (15/52)".
        `--` for missing entries and explicitly-skipped ones (NaN merged in).
        """
        if not node:
            return "--"
        v = node.get(key)
        if isinstance(v, str) or v is None:
            return "--"
        if key == "precision":
            num, denom = node.get("match_num"), node.get("pred_num")
        elif key == "recall":
            num, denom = node.get("match_num"), node.get("gold_num")
        else:
            return f"{v:.2f}"
        if isinstance(num, (int, float)) and isinstance(denom, (int, float)) and denom:
            return f"{v:.2f} ({int(num)}/{int(denom)})"
        return f"{v:.2f}"

    # Decorate factor names: MAINS in UPPERCASE, subs prefixed with " ↳ ".
    def decorate(f):
        if f in is_main:
            return f.upper()
        if f in is_sub:
            return " ↳" + f
        return f

    val_w   = 7
    name_w  = max(len("model"), max(len(m) for m in models))
    task_w  = max(len("task"), max(len(t) for t in tasks))
    k_w     = max(len("kshot"), max(len(k) for k in shots))
    fac_w   = {f: max(val_w, len(decorate(f))) for f in factors}

    SEP = " "

    def render_section(title, total_getter, factor_getter):
        """Per-factor matrix with a two-row header.

        Layout:
          row 1 of header: TOTAL + one block per factor (centered factor name).
          row 2 of header: "P  R  F1" sub-columns under each factor block.

        Each data row stays a single line — three sub-cells per factor
        (precision, recall, F1) so readers can scan all three side-by-side.

        Cell formatting:
          P, R → `value (matched/denom)`  e.g. "29.27 (15/52)"
          F1   → bare percent              e.g. "29.27"
        """
        metric_labels = [("P", "precision"), ("R", "recall"), ("F1", "f1")]
        def fmt_cell(node, key):
            return cell_text(node) if key == "f1" else cell_pr(node, key)

        # Sub-cell widths per (factor, metric). Take max over all rows.
        sub_w = {}
        for f in factors:
            for lbl, key in metric_labels:
                widest = len(lbl)
                for (m, t, k) in rows:
                    node = factor_getter(scores_by_mk[(m, t, k)], f)
                    widest = max(widest, len(fmt_cell(node, key)))
                sub_w[(f, key)] = widest
        # Factor block must also fit the decorated factor name in row 1.
        # Block width = sum(sub widths) + 2 spaces (between sub-cells).
        SUB_SEP = " "
        block_w = {}
        for f in factors:
            inner = SUB_SEP.join(" " * sub_w[(f, key)] for _, key in metric_labels)
            block_w[f] = max(len(decorate(f)), len(inner))
            # If the factor name is wider than the inner block, expand the
            # last sub-column to fill the gap so columns still align.
            gap = block_w[f] - len(inner)
            if gap > 0:
                last_key = metric_labels[-1][1]
                sub_w[(f, last_key)] += gap

        # TOTAL also gets P/R/F1 sub-cells using the same formatting.
        total_sub_w = {}
        for lbl, key in metric_labels:
            widest = len(lbl)
            for (m, t, k) in rows:
                tn = total_getter(scores_by_mk[(m, t, k)])
                widest = max(widest, len(fmt_cell(tn, key)))
            total_sub_w[key] = widest
        total_inner = SUB_SEP.join(" " * total_sub_w[k] for _, k in metric_labels)
        total_block_w = max(len("TOTAL"), len(total_inner))
        gap = total_block_w - len(total_inner)
        if gap > 0:
            total_sub_w[metric_labels[-1][1]] += gap

        # Build the two header rows.
        head1_left = (
            "kshot".rjust(k_w) + SEP +
            "model".ljust(name_w) + SEP +
            "task".ljust(task_w) + SEP +
            "TOTAL".center(total_block_w)
        )
        head2_left = (
            " " * k_w + SEP +
            " " * name_w + SEP +
            " " * task_w + SEP +
            SUB_SEP.join(lbl.rjust(total_sub_w[key]) for lbl, key in metric_labels)
        )
        head1_right = ""
        head2_right = ""
        for f in factors:
            head1_right += SEP + decorate(f).center(block_w[f])
            head2_right += SEP + SUB_SEP.join(
                lbl.rjust(sub_w[(f, key)]) for lbl, key in metric_labels)
        header1 = head1_left + head1_right
        header2 = head2_left + head2_right
        line = "-" * max(len(header1), len(header2))

        print(f"\n  {title}")
        print("  " + line)
        print("  " + header1)
        print("  " + header2)
        print("  " + line)
        for (m, t, k) in rows:
            sc = scores_by_mk[(m, t, k)]
            total_node = total_getter(sc)
            total_cells = SUB_SEP.join(
                fmt_cell(total_node, key).rjust(total_sub_w[key])
                for _, key in metric_labels)
            row = (
                k.rjust(k_w) + SEP +
                m.ljust(name_w) + SEP +
                t.ljust(task_w) + SEP +
                total_cells
            )
            for f in factors:
                node = factor_getter(sc, f)
                row += SEP + SUB_SEP.join(
                    fmt_cell(node, key).rjust(sub_w[(f, key)])
                    for _, key in metric_labels)
            print("  " + row)
        print("  " + line)

    # Summary section: one row per (kshot, model, task) showing scope-level
    # precision / recall / F1 for the same scope_getter used by the matching
    # per-factor table below. Lives above the detailed tables so readers see
    # the headline numbers before drilling into individual factor columns.
    # Precision/recall cells append the raw match/denom ratio (e.g.
    # "29.27 (15/52)") so the reader can audit the percentage against the
    # underlying counts. F1's natural denominator (pred+gold) would print a
    # number that doesn't reconcile to the displayed F1, so it's left as a
    # bare percentage.
    def cell_metric(node, key):
        if not node:
            return "--"
        v = node.get(key)
        if isinstance(v, str) or v is None:
            return "--"
        if key == "precision":
            num, denom = node.get("match_num"), node.get("pred_num")
        elif key == "recall":
            num, denom = node.get("match_num"), node.get("gold_num")
        else:
            return f"{v:.2f}"
        if isinstance(num, (int, float)) and isinstance(denom, (int, float)) and denom:
            return f"{v:.2f} ({int(num)}/{int(denom)})"
        return f"{v:.2f}"

    # Pre-compute per-column widths so the ratio strings line up. Walk every
    # row to find the widest "(m/d)" tail we'll need to fit.
    summary_cols = [("precision", "precision"), ("recall", "recall"), ("F1", "f1")]

    def _summary_col_widths(scope_getter):
        widths = {}
        for label, key in summary_cols:
            widest = max(val_w, len(label))
            for (m, t, k) in rows:
                node = scope_getter(scores_by_mk[(m, t, k)])
                widest = max(widest, len(cell_metric(node, key)))
            widths[key] = widest
        return widths

    def render_summary(title, scope_getter):
        widths = _summary_col_widths(scope_getter)
        header = (
            "kshot".rjust(k_w) + SEP +
            "model".ljust(name_w) + SEP +
            "task".ljust(task_w)
        )
        for label, key in summary_cols:
            header += SEP + label.rjust(widths[key])
        line = "-" * len(header)
        print(f"\n  {title}")
        print("  " + line)
        print("  " + header)
        print("  " + line)
        # Sort by scope F1 desc — surfaces the leaderboard at a glance.
        # Missing/NaN F1 sinks to the bottom; ties fall back to the global
        # (task, kshot, model) ordering for a stable read.
        def _f1_sort_key(mtk):
            node = scope_getter(scores_by_mk[mtk])
            if not node:
                return (1, 0.0)
            v = node.get("f1")
            if isinstance(v, str) or v is None:
                return (1, 0.0)
            return (0, -float(v))
        sorted_rows = sorted(
            rows,
            key=lambda mtk: (
                _f1_sort_key(mtk),
                mtk[1], int(mtk[2]), _natural_key(mtk[0]),
            ),
        )
        for (m, t, k) in sorted_rows:
            node = scope_getter(scores_by_mk[(m, t, k)])
            row = (
                k.rjust(k_w) + SEP +
                m.ljust(name_w) + SEP +
                t.ljust(task_w)
            )
            for _, key in summary_cols:
                row += SEP + cell_metric(node, key).rjust(widths[key])
            print("  " + row)
        print("  " + line)

    if not _suppress_banner:
        print("\n" + "=" * 78)
        print(" CROSS-MODEL F1 COMPARISON ".center(78, "="))
        print("=" * 78)

    # 1) Summary block (precision / F1 / recall) — OVERALL and each acctype.
    overall_getter = lambda sc: sc.get("factors", {}).get("overall")
    render_summary("SUMMARY  (OVERALL)", overall_getter)
    for atype in atypes:
        render_summary(
            f"SUMMARY  [{atype}]",
            lambda sc, a=atype: (sc.get("factors", {}).get("per_acc_type") or {}).get(a),
        )

    # 1b) Per-factor ranking — one render_summary per factor, sorted by that
    # factor's F1 desc. Factors with no real F1 across any run are suppressed
    # so the section stays focused. Uses the same precision/recall/F1 layout
    # as the OVERALL summary so readers can compare counts side-by-side.
    for f in factors:
        per_factor_getter = (
            lambda sc, fk=f:
            ((sc.get("factors", {}).get("per_factor") or {}).get(fk) or {}).get("overall")
        )
        # Skip the table when no run has a numeric F1 for this factor
        # (every cell would be -- or NaN).
        has_data = any(
            isinstance((per_factor_getter(scores_by_mk[mtk]) or {}).get("f1"), (int, float))
            for mtk in rows
        )
        if not has_data:
            continue
        render_summary(f"RANK [{decorate(f)}]", per_factor_getter)

    # 2) Per-factor breakdown (F1 only, one column per factor) — same scopes.
    # One matrix per scope — each (model, task, kshot) entry gets three
    # stacked rows (P / R / F1) so precision, recall, and F1 sit
    # side-by-side per factor without flipping between separate tables.
    overall_factor_getter = lambda sc, fk: (
        (sc.get("factors", {}).get("per_factor") or {}).get(fk) or {}
    ).get("overall")
    render_section("AC OVERALL", overall_getter, overall_factor_getter)
    for atype in atypes:
        atype_total_getter = (
            lambda sc, a=atype: (sc.get("factors", {}).get("per_acc_type") or {}).get(a)
        )
        atype_factor_getter = (
            lambda sc, fk, a=atype:
                (((sc.get("factors", {}).get("per_factor") or {}).get(fk) or {})
                 .get("per_acc_type") or {}).get(a)
        )
        render_section(f"[{atype}]", atype_total_getter, atype_factor_getter)

    # 3) Diff tables. Helpers shared by both.
    #
    # `_get_f1` pulls a numeric F1 out of a (scope) score node — returns None
    # for missing nodes or explicit "NaN" strings (skipped factors). The diff
    # renderers fall back to "--" in that case so the reader can distinguish
    # "not computable" from a real numeric delta.
    def _get_f1(node):
        if not node:
            return None
        v = node.get("f1")
        if isinstance(v, str) or v is None:
            return None
        return float(v)

    def _scope_overall_f1(sc):
        return _get_f1(sc.get("factors", {}).get("overall"))

    def _scope_factor_f1(sc, fk):
        return _get_f1(((sc.get("factors", {}).get("per_factor") or {}).get(fk) or {}).get("overall"))

    def _diff_cell(a, b):
        if a is None or b is None:
            return "--"
        return f"{a - b:+6.2f}"

    # 3a) TASK DIFF (AC − AC1) — one row per (model, kshot) where BOTH tasks
    # exist for that model+kshot. Helps spot whether the AC1 prompt format
    # consistently helps or hurts a particular model.
    if "AC" in tasks and "AC1" in tasks:
        # (model, kshot) pairs that have AT LEAST ONE task entry. Cells
        # render "--" when the other side is missing — surfaces partial
        # sweeps rather than hiding them.
        mk_pairs = sorted(
            {(m, k) for (m, _, k) in scores_by_mk.keys()},
            key=lambda mk: (int(mk[1]), _natural_key(mk[0])),
        )
        header = (
            "kshot".rjust(k_w) + SEP +
            "model".ljust(name_w) + SEP +
            "task".ljust(task_w) + SEP +
            "TOTAL".rjust(val_w)
        )
        for f in factors:
            header += SEP + decorate(f).rjust(fac_w[f])
        line = "-" * len(header)
        print("\n  TASK DIFF (AC - AC1, per model+kshot)")
        print("  " + line)
        print("  " + header)
        print("  " + line)
        for (m, k) in mk_pairs:
            ac  = scores_by_mk.get((m, "AC",  k))
            ac1 = scores_by_mk.get((m, "AC1", k))
            row = (
                k.rjust(k_w) + SEP +
                m.ljust(name_w) + SEP +
                "diff".ljust(task_w) + SEP +
                _diff_cell(_scope_overall_f1(ac) if ac else None,
                           _scope_overall_f1(ac1) if ac1 else None).rjust(val_w)
            )
            for f in factors:
                row += SEP + _diff_cell(
                    _scope_factor_f1(ac, f)  if ac  else None,
                    _scope_factor_f1(ac1, f) if ac1 else None,
                ).rjust(fac_w[f])
            print("  " + row)
        print("  " + line)

    # 3b) CLASS DIFF (supervised − LLM, mean within class) — one row per
    # (task, kshot). Averaging within class loses per-model detail (drill
    # into the AC OVERALL table for that), but a single delta makes the
    # supervised-vs-LLM gap obvious at a glance.
    sup_map = is_supervised_map or {}
    sup_models = {m for m, is_sup in sup_map.items() if is_sup}
    llm_models = {m for m, is_sup in sup_map.items() if not is_sup}
    if sup_models and llm_models:
        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        tk_pairs = sorted(
            {(t, k) for (_, t, k) in scores_by_mk.keys()},
            key=lambda tk: (tk[0], int(tk[1])),
        )
        header = (
            "kshot".rjust(k_w) + SEP +
            "model".ljust(name_w) + SEP +
            "task".ljust(task_w) + SEP +
            "TOTAL".rjust(val_w)
        )
        for f in factors:
            header += SEP + decorate(f).rjust(fac_w[f])
        line = "-" * len(header)
        print("\n  CLASS DIFF (mean supervised - mean LLM, per task+kshot)")
        print("  " + line)
        print("  " + header)
        print("  " + line)
        for (t, k) in tk_pairs:
            sup_scs = [scores_by_mk.get((m, t, k)) for m in sup_models]
            llm_scs = [scores_by_mk.get((m, t, k)) for m in llm_models]
            sup_scs = [s for s in sup_scs if s is not None]
            llm_scs = [s for s in llm_scs if s is not None]
            if not sup_scs or not llm_scs:
                continue  # no comparison possible at this (task, kshot)
            row = (
                k.rjust(k_w) + SEP +
                "sup-LLM".ljust(name_w) + SEP +
                t.ljust(task_w) + SEP +
                _diff_cell(_mean(_scope_overall_f1(s) for s in sup_scs),
                           _mean(_scope_overall_f1(s) for s in llm_scs)).rjust(val_w)
            )
            for f in factors:
                row += SEP + _diff_cell(
                    _mean(_scope_factor_f1(s, f) for s in sup_scs),
                    _mean(_scope_factor_f1(s, f) for s in llm_scs),
                ).rjust(fac_w[f])
            print("  " + row)
        print("  " + line)

    if coverage:
        print("\n  coverage (task/dataset contributing to each cell):")
        for (m, t, k) in rows:
            ds = coverage.get((m, t, k), [])
            if ds:
                print(f"    {m}  {t}  k={k}: {', '.join(ds)}")
    print("=" * 78)

def compute_ED_scores(preds, golds, metrics={"trigger_id", "trigger_cls"}):
    assert len(preds) == len(golds)
    scores = {}
    if "trigger_id" in metrics:
        scores["trigger_id"] = compute_ED_trigger_id_score(preds, golds)
    if "trigger_cls" in metrics:
        scores["trigger_cls"] = compute_ED_trigger_cls_score(preds, golds)
    return scores

def compute_EAE_scores(preds, golds, metrics={"argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"}):
    assert len(preds) == len(golds)
    scores = {}
    if "argument_id" in metrics:
        scores["argument_id"] = compute_EAE_argument_id_score(preds, golds)
    if "argument_cls" in metrics:
        scores["argument_cls"] = compute_EAE_argument_cls_score(preds, golds)
    if "argument_attached_id" in metrics:
        scores["argument_attached_id"] = compute_EAE_argument_attached_id_score(preds, golds)
    if "argument_attached_cls" in metrics:
        scores["argument_attached_cls"] = compute_EAE_argument_attached_cls_score(preds, golds)
    return scores

def compute_EARL_scores(preds, golds, metrics={"argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"}):
    assert len(preds) == len(golds)
    scores = {}
    if "argument_id" in metrics:
        scores["argument_id"] = compute_EARL_argument_id_score(preds, golds)
    if "argument_cls" in metrics:
        scores["argument_cls"] = compute_EARL_argument_cls_score(preds, golds)
    if "argument_attached_id" in metrics:
        scores["argument_attached_id"] = compute_EARL_argument_attached_id_score(preds, golds)
    if "argument_attached_cls" in metrics:
        scores["argument_attached_cls"] = compute_EARL_argument_attached_cls_score(preds, golds)
    return scores

def compute_E2E_scores(preds, golds, metrics={"trigger_id", "trigger_cls", "argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"}):
    assert len(preds) == len(golds)
    scores = {}
    if "trigger_id" in metrics:
        scores["trigger_id"] = compute_E2E_trigger_id_score(preds, golds)
    if "trigger_cls" in metrics:
        scores["trigger_cls"] = compute_E2E_trigger_cls_score(preds, golds)
    if "argument_id" in metrics:
        scores["argument_id"] = compute_E2E_argument_id_score(preds, golds)
    if "argument_cls" in metrics:
        scores["argument_cls"] = compute_E2E_argument_cls_score(preds, golds)
    if "argument_attached_id" in metrics:
        scores["argument_attached_id"] = compute_E2E_argument_attached_id_score(preds, golds)
    if "argument_attached_cls" in metrics:
        scores["argument_attached_cls"] = compute_E2E_argument_attached_cls_score(preds, golds)
    return scores

def compute_ED_trigger_id_score(preds, golds):
    gold_tri_id, pred_tri_id = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        gold_tri_id_ = [(gold["doc_id"], gold["wnd_id"], t[0], t[1]) for t in gold["triggers"]]
        pred_tri_id_ = [(pred["doc_id"], pred["wnd_id"], t[0], t[1]) for t in pred["triggers"]]
        gold_tri_id.extend(gold_tri_id_)
        pred_tri_id.extend(pred_tri_id_)
        
    gold_tri_id = set(gold_tri_id)
    pred_tri_id = set(pred_tri_id)
    tri_id_f1 = compute_f1(len(pred_tri_id), len(gold_tri_id), len(gold_tri_id & pred_tri_id))
    scores = {
        "pred_num": len(pred_tri_id), 
        "gold_num": len(gold_tri_id), 
        "match_num": len(gold_tri_id & pred_tri_id), 
        "precision": tri_id_f1[0], 
        "recall": tri_id_f1[1], 
        "f1": tri_id_f1[2], 
    }
    return scores

def compute_ED_trigger_cls_score(preds, golds):
    gold_tri_cls, pred_tri_cls = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        gold_tri_cls_ = [(gold["doc_id"], gold["wnd_id"], t[0], t[1], t[2]) for t in gold["triggers"]]
        pred_tri_cls_ = [(pred["doc_id"], pred["wnd_id"], t[0], t[1], t[2]) for t in pred["triggers"]]
        gold_tri_cls.extend(gold_tri_cls_)
        pred_tri_cls.extend(pred_tri_cls_)
        
    gold_tri_cls = set(gold_tri_cls)
    pred_tri_cls = set(pred_tri_cls)
    tri_cls_f1 = compute_f1(len(pred_tri_cls), len(gold_tri_cls), len(gold_tri_cls & pred_tri_cls))
    scores = {
        "pred_num": len(pred_tri_cls), 
        "gold_num": len(gold_tri_cls), 
        "match_num": len(gold_tri_cls & pred_tri_cls), 
        "precision": tri_cls_f1[0], 
        "recall": tri_cls_f1[1], 
        "f1": tri_cls_f1[2], 
    }
    return scores

def compute_EAE_argument_id_score(preds, golds):
    gold_arg_id, pred_arg_id = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        assert pred["trigger"][0] == gold["trigger"][0]
        assert pred["trigger"][1] == gold["trigger"][1]
        assert pred["trigger"][2] == gold["trigger"][2]
        gold_arg_id_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][2], r[0], r[1]) for r in gold["arguments"]]
        pred_arg_id_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][2], r[0], r[1]) for r in pred["arguments"]]
        gold_arg_id.extend(gold_arg_id_)
        pred_arg_id.extend(pred_arg_id_)
        
    gold_arg_id = set(gold_arg_id)
    pred_arg_id = set(pred_arg_id)
    arg_id_f1 = compute_f1(len(pred_arg_id), len(gold_arg_id), len(gold_arg_id & pred_arg_id))
    scores = {
        "pred_num": len(pred_arg_id), 
        "gold_num": len(gold_arg_id), 
        "match_num": len(gold_arg_id & pred_arg_id), 
        "precision": arg_id_f1[0], 
        "recall": arg_id_f1[1], 
        "f1": arg_id_f1[2], 
    }
    return scores

def compute_EAE_argument_cls_score(preds, golds):
    gold_arg_cls, pred_arg_cls = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        assert pred["trigger"][0] == gold["trigger"][0]
        assert pred["trigger"][1] == gold["trigger"][1]
        assert pred["trigger"][2] == gold["trigger"][2]
        gold_arg_cls_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][2], r[0], r[1], r[2]) for r in gold["arguments"]]
        pred_arg_cls_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][2], r[0], r[1], r[2]) for r in pred["arguments"]]
        gold_arg_cls.extend(gold_arg_cls_)
        pred_arg_cls.extend(pred_arg_cls_)
        
    gold_arg_cls = set(gold_arg_cls)
    pred_arg_cls = set(pred_arg_cls)
    arg_cls_f1 = compute_f1(len(pred_arg_cls), len(gold_arg_cls), len(gold_arg_cls & pred_arg_cls))
    scores = {
        "pred_num": len(pred_arg_cls), 
        "gold_num": len(gold_arg_cls), 
        "match_num": len(gold_arg_cls & pred_arg_cls), 
        "precision": arg_cls_f1[0], 
        "recall": arg_cls_f1[1], 
        "f1": arg_cls_f1[2], 
    }
    return scores

def compute_EAE_argument_attached_id_score(preds, golds):
    gold_arg_id, pred_arg_id = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        assert pred["trigger"][0] == gold["trigger"][0]
        assert pred["trigger"][1] == gold["trigger"][1]
        assert pred["trigger"][2] == gold["trigger"][2]
        gold_arg_id_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][0], gold["trigger"][1], gold["trigger"][2], r[0], r[1]) for r in gold["arguments"]]
        pred_arg_id_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][0], pred["trigger"][1], pred["trigger"][2], r[0], r[1]) for r in pred["arguments"]]
        gold_arg_id.extend(gold_arg_id_)
        pred_arg_id.extend(pred_arg_id_)
        
    gold_arg_id = set(gold_arg_id)
    pred_arg_id = set(pred_arg_id)
    arg_id_f1 = compute_f1(len(pred_arg_id), len(gold_arg_id), len(gold_arg_id & pred_arg_id))
    scores = {
        "pred_num": len(pred_arg_id), 
        "gold_num": len(gold_arg_id), 
        "match_num": len(gold_arg_id & pred_arg_id), 
        "precision": arg_id_f1[0], 
        "recall": arg_id_f1[1], 
        "f1": arg_id_f1[2], 
    }
    return scores

def compute_EAE_argument_attached_cls_score(preds, golds):
    gold_arg_cls, pred_arg_cls = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        assert pred["trigger"][0] == gold["trigger"][0]
        assert pred["trigger"][1] == gold["trigger"][1]
        assert pred["trigger"][2] == gold["trigger"][2]
        gold_arg_cls_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][0], gold["trigger"][1], gold["trigger"][2], r[0], r[1], r[2]) for r in gold["arguments"]]
        pred_arg_cls_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][0], pred["trigger"][1], pred["trigger"][2], r[0], r[1], r[2]) for r in pred["arguments"]]
        gold_arg_cls.extend(gold_arg_cls_)
        pred_arg_cls.extend(pred_arg_cls_)
        
    gold_arg_cls = set(gold_arg_cls)
    pred_arg_cls = set(pred_arg_cls)
    arg_cls_f1 = compute_f1(len(pred_arg_cls), len(gold_arg_cls), len(gold_arg_cls & pred_arg_cls))
    scores = {
        "pred_num": len(pred_arg_cls), 
        "gold_num": len(gold_arg_cls), 
        "match_num": len(gold_arg_cls & pred_arg_cls), 
        "precision": arg_cls_f1[0], 
        "recall": arg_cls_f1[1], 
        "f1": arg_cls_f1[2], 
    }
    return scores

def compute_EARL_argument_id_score(preds, golds):
    gold_arg_id, pred_arg_id = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        assert pred["trigger"][0] == gold["trigger"][0]
        assert pred["trigger"][1] == gold["trigger"][1]
        assert pred["trigger"][2] == gold["trigger"][2]
        assert len(pred["arguments"]) == len(gold["arguments"])
        gold_arg_id_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][2], r[0], r[1]) for r in gold["arguments"] if r[2] is not None]
        pred_arg_id_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][2], r[0], r[1]) for r in pred["arguments"] if r[2] is not None]
        gold_arg_id.extend(gold_arg_id_)
        pred_arg_id.extend(pred_arg_id_)
        
    gold_arg_id = set(gold_arg_id)
    pred_arg_id = set(pred_arg_id)
    arg_id_f1 = compute_f1(len(pred_arg_id), len(gold_arg_id), len(gold_arg_id & pred_arg_id))
    scores = {
        "pred_num": len(pred_arg_id), 
        "gold_num": len(gold_arg_id), 
        "match_num": len(gold_arg_id & pred_arg_id), 
        "precision": arg_id_f1[0], 
        "recall": arg_id_f1[1], 
        "f1": arg_id_f1[2], 
    }
    return scores

def compute_EARL_argument_cls_score(preds, golds):
    gold_arg_cls, pred_arg_cls = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        assert pred["trigger"][0] == gold["trigger"][0]
        assert pred["trigger"][1] == gold["trigger"][1]
        assert pred["trigger"][2] == gold["trigger"][2]
        gold_arg_cls_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][2], r[0], r[1], r[2]) for r in gold["arguments"] if r[2] is not None]
        pred_arg_cls_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][2], r[0], r[1], r[2]) for r in pred["arguments"] if r[2] is not None]
        gold_arg_cls.extend(gold_arg_cls_)
        pred_arg_cls.extend(pred_arg_cls_)
        
    gold_arg_cls = set(gold_arg_cls)
    pred_arg_cls = set(pred_arg_cls)
    arg_cls_f1 = compute_f1(len(pred_arg_cls), len(gold_arg_cls), len(gold_arg_cls & pred_arg_cls))
    scores = {
        "pred_num": len(pred_arg_cls), 
        "gold_num": len(gold_arg_cls), 
        "match_num": len(gold_arg_cls & pred_arg_cls), 
        "precision": arg_cls_f1[0], 
        "recall": arg_cls_f1[1], 
        "f1": arg_cls_f1[2], 
    }
    return scores

def compute_EARL_argument_attached_id_score(preds, golds):
    gold_arg_id, pred_arg_id = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        assert pred["trigger"][0] == gold["trigger"][0]
        assert pred["trigger"][1] == gold["trigger"][1]
        assert pred["trigger"][2] == gold["trigger"][2]
        gold_arg_id_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][0], gold["trigger"][1], gold["trigger"][2], r[0], r[1]) for r in gold["arguments"] if r[2] is not None]
        pred_arg_id_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][0], pred["trigger"][1], pred["trigger"][2], r[0], r[1]) for r in pred["arguments"] if r[2] is not None]
        gold_arg_id.extend(gold_arg_id_)
        pred_arg_id.extend(pred_arg_id_)
        
    gold_arg_id = set(gold_arg_id)
    pred_arg_id = set(pred_arg_id)
    arg_id_f1 = compute_f1(len(pred_arg_id), len(gold_arg_id), len(gold_arg_id & pred_arg_id))
    scores = {
        "pred_num": len(pred_arg_id), 
        "gold_num": len(gold_arg_id), 
        "match_num": len(gold_arg_id & pred_arg_id), 
        "precision": arg_id_f1[0], 
        "recall": arg_id_f1[1], 
        "f1": arg_id_f1[2], 
    }
    return scores

def compute_EARL_argument_attached_cls_score(preds, golds):
    gold_arg_cls, pred_arg_cls = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        assert pred["trigger"][0] == gold["trigger"][0]
        assert pred["trigger"][1] == gold["trigger"][1]
        assert pred["trigger"][2] == gold["trigger"][2]
        assert len(pred["arguments"]) == len(gold["arguments"])
        gold_arg_cls_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][0], gold["trigger"][1], gold["trigger"][2], r[0], r[1], r[2]) for r in gold["arguments"] if r[2] is not None]
        pred_arg_cls_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][0], pred["trigger"][1], pred["trigger"][2], r[0], r[1], r[2]) for r in pred["arguments"] if r[2] is not None]
        gold_arg_cls.extend(gold_arg_cls_)
        pred_arg_cls.extend(pred_arg_cls_)
        
    gold_arg_cls = set(gold_arg_cls)
    pred_arg_cls = set(pred_arg_cls)
    arg_cls_f1 = compute_f1(len(pred_arg_cls), len(gold_arg_cls), len(gold_arg_cls & pred_arg_cls))
    scores = {
        "pred_num": len(pred_arg_cls), 
        "gold_num": len(gold_arg_cls), 
        "match_num": len(gold_arg_cls & pred_arg_cls), 
        "precision": arg_cls_f1[0], 
        "recall": arg_cls_f1[1], 
        "f1": arg_cls_f1[2], 
    }
    return scores

def compute_E2E_trigger_id_score(preds, golds):
    gold_tri_id, pred_tri_id = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        gold_tri_id_ = [(gold["doc_id"], gold["wnd_id"], e["trigger"][0], e["trigger"][1]) for e in gold["events"]]
        pred_tri_id_ = [(pred["doc_id"], pred["wnd_id"], e["trigger"][0], e["trigger"][1]) for e in pred["events"]]
        gold_tri_id.extend(gold_tri_id_)
        pred_tri_id.extend(pred_tri_id_)
        
    gold_tri_id = set(gold_tri_id)
    pred_tri_id = set(pred_tri_id)
    tri_id_f1 = compute_f1(len(pred_tri_id), len(gold_tri_id), len(gold_tri_id & pred_tri_id))
    scores = {
        "pred_num": len(pred_tri_id), 
        "gold_num": len(gold_tri_id), 
        "match_num": len(gold_tri_id & pred_tri_id), 
        "precision": tri_id_f1[0], 
        "recall": tri_id_f1[1], 
        "f1": tri_id_f1[2], 
    }
    return scores

def compute_E2E_trigger_cls_score(preds, golds):
    gold_tri_cls, pred_tri_cls = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        gold_tri_cls_ = [(gold["doc_id"], gold["wnd_id"], e["trigger"][0], e["trigger"][1], e["trigger"][2]) for e in gold["events"]]
        pred_tri_cls_ = [(pred["doc_id"], pred["wnd_id"], e["trigger"][0], e["trigger"][1], e["trigger"][2]) for e in pred["events"]]
        gold_tri_cls.extend(gold_tri_cls_)
        pred_tri_cls.extend(pred_tri_cls_)
        
    gold_tri_cls = set(gold_tri_cls)
    pred_tri_cls = set(pred_tri_cls)
    tri_cls_f1 = compute_f1(len(pred_tri_cls), len(gold_tri_cls), len(gold_tri_cls & pred_tri_cls))
    scores = {
        "pred_num": len(pred_tri_cls), 
        "gold_num": len(gold_tri_cls), 
        "match_num": len(gold_tri_cls & pred_tri_cls), 
        "precision": tri_cls_f1[0], 
        "recall": tri_cls_f1[1], 
        "f1": tri_cls_f1[2], 
    }
    return scores

def compute_E2E_argument_id_score(preds, golds):
    gold_arg_id, pred_arg_id = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        for event in gold["events"]:
            gold_arg_id_ = [(gold["doc_id"], gold["wnd_id"], event["trigger"][2], r[0], r[1]) for r in event["arguments"]]
            gold_arg_id.extend(gold_arg_id_)
        for event in pred["events"]:
            pred_arg_id_ = [(pred["doc_id"], pred["wnd_id"], event["trigger"][2], r[0], r[1]) for r in event["arguments"]]
            pred_arg_id.extend(pred_arg_id_)
        
    gold_arg_id = set(gold_arg_id)
    pred_arg_id = set(pred_arg_id)
    arg_id_f1 = compute_f1(len(pred_arg_id), len(gold_arg_id), len(gold_arg_id & pred_arg_id))
    scores = {
        "pred_num": len(pred_arg_id), 
        "gold_num": len(gold_arg_id), 
        "match_num": len(gold_arg_id & pred_arg_id), 
        "precision": arg_id_f1[0], 
        "recall": arg_id_f1[1], 
        "f1": arg_id_f1[2], 
    }
    return scores

def compute_E2E_argument_cls_score(preds, golds):
    gold_arg_cls, pred_arg_cls = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        for event in gold["events"]:
            gold_arg_cls_ = [(gold["doc_id"], gold["wnd_id"], event["trigger"][2], r[0], r[1], r[2]) for r in event["arguments"]]
            gold_arg_cls.extend(gold_arg_cls_)
        for event in pred["events"]:
            pred_arg_cls_ = [(pred["doc_id"], pred["wnd_id"], event["trigger"][2], r[0], r[1], r[2]) for r in event["arguments"]]
            pred_arg_cls.extend(pred_arg_cls_)
        
    gold_arg_cls = set(gold_arg_cls)
    pred_arg_cls = set(pred_arg_cls)
    arg_cls_f1 = compute_f1(len(pred_arg_cls), len(gold_arg_cls), len(gold_arg_cls & pred_arg_cls))
    scores = {
        "pred_num": len(pred_arg_cls), 
        "gold_num": len(gold_arg_cls), 
        "match_num": len(gold_arg_cls & pred_arg_cls), 
        "precision": arg_cls_f1[0], 
        "recall": arg_cls_f1[1], 
        "f1": arg_cls_f1[2], 
    }
    return scores

def compute_E2E_argument_attached_id_score(preds, golds):
    gold_arg_id, pred_arg_id = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        for event in gold["events"]:
            gold_arg_id_ = [(gold["doc_id"], gold["wnd_id"], event["trigger"][0], event["trigger"][1], event["trigger"][2], r[0], r[1]) for r in event["arguments"]]
            gold_arg_id.extend(gold_arg_id_)
        for event in pred["events"]:
            pred_arg_id_ = [(pred["doc_id"], pred["wnd_id"], event["trigger"][0], event["trigger"][1], event["trigger"][2], r[0], r[1]) for r in event["arguments"]]
            pred_arg_id.extend(pred_arg_id_)
        
    gold_arg_id = set(gold_arg_id)
    pred_arg_id = set(pred_arg_id)
    arg_id_f1 = compute_f1(len(pred_arg_id), len(gold_arg_id), len(gold_arg_id & pred_arg_id))
    scores = {
        "pred_num": len(pred_arg_id), 
        "gold_num": len(gold_arg_id), 
        "match_num": len(gold_arg_id & pred_arg_id), 
        "precision": arg_id_f1[0], 
        "recall": arg_id_f1[1], 
        "f1": arg_id_f1[2], 
    }
    return scores

def compute_E2E_argument_attached_cls_score(preds, golds):
    gold_arg_cls, pred_arg_cls = [], []
    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]
        for event in gold["events"]:
            gold_arg_cls_ = [(gold["doc_id"], gold["wnd_id"], event["trigger"][0], event["trigger"][1], event["trigger"][2], r[0], r[1], r[2]) for r in event["arguments"]]
            gold_arg_cls.extend(gold_arg_cls_)
        for event in pred["events"]:
            pred_arg_cls_ = [(pred["doc_id"], pred["wnd_id"], event["trigger"][0], event["trigger"][1], event["trigger"][2], r[0], r[1], r[2]) for r in event["arguments"]]
            pred_arg_cls.extend(pred_arg_cls_)
        
    gold_arg_cls = set(gold_arg_cls)
    pred_arg_cls = set(pred_arg_cls)
    arg_cls_f1 = compute_f1(len(pred_arg_cls), len(gold_arg_cls), len(gold_arg_cls & pred_arg_cls))
    scores = {
        "pred_num": len(pred_arg_cls), 
        "gold_num": len(gold_arg_cls), 
        "match_num": len(gold_arg_cls & pred_arg_cls), 
        "precision": arg_cls_f1[0], 
        "recall": arg_cls_f1[1], 
        "f1": arg_cls_f1[2], 
    }
    return scores

def safe_div(num, denom):
    return num / denom if denom > 0 else 0.0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision*100.0, recall*100.0, f1*100.0

# ------------------------------------------------------------------------------
# NEW AC TASK FUNCTIONS
# ------------------------------------------------------------------------------

# import numpy as np

def parse_ac_texts(text_input, norm=False):
    """Split an ACCD factor value on `;` and return the non-empty pieces.

    Default (`norm=False`): split on `;` only — pieces are returned verbatim
        (no whitespace strip, no case change, no punctuation removal).
    With `norm=True`: each piece is canonicalized via
        strip outer whitespace → lowercase → strip leading/trailing sentence
        punctuation (`. , ! ? : ;`) → collapse runs of internal whitespace.
        Use when the caller wants a canonical form for equality comparison
        and the ACCD annotation convention isn't consistent (e.g. work_ctx,
        which is ~45/55 with-vs-without trailing periods in gold).
    """
    if not text_input:
        return []
    if isinstance(text_input, list):
        items = text_input
    else:
        items = [text_input]

    res = []
    for item in items:
        if isinstance(item, (float, np.floating)) and np.isnan(item):
            continue
        for part in str(item).split(';'):
            if norm:
                p = re.sub(r'\s+', ' ',
                           part.strip().lower().strip('.,!?:;').strip())
            else:
                p = part
            if p:
                res.append(p)
    return res

def calculate_overlap(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def is_factor_match(p_texts, g_texts, overlap_pct=0.8):
    # Strict 1-to-1 bijection: every pred span pairs with a unique gold span
    # at overlap >= overlap_pct AND every gold span pairs with a unique pred
    # span at overlap >= overlap_pct. Requires |P| == |G|, and we find the
    # optimal pairing via Hungarian to avoid greedy starvation.
    if len(p_texts) != len(g_texts):
        return False
    n = len(p_texts)
    if n == 0:
        return True

    overlaps = np.array([[calculate_overlap(p, g) for g in g_texts] for p in p_texts])
    # Forbid sub-threshold edges with a large cost so Hungarian won't pick
    # them when an all-valid assignment exists. If it's forced to pick one
    # (no valid bijection), the post-check below will detect it.
    cost = np.where(overlaps >= overlap_pct, -overlaps, 1e9)
    row_ind, col_ind = linear_sum_assignment(cost)
    return bool(np.all(overlaps[row_ind, col_ind] >= overlap_pct))

def count_factor_matches(preds, golds, overlap_pct=0.8, strict_keys=None):
    """Count (record, factor) matches.

    `strict_keys`: factor keys whose match should be exact rather than
    `overlap_pct`-fuzzy. Used for classification factors like `severity` /
    `construction_trade` where partial overlap (e.g. "fatality" vs
    "fatalities" at 0.89) would otherwise count as a hit.
    """
    matched = 0
    gold_lookup = {(g[0], g[1], g[2]): g[3] for g in golds}
    strict_keys = set(strict_keys or [])

    for p in preds:
        p_head = (p[0], p[1], p[2])
        p_texts = p[3]
        factor_key = p[2]
        thr = 1.0 if factor_key in strict_keys else overlap_pct

        if p_head in gold_lookup:
            g_texts = gold_lookup[p_head]
            if is_factor_match(p_texts, g_texts, thr):
                matched += 1
    return matched

# def compute_f1(pred_num, gold_num, match_num):
#     precision = match_num / pred_num if pred_num > 0 else 0.0
#     recall = match_num / gold_num if gold_num > 0 else 0.0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#     return precision * 100, recall * 100, f1 * 100

def compute_AC_scores(preds, golds, metrics={"factors"}, sim_threshold=0.8,
                      skip_columns=None, norm=True, strict_keys=None):
    assert len(preds) == len(golds)
    scores = {}
    if "factors" in metrics:
        scores["factors"] = compute_AC_flat_score(preds, golds,
                                                  sim_threshold=sim_threshold,
                                                  skip_columns=skip_columns,
                                                  norm=norm,
                                                  strict_keys=strict_keys)
    return scores


def compute_AC_keyword_scores(preds, golds, metrics={"factors"}, skip_columns=None,
                              strict_keys=None):
    """Keyword-level companion to compute_AC_scores.

    Reads parallel `<factor>_keywords` lists from each gold record. A pred
    span matches a gold span iff it contains ALL of that span's keywords as
    whole-word case-insensitive substrings. See `compute_AC_flat_keyword_score`
    for the bijection logic.

    `strict_keys` lists classification-factor keys whose keyword is
    synthesized from the gold label itself (no `<factor>_keywords` needed),
    matching how `compute_AC_flat_score` already treats them as exact-match.
    """
    assert len(preds) == len(golds)
    scores = {}
    if "factors" in metrics:
        scores["factors"] = compute_AC_flat_keyword_score(
            preds, golds, skip_columns=skip_columns, strict_keys=strict_keys)
    return scores


def _normalize_for_keyword(s):
    """Lowercase + collapse internal whitespace runs. Used on both pred spans
    and gold keywords so whitespace variation (e.g. tabs vs spaces) doesn't
    block an otherwise-clean whole-word match."""
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _parse_keyword_groups(raw):
    """Normalize a gold `<factor>_keywords` value into list[list[str]].

    Accepted shapes:
        [[kw1, kw2, ...], [kw1, ...], ...]   list of inner-lists, one per span
        []                                    explicit empty
        None / missing                        no keywords
    Non-string keywords inside an inner list are coerced via str(); falsy
    entries (None / empty string) are dropped. Returns [] when the input
    isn't a list-of-lists at the outer level so callers can short-circuit.
    """
    if not raw or not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, list):
            out.append([str(k) for k in item if k not in (None, "")])
        elif item in (None, ""):
            out.append([])
        else:
            out.append([str(item)])
    return out


def is_factor_keyword_match(p_texts, g_keywords):
    """Strict 1-to-1 bijection where pred[i] matches gold[j] iff pred[i]
    contains every keyword in g_keywords[j] as a whole-word case-insensitive
    substring.

    Requires |P| == |G|; otherwise no bijection is possible. An empty gold
    keyword list at position j is treated as a vacuous match (pred[i] is
    always considered to contain "all zero" required keywords). Hungarian
    finds the optimal pairing; the post-check rejects any forced sub-threshold
    assignment.
    """
    if len(p_texts) != len(g_keywords):
        return False
    n = len(p_texts)
    if n == 0:
        return True

    norm_preds = [_normalize_for_keyword(p) for p in p_texts]
    norm_kws = [[_normalize_for_keyword(k) for k in group if k]
                for group in g_keywords]

    cost = np.full((n, n), 1e9, dtype=float)
    for i, p_norm in enumerate(norm_preds):
        for j, kws in enumerate(norm_kws):
            if all(re.search(r"\b" + re.escape(k) + r"\b", p_norm) for k in kws):
                cost[i, j] = -1.0
    row_ind, col_ind = linear_sum_assignment(cost)
    return bool(np.all(cost[row_ind, col_ind] < 0))


def compute_AC_flat_keyword_score(preds, golds, skip_columns=None, strict_keys=None):
    """Per-factor / per-acc_type keyword-match scoring.

    Matching pairs gold factors with pred factors by (doc_id, acc_type,
    factor_key) — same head-key as `compute_AC_flat_score` — then within each
    pair runs `is_factor_keyword_match` over the spans. Factors without a
    `<factor>_keywords` parallel list in gold are SKIPPED unless they're
    classification factors (listed in `strict_keys`), for which the keyword
    is synthesized from the gold label itself — same convention as
    `compute_AC_flat_score`'s exact-match handling. Pred factors are
    likewise dropped when the gold side has neither annotated nor
    synthesizable keywords.

    `skip_columns` mirrors `compute_AC_flat_score`: listed factor keys are
    excluded from gold/pred iteration AND surfaced in `per_factor` with NaN
    cells so renderers show them as explicitly-skipped rather than missing.
    `strict_keys` mirrors the same parameter on `compute_AC_flat_score`.
    """
    ignore_keys = {"doc_id", "tokens", "event_mentions"}
    if skip_columns:
        ignore_keys = ignore_keys | set(skip_columns)
    # Classification factors whose keyword is the label itself. A factor in
    # both `strict_keys` and `skip_columns` is dropped by `ignore_keys` above
    # before synthesis can fire, so the two flags compose correctly.
    synth_keys = set(strict_keys or [])

    gold_factors, pred_factors = [], []
    # `(doc_id, acc_type, factor_key)` -> gold_keywords for lookup at match time.
    keyword_lookup = {}

    for pred, gold in zip(preds, golds):
        doc_id = gold.get("id", gold.get("doc_id"))
        raw_acc = gold.get("accident_type", ["accident_report"])
        acc_type = raw_acc[0] if isinstance(raw_acc, list) and len(raw_acc) > 0 else raw_acc

        for key, val in gold.items():
            if key in ignore_keys: continue
            if key.endswith("_keywords"): continue
            texts = parse_ac_texts(val, norm=False)
            if not texts:
                continue
            kw_raw = gold.get(f"{key}_keywords")
            kw_groups = _parse_keyword_groups(kw_raw)
            if not kw_groups:
                if key in synth_keys:
                    # Classification label is its own keyword — pred must
                    # contain the gold label as a whole-word substring.
                    kw_groups = [[t] for t in texts]
                else:
                    continue  # no keyword annotation — factor sits outside this metric
            if len(texts) != len(kw_groups):
                continue  # alignment broken — skip this (record, factor) pair
            gold_factors.append((doc_id, acc_type, key, texts))
            keyword_lookup[(doc_id, acc_type, key)] = kw_groups

        for key, val in pred.items():
            if key in ignore_keys: continue
            if key.endswith("_keywords"): continue
            # Only score pred factors whose gold side has a keyword list. A
            # predicted factor with no gold-keyword counterpart can't be
            # judged under this metric. Classification factors (strict_keys)
            # synthesize from the gold label, so they qualify whenever the
            # gold label is present.
            has_kws = bool(_parse_keyword_groups(gold.get(f"{key}_keywords")))
            if not has_kws and key in synth_keys:
                has_kws = bool(parse_ac_texts(gold.get(key), norm=False))
            if not has_kws:
                continue
            texts = parse_ac_texts(val, norm=False)
            if texts:
                pred_factors.append((doc_id, acc_type, key, texts))

    def _count(p_list, g_list):
        matched = 0
        gold_index = {(g[0], g[1], g[2]): g[3] for g in g_list}
        for p in p_list:
            head = (p[0], p[1], p[2])
            if head in gold_index:
                kws = keyword_lookup.get(head)
                if kws is None:
                    continue
                if is_factor_keyword_match(p[3], kws):
                    matched += 1
        return matched

    def _stats(p_list, g_list):
        match_num = _count(p_list, g_list)
        p_num, g_num = len(p_list), len(g_list)
        f1_scores = compute_f1(p_num, g_num, match_num)
        return {
            "pred_num": p_num, "gold_num": g_num, "match_num": match_num,
            "precision": f1_scores[0], "recall": f1_scores[1], "f1": f1_scores[2],
        }

    results = {
        "overall": _stats(pred_factors, gold_factors),
        "per_acc_type": {},
        "per_factor": {},
    }

    unique_acc_types = sorted(set(g[1] for g in gold_factors + pred_factors))
    unique_keys = sorted(set(g[2] for g in gold_factors + pred_factors))

    for atype in unique_acc_types:
        a_gold = [g for g in gold_factors if g[1] == atype]
        a_pred = [p for p in pred_factors if p[1] == atype]
        if a_gold or a_pred:
            results["per_acc_type"][atype] = _stats(a_pred, a_gold)

    for key in unique_keys:
        k_gold = [g for g in gold_factors if g[2] == key]
        k_pred = [p for p in pred_factors if p[2] == key]
        results["per_factor"][key] = {
            "overall": _stats(k_pred, k_gold),
            "per_acc_type": {},
        }
        for atype in unique_acc_types:
            ka_gold = [g for g in k_gold if g[1] == atype]
            ka_pred = [p for p in k_pred if p[1] == atype]
            if ka_gold or ka_pred:
                results["per_factor"][key]["per_acc_type"][atype] = _stats(ka_pred, ka_gold)

    if skip_columns:
        nan_stats = {"pred_num": "NaN", "gold_num": "NaN", "match_num": "NaN",
                     "precision": "NaN", "recall": "NaN", "f1": "NaN"}
        for key in skip_columns:
            results["per_factor"][key] = {
                "overall": dict(nan_stats),
                "per_acc_type": {atype: dict(nan_stats) for atype in unique_acc_types},
            }

    return results

def compute_AC_flat_score(preds, golds, sim_threshold=0.8, skip_columns=None,
                          norm=True, strict_keys=None):
    """`strict_keys`: factor keys whose match must be exact (overlap=1.0)
    rather than `sim_threshold`-fuzzy. Pass classification keys here so
    e.g. "fatality" vs "fatalities" (ratio 0.89) doesn't count as match.
    """
    gold_factors, pred_factors = [], []
    # `accident_report` and `accident_type` (when relevant) come via
    # `skip_columns` from the global mapping — task-level choice. Hardcode
    # only the IE-format structural leftovers + `doc_id` (the canonical id
    # field) so they're never accidentally treated as factors.
    ignore_keys = {"doc_id", "tokens", "event_mentions"}
    if skip_columns:
        ignore_keys = ignore_keys | set(skip_columns)

    for pred, gold in zip(preds, golds):
        doc_id = gold.get("id", gold.get("doc_id"))
        raw_acc = gold.get("accident_type", ["accident_report"])
        acc_type = raw_acc[0] if isinstance(raw_acc, list) and len(raw_acc) > 0 else raw_acc

        for key, val in gold.items():
            if key in ignore_keys: continue
            if key.endswith("_keywords"): continue  # parallel keyword list — handled by keyword scorer
            texts = parse_ac_texts(val, norm=norm)
            if texts:
                gold_factors.append((doc_id, acc_type, key, texts))

        for key, val in pred.items():
            if key in ignore_keys: continue
            if key.endswith("_keywords"): continue
            texts = parse_ac_texts(val, norm=norm)
            if texts:
                pred_factors.append((doc_id, acc_type, key, texts))

    def get_stats(p_list, g_list):
        match_num = count_factor_matches(p_list, g_list,
                                         overlap_pct=sim_threshold,
                                         strict_keys=strict_keys)
        p_num, g_num = len(p_list), len(g_list)
        f1_scores = compute_f1(p_num, g_num, match_num)
        return {
            "pred_num": p_num, "gold_num": g_num, "match_num": match_num,
            "precision": f1_scores[0], "recall": f1_scores[1], "f1": f1_scores[2]
        }

    results = {
        "overall": get_stats(pred_factors, gold_factors),
        "per_acc_type": {},
        "per_factor": {}
    }
    
    unique_acc_types = sorted(list(set(g[1] for g in gold_factors + pred_factors)))
    unique_keys = sorted(list(set(g[2] for g in gold_factors + pred_factors)))

    for atype in unique_acc_types:
        a_gold = [g for g in gold_factors if g[1] == atype]
        a_pred = [p for p in pred_factors if p[1] == atype]
        if a_gold or a_pred:
            results["per_acc_type"][atype] = get_stats(a_pred, a_gold)
            
    for key in unique_keys:
        k_gold = [g for g in gold_factors if g[2] == key]
        k_pred = [p for p in pred_factors if p[2] == key]

        results["per_factor"][key] = {
            "overall": get_stats(k_pred, k_gold),
            "per_acc_type": {}
        }

        for atype in unique_acc_types:
            ka_gold = [g for g in k_gold if g[1] == atype]
            ka_pred = [p for p in k_pred if p[1] == atype]

            if ka_gold or ka_pred:
                results["per_factor"][key]["per_acc_type"][atype] = get_stats(ka_pred, ka_gold)

    # Skipped columns: show in per_factor with NaN values so callers can see
    # they were intentionally excluded from overall/per_acc_type aggregates.
    # Populate per_acc_type for every unique acc_type too, so cell renderers
    # consistently get f1=="NaN" rather than missing-entry → "--".
    if skip_columns:
        nan_stats = {"pred_num": "NaN", "gold_num": "NaN", "match_num": "NaN",
                     "precision": "NaN", "recall": "NaN", "f1": "NaN"}
        for key in skip_columns:
            results["per_factor"][key] = {
                "overall": dict(nan_stats),
                "per_acc_type": {atype: dict(nan_stats) for atype in unique_acc_types},
            }

    return results