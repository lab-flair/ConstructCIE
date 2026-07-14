import os, logging, json, glob, re, shutil
logger = logging.getLogger(__name__)

# Concurrent writes from independent SLURM jobs (e.g. parallel buckets sharing
# the same output_dir) would race on progress JSON files. We guard writes with
# `fcntl.flock` on POSIX; Windows isn't a target for SLURM-style parallelism
# so we silently no-op there and accept last-writer-wins on dev machines.
try:
    import fcntl
    _HAS_FLOCK = True
except ImportError:
    _HAS_FLOCK = False


def _deep_merge(disk, mem):
    """Recursive dict union: `disk` is the on-disk state, `mem` is the
    in-memory state to layer on top. For overlapping leaves, `mem` wins.
    For overlapping dict subtrees, recurse. Returns a fresh dict.

    This is the merge step that lets two SLURM jobs each write *their own*
    k_shot entries without trampling each other — they only conflict on the
    exact same (k_shot, split) field, in which case the second writer's
    value is kept.
    """
    if not isinstance(disk, dict) or not isinstance(mem, dict):
        return mem
    out = dict(disk)
    for k, v in mem.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _sort_kshot_keys(obj):
    """Reorder a dict so `k_shot_<N>` keys appear first in numeric ascending
    order, with any other keys after them in lexical order. Only top-level
    reordering — nested dicts are returned as-is.

    Plain alphabetical sort would put `k_shot_10` before `k_shot_2` because
    of lexical comparison; this helper sorts by the integer N instead so the
    on-disk file reads naturally low→high.
    """
    if not isinstance(obj, dict):
        return obj
    def _sk(k):
        if isinstance(k, str) and k.startswith("k_shot_"):
            try:
                return (0, int(k[len("k_shot_"):]))
            except ValueError:
                pass
        return (1, str(k))
    return {k: obj[k] for k in sorted(obj, key=_sk)}

class ProgressManager:
    """Tracks training and inference progress in two separate JSON files.

    training_progress.json  — maps split_name → absolute checkpoint dir.
                              Written after each split finishes training.
    prediction_progress.json — maps k_shot_N → {split_name: pred_file,
                              total_scores: score_file}.
                              Written after each split's inference and after
                              the final aggregate score.
    """

    # PRED_PATTERN_DEFAULT  = "pred_{model}_{dataset}_{task}_{split}_k{shot}.json"
    # SCORE_PATTERN_DEFAULT = "scores_{model}_{dataset}_{task}_k{shot}.json"

    def __init__(self, output_dir, model_name, dataset, task, pred_pattern, score_pattern,
                 readonly=False, train_dir=None):
        self.output_dir    = output_dir
        self.model_name    = model_name
        self.dataset       = dataset
        self.task          = task
        self.pred_pattern  = pred_pattern
        self.score_pattern = score_pattern

        self.readonly = readonly
        # Training-progress lives alongside the trainer-state checkpoints
        # (`<trained_model_dir>/<task>/<dataset>/<model>/`), not in the
        # prediction-results folder. Prediction/metadata files stay in
        # `output_dir`. When `train_dir` isn't provided we fall back to
        # `output_dir` for back-compat.
        self.train_dir         = train_dir or output_dir
        self.train_filepath    = os.path.join(self.train_dir, "training_progress.json")
        self.pred_filepath     = os.path.join(output_dir, "prediction_progress.json")
        self.metadata_filepath = os.path.join(output_dir, "prediction_metadata.json")

        # Backward compat: migrate old progress.json → prediction_progress.json
        if not self.readonly:
            _old = os.path.join(output_dir, "progress.json")
            if os.path.exists(_old) and not os.path.exists(self.pred_filepath):
                shutil.copy2(_old, self.pred_filepath)

        self.train_state = self._load_json(self.train_filepath) or {}
        self.pred_state = self._load_or_reconstruct()
        self.metadata_state = self._load_json(self.metadata_filepath) or {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_json(self, filepath):
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {os.path.basename(filepath)}: {e}")
        return None

    def _save(self, state, filepath):
        """Read-modify-write the JSON at `filepath` under an exclusive
        cross-process lock, with atomic rename to avoid torn reads.

        Returns the merged state that was actually written. Callers MUST
        adopt this return value as their in-memory cache so subsequent
        operations see entries other processes added under the lock.

        On non-POSIX platforms there's no flock — the call degrades to a
        plain atomic write. That's fine for dev machines (no parallelism)
        but means SLURM-style parallel jobs MUST run on POSIX.
        """
        if self.readonly:
            return state
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        lock_path = filepath + ".lock"
        tmp_path = f"{filepath}.tmp.{os.getpid()}"

        with open(lock_path, "a+") as lock:
            if _HAS_FLOCK:
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                disk = self._load_json(filepath) or {}
                merged = _sort_kshot_keys(_deep_merge(disk, state))
                with open(tmp_path, "w") as f:
                    json.dump(merged, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, filepath)
                return merged
            finally:
                if _HAS_FLOCK:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _load_or_reconstruct(self):
        data = self._load_json(self.pred_filepath)
        if data is not None:
            return data
        return self._reconstruct_from_files()

    def _reconstruct_from_files(self):
        state = {}

        glob_fmt = self.pred_pattern.format(
            model=self.model_name, dataset=self.dataset,
            task=self.task.lower(), split="*", shot="*",
        )
        pred_files = glob.glob(os.path.join(self.output_dir, glob_fmt))

        glob_fmt_2 = self.score_pattern.format(
            model=self.model_name, dataset=self.dataset,
            task=self.task.lower(), shot="*",
        )
        score_files = glob.glob(os.path.join(self.output_dir, glob_fmt_2))

        logger.info(f"Reconstructing inference progress from "
                    f"{len(pred_files) + len(score_files)} files.")

        pred_temp = self.pred_pattern.format(
            model=self.model_name, dataset=self.dataset,
            task=self.task.lower(), split="___SPLIT___", shot="___SHOT___",
        )
        regex_str = re.escape(pred_temp).replace(
            "___SPLIT___", r"(?P<split>.+?)"
        ).replace("___SHOT___", r"(?P<shot>\d+)")
        pattern_re = re.compile(regex_str)

        for pfile in pred_files:
            fname = os.path.basename(pfile)
            m = pattern_re.match(fname)
            if m:
                try:
                    k_key = f"k_shot_{int(m.group('shot'))}"
                    if k_key not in state:
                        state[k_key] = {}
                    state[k_key][m.group("split")] = fname
                except ValueError:
                    continue

        score_temp = self.score_pattern.format(
            model=self.model_name, dataset=self.dataset,
            task=self.task.lower(), shot="___SHOT___",
        )
        regex_score = re.compile(
            re.escape(score_temp).replace("___SHOT___", r"(?P<shot>\d+)")
        )
        for sfile in score_files:
            fname = os.path.basename(sfile)
            m = regex_score.match(fname)
            if m:
                try:
                    k_key = f"k_shot_{int(m.group('shot'))}"
                    if k_key not in state:
                        state[k_key] = {}
                    state[k_key]["total_scores"] = fname
                except ValueError:
                    continue

        if not self.readonly:
            state = self._save(state, self.pred_filepath)
        return state

    # ------------------------------------------------------------------ #
    # Training progress                                                    #
    # ------------------------------------------------------------------ #

    def complete_training(self, split_name, ckpt_dir):
        """Record that *split_name* has a trained checkpoint at *ckpt_dir*."""
        self.train_state[split_name] = ckpt_dir
        self.train_state = self._save(self.train_state, self.train_filepath)
        logger.info(f"Training progress: {split_name} → {ckpt_dir}")

    def is_trained(self, split_name):
        """True if a valid best_model.state exists for *split_name*."""
        ckpt_dir = self.train_state.get(split_name)
        return bool(ckpt_dir and os.path.exists(os.path.join(ckpt_dir, "best_model.state")))

    def all_splits_trained(self, split_names):
        """True if every split in *split_names* has a valid checkpoint."""
        return bool(split_names) and all(self.is_trained(s) for s in split_names)

    # ------------------------------------------------------------------ #
    # Inference progress                                                   #
    # ------------------------------------------------------------------ #

    def get_result(self, shot_size, split_name):
        k_key = f"k_shot_{shot_size}"
        if k_key in self.pred_state:
            return (
                self.pred_state[k_key].get(split_name),
                self.pred_state[k_key].get("total_scores"),
            )
        return None, None

    def complete_split(self, shot_size, split_name, predfile):
        k_key = f"k_shot_{shot_size}"
        if k_key not in self.pred_state:
            self.pred_state[k_key] = {}
        self.pred_state[k_key][split_name] = predfile
        self.pred_state = self._save(self.pred_state, self.pred_filepath)
        logger.info(f"Inference progress: {split_name} (shot {shot_size}) → {predfile}")

    def complete_shot(self, shot_size, scorefile):
        k_key = f"k_shot_{shot_size}"
        if k_key not in self.pred_state:
            self.pred_state[k_key] = {}
        self.pred_state[k_key]["total_scores"] = scorefile
        self.pred_state = self._save(self.pred_state, self.pred_filepath)
        logger.info(f"Inference progress: shot {shot_size} complete → {scorefile}")

    def set_skip_columns(self, shot_size, split_name, skip_columns):
        """Persist the per-split list of column keys to skip.

        Stored as: metadata_state["skip_columns"][k_shot_<size>][split_name] = [...]
        """
        sv = self.metadata_state.setdefault("skip_columns", {})
        sv.setdefault(f"k_shot_{shot_size}", {})[split_name] = list(skip_columns)
        self.metadata_state = self._save(self.metadata_state, self.metadata_filepath)

    def get_skip_columns(self, shot_size, split_name):
        """Return the stored list of skipped columns for this shot/split, or None."""
        return (self.metadata_state.get("skip_columns", {})
                                   .get(f"k_shot_{shot_size}", {})
                                   .get(split_name))

    def clear_pred_state(self):
        """Reset prediction progress and delete the progress file."""
        self.pred_state = {}
        if os.path.exists(self.pred_filepath):
            os.remove(self.pred_filepath)
            logger.info(f"Cleared prediction progress: {self.pred_filepath}")

    def clear_train_state(self):
        """Reset training progress and delete the progress file."""
        self.train_state = {}
        if os.path.exists(self.train_filepath):
            os.remove(self.train_filepath)
            logger.info(f"Cleared training progress: {self.train_filepath}")

    def is_done(self, shot_size):
        """True if the aggregate score file for *shot_size* exists on disk."""
        k_key = f"k_shot_{shot_size}"
        score_fname = self.pred_state.get(k_key, {}).get("total_scores")
        return bool(score_fname and os.path.exists(os.path.join(self.output_dir, score_fname)))

    def all_splits_done(self, shot_size, split_names):
        """True if every named split has a pred file recorded and present on disk."""
        k_key = f"k_shot_{shot_size}"
        split_state = self.pred_state.get(k_key, {})
        return bool(split_names) and all(
            sn in split_state
            and os.path.exists(os.path.join(self.output_dir, split_state[sn]))
            for sn in split_names
        )

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def load_split(self, splitpath):
        from TextEE.utils import load_all_predictions
        return load_all_predictions(os.path.join(self.output_dir, splitpath))

    def aggregate_skip_columns(self, shot_size, split_names, skip_threshold):
        """Combine per-split skip vectors into a single list of columns whose
        presence fraction across `split_names` is >= `skip_threshold`.
        Returns [] when threshold <= 0 or no splits provided.
        """
        if skip_threshold <= 0 or not split_names:
            return []
        counts = {}
        for sn in split_names:
            for col in (self.get_skip_columns(shot_size, sn) or []):
                counts[col] = counts.get(col, 0) + 1
        n = len(split_names)
        return [c for c, k in counts.items() if (k / n) >= skip_threshold]

    def print_aggregate_scores(self, shot_size, override_skip=None, per_split_skip=None):
        """Load the stored score JSON for `shot_size` and print each split's
        scores + the aggregate, including the skip columns associated with
        each. No recomputation, no save.

        `override_skip` (list)        — replaces the AGGREGATE skip vector.
        `per_split_skip` (dict)       — split_name → list of cols, replaces
                                        each split's skip vector individually.
                                        Falls back to `override_skip` per split
                                        when not provided.
        """
        fname = self.pred_state.get(f"k_shot_{shot_size}", {}).get("total_scores")
        if not fname:
            return
        path = os.path.join(self.output_dir, fname)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load score file {fname}: {e}")
            return

        from TextEE.scorer import print_scores

        def _mask_skipped(scores_dict, skip_cols):
            """Walk the per-factor section and replace skipped columns' P/R/F1
            with the string "NaN" so print_scores renders them as SKIPPED."""
            if not skip_cols:
                return scores_dict
            skip_set = {c.lower() for c in skip_cols}
            per = scores_dict.get("per_factor") or scores_dict.get("factors") or {}
            for k, v in per.items():
                if isinstance(v, dict) and k.lower() in skip_set:
                    for metric in ("p", "r", "f1", "precision", "recall"):
                        if metric in v:
                            v[metric] = "NaN"
            return scores_dict

        for sn, split_scores in (data.get("splits") or {}).items():
            if per_split_skip is not None and sn in per_split_skip:
                split_skip = list(per_split_skip[sn])
            elif override_skip is not None:
                split_skip = list(override_skip)
            else:
                split_skip = list(self.get_skip_columns(shot_size, sn) or [])
            print(f"    skip[{sn}]: {split_skip}")
            print_scores(_mask_skipped(split_scores, split_skip), split=sn, shot=shot_size)

        agg = data.get("aggregate") or {}
        if agg:
            agg_skip = list(override_skip) if override_skip is not None else data.get("skipped_columns", [])
            print(f"    skip: {agg_skip}")
            print_scores(_mask_skipped(agg, agg_skip), split=None, shot=shot_size)

    def load_shot(self, shot_size, split_names=None):
        k_key = f"k_shot_{shot_size}"
        if k_key not in self.pred_state:
            logger.warning(f"No inference state found for shot {shot_size}")
            return []
        try:
            if split_names is None:
                split_paths = [v for k, v in self.pred_state[k_key].items()
                               if k.startswith("split")]
            else:
                split_paths = []
                for name in split_names:
                    if name in self.pred_state[k_key]:
                        split_paths.append(self.pred_state[k_key][name])
                    else:
                        logger.warning(f"Split {name} missing from inference state "
                                       f"for shot {shot_size}")
            logger.info(f"Loading {len(split_paths)} files for shot {shot_size}…")
            return [item for sp in split_paths for item in self.load_split(sp)]
        except Exception as e:
            logger.error(f"Failed to load shot {shot_size}: {e}")
            return []
