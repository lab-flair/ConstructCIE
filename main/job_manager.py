import os
import sys
import re
import json
import csv
import glob
import logging
import argparse
import subprocess
import itertools
from datetime import datetime
from main.progress_manager import ProgressManager
from main.job_runner import build_llm_cmd_list
from typing import Dict, Tuple

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:{num_gpus}
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}

# --- SETUP ENVIRONMENT ---
if command -v module &> /dev/null; then
    module load CUDA/11.8.0
fi
if [ -n "$CONDA_EXE" ] && [ -x "$CONDA_EXE" ]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif command -v module &> /dev/null; then
    module load "${{CONDA_MODULE:-Miniconda3/23.10.0-1}}"
    eval "$(conda shell.bash hook)"
fi
conda activate textee3

cd {work_dir}

echo "Starting run on $(hostname) with GPUs: $CUDA_VISIBLE_DEVICES"
echo "Job: {job_name}"
echo "Command: {cmd_str}"

{cmd_str}
"""

# --- New-pipeline job creation (used by run_arg.py) ---
# run_arg.py resolves arguments (paths, gpu/batch grouping, durations, command
# lists) and hands fully-formed job_spec dicts here; this module owns turning
# those specs into actual files on disk. Distinct from SLURM_TEMPLATE/JobManager
# above, which belong to the older standalone log-parsing/report CLI.
JOB_SLURM_TEMPLATE = """#!/bin/bash

# --- SLURM RESOURCE REQUESTS ---
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_type}:{num_gpus}
#SBATCH --time={duration}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}

# --- SETUP ENVIRONMENT ---
# Conda lives at a different path on every host (e.g. /sw/eb/sw/Miniconda3
# vs ~/miniconda3), so activate dynamically via $CONDA_EXE (exported once
# `conda init` has run in .bashrc) rather than hardcoding one. Falls back to
# `module load` only if neither is already available. Same logic as
# scripts/reset_env.sh and scripts/download_models_job.sh.
if command -v module &> /dev/null; then
    module load CUDA/11.8.0
fi
if [ -n "$CONDA_EXE" ] && [ -x "$CONDA_EXE" ]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif command -v module &> /dev/null; then
    module load "${{CONDA_MODULE:-Miniconda3/23.10.0-1}}"
    eval "$(conda shell.bash hook)"
fi
conda activate textee3

# Navigate to directory
cd {work_dir}
{debug_env}
# --- RUN SCRIPT ---
echo "Starting run on $(hostname) with GPUs: $CUDA_VISIBLE_DEVICES"
echo "Job: {job_name}"
echo "Command: {cmd_str}"

{cmd_str}
"""


def make_log_filename(job_name, suffix):
    """Build a job log filename: <job_name>_<suffix>.txt. Shared by the
    generate path (suffix is SLURM's %j placeholder, substituted at submit
    time) and run.py's `-a run` path (suffix is a run timestamp), so the two
    naming schemes can't drift apart."""
    return f"{job_name}_{suffix}.txt"


def create_sbatch_script(job_spec, dry_run=False):
    """Render JOB_SLURM_TEMPLATE from a resolved job_spec and write it to
    job_spec["path"] (or just log a preview when dry_run). job_spec keys:
    partition, gpu_type, num_gpus, duration, job_name, log_path, work_dir,
    cmd_str, debug_env, path. Returns job_spec["path"]."""
    content = JOB_SLURM_TEMPLATE.format(
        partition=job_spec["partition"], gpu_type=job_spec["gpu_type"],
        num_gpus=job_spec["num_gpus"], duration=job_spec["duration"],
        job_name=job_spec["job_name"], log_path=job_spec["log_path"],
        work_dir=job_spec["work_dir"], cmd_str=job_spec["cmd_str"],
        debug_env=job_spec.get("debug_env", ""),
    )
    path = job_spec["path"]
    if dry_run:
        logger.info("\n  🔍 [DRY] Preview for: %s", path)
        logger.info("%s", content.strip())
    else:
        with open(path, "w") as f:
            f.write(content)
        logger.info("  ✅ [GEN] Created: %s", path)
    return path


def create_grouped_sbatch_parts(slurm_dir, bucket_job_name, log_path, work_dir,
                                 target_script, sub_jobs, gpu_partition_fn, duration,
                                 debug_env="", dry_run=False):
    """Turn a bucket's resolved per-(gpu, gpu_type, batch) sub-jobs into sbatch
    part file(s), building each sub-job's command via the shared
    `build_llm_cmd_list` (also used by job_runner's "run" path) — so a
    SLURM-generated command and a directly-executed one never drift apart.

    Sub-jobs sharing the same (task, model, gpu_list, gpu_type) — i.e. the
    same SLURM resource request for the same job — run as sequential commands
    inside ONE sbatch job instead of one job each: batch size alone doesn't
    change what a job needs to request, and every part already requests the
    bucket's full `duration` regardless of how many sub-jobs it holds, so one
    job covering several sequential commands fits the same time budget a
    chain of separate jobs would have used. Task and model are part of the
    grouping key too — even if two sub-jobs happened to need the same GPUs,
    one job must never mix commands from different tasks/models. Only a
    genuinely different key — rare, only when gpu_requirements/gpu_mappings
    vary by k — still needs its own part file (a running job can't change
    its resource request mid-run); the caller chains those via afterok
    (see `create_bucket_wrapper`).

    sub_jobs: list of dicts with keys gpu_list, gpu_type, eval_bs, shots, mins,
    task, d_path, d_name, model_path, out_path, max_out, model_max,
    sim_threshold, skip_threshold, key_map, col_defs, base_cmd_tail, skip_cols
    — every resolved value `build_llm_cmd_list` needs; nothing pre-built.
    Returns the list of part file basenames, in order.
    """
    resource_order = []
    resource_groups = {}
    for sj in sub_jobs:
        key = (sj["task"], sj["model_path"], tuple(sj["gpu_list"]), sj["gpu_type"])
        if key not in resource_groups:
            resource_groups[key] = []
            resource_order.append(key)
        resource_groups[key].append(sj)

    part_files = []
    multi_part = len(resource_order) > 1
    for i, (task, model_path, gpu_tuple, gpu_type) in enumerate(resource_order, 1):
        gpu_list = list(gpu_tuple)
        group = resource_groups[(task, model_path, gpu_tuple, gpu_type)]

        sub_cmd_blocks = []
        for sj in group:
            cmd_list = build_llm_cmd_list(
                target_script, task=sj["task"], d_path=sj["d_path"], d_name=sj["d_name"],
                model_path=sj["model_path"], out_path=sj["out_path"],
                gpu_list=gpu_list, eval_bs=sj["eval_bs"],
                max_out=sj["max_out"], model_max=sj["model_max"],
                sim_threshold=sj["sim_threshold"], skip_threshold=sj["skip_threshold"],
                key_map=sj["key_map"], col_defs=sj["col_defs"],
                shots=sj["shots"], mins=sj["mins"],
                base_cmd_tail=sj.get("base_cmd_tail"), skip_cols=sj.get("skip_cols"),
            )
            cmd_str = " ".join(str(c) for c in cmd_list)
            # Single-quote the label, and join shots with commas rather than
            # using the raw list's str() (which renders as ['0', '5'] —
            # those embedded single quotes would close the label's own
            # single-quoted bash string early). The label itself also can't
            # use double quotes: the whole combined block below gets embedded
            # in the template's own `echo "Command: {cmd_str}"` wrapper, and
            # a double-quoted label here would close THAT outer quote early.
            label = f"batch={sj['eval_bs']}, k={','.join(sj['shots'])}"
            sub_cmd_blocks.append(f"echo '--- {label} ---'\n{cmd_str}")
        combined_cmd_str = "\n\n".join(sub_cmd_blocks)

        part_name = f"{bucket_job_name}_part{i}" if multi_part else f"{bucket_job_name}_part1"
        part_filename = os.path.join(slurm_dir, f"{part_name}.sbatch")

        # This is the only place that knows exactly what ended up in THIS
        # part file (run.py's own grouping can't tell — several of its
        # (gpu, batch, gpu_type) groups may collapse into the same part
        # here), so log the per-job batch/k-shot breakdown right here.
        job_breakdown = "; ".join(
            f"batch={sj['eval_bs']} k={sj['shots']}" for sj in group)
        logger.info("    📄 %s [gpu=%s %s] → %s",
                    part_name, ",".join(str(g) for g in gpu_list), gpu_type, job_breakdown)

        create_sbatch_script({
            "partition": gpu_partition_fn(gpu_type),
            "num_gpus": len(gpu_list), "gpu_type": gpu_type,
            "job_name": part_name, "log_path": log_path,
            "work_dir": work_dir, "cmd_str": combined_cmd_str,
            "debug_env": debug_env,
            "duration": duration,
            "path": part_filename,
        }, dry_run=dry_run)
        part_files.append(os.path.basename(part_filename))

    return part_files


def create_bucket_wrapper(wrapper_path, part_files, log_dir, bucket_job_name, dry_run=False):
    """Write the .sh wrapper that chains a bucket's sbatch parts via afterok
    (parts inside one bucket share a log file and run serially on the GPU;
    separate buckets are independent and chained by `create_master_submit`)."""
    wrapper_lines = ["#!/bin/bash", "set -e", 'cd "$(dirname "$0")"']
    wrapper_lines.append(f'JID0=$(sbatch --parsable "{part_files[0]}")')
    wrapper_lines.append('JID=$JID0')
    if len(part_files) > 1:
        wrapper_lines.append(f'LOG="{log_dir}/{bucket_job_name}_${{JID0}}.txt"')
        wrapper_lines.append(f'echo "Submitted {part_files[0]} as $JID0 (log: $LOG)"')
        for pf in part_files[1:]:
            wrapper_lines.append(
                f'JID=$(sbatch --parsable --dependency=afterok:$JID '
                f'--output="$LOG" --open-mode=append "{pf}")')
            wrapper_lines.append(
                f'echo "Submitted {pf} as $JID (depends on previous, appending to $LOG)"')
    else:
        wrapper_lines.append(f'echo "Submitted {part_files[0]} as $JID"')
    content = "\n".join(wrapper_lines) + "\n"

    if dry_run:
        logger.info("\n  🔍 [DRY] Preview for: %s", wrapper_path)
        logger.info("%s", content.strip())
    else:
        with open(wrapper_path, "w") as f:
            f.write(content)
        logger.info("  ✅ [GEN] Created: %s", wrapper_path)
    return wrapper_path


def create_master_submit(master_path, bucket_wrappers, dry_run=False):
    """Write the top-level submit.sh that runs each independent bucket
    wrapper in sequence (buckets do NOT chain via afterok with each other —
    only parts within the same bucket do, via `create_bucket_wrapper`)."""
    lines = ["#!/bin/bash", "set -e", 'cd "$(dirname "$0")"',
             f'echo "Submitting {len(bucket_wrappers)} bucket(s) independently..."']
    for w in bucket_wrappers:
        lines.append(f'echo "--- {w} ---"')
        lines.append(f'bash "{w}"')
    content = "\n".join(lines) + "\n"

    if dry_run:
        logger.info("\n  🔍 [DRY] Preview for: %s", master_path)
        logger.info("%s", content.strip())
    else:
        with open(master_path, "w") as f:
            f.write(content)
        logger.info("  ✅ [GEN] Created: %s (master)", master_path)
    return master_path


def filter_filenames(filenames, pattern: str, filters: Dict, filter_only=True):
    """
    Filters filenames using a template pattern and extracts metadata.
    
    Args:
        filenames: List of strings (file paths).
        pattern: Format string, e.g., "result_{split}_{shot}.json".
        filters: Dict of constraints, e.g., {'split': ['train', 'test'], 'shot': 5}.
        filter_only: If True, returns list of filenames. 
                     If False, returns list of tuples: (filename, dict_of_parsed_data).
    """

    # 1. Helper: Converts filter values into Regex Strings
    def reg_formatter(val):
        if isinstance(val, list):
            # Lists become non-capturing OR groups: (?:a|b|c)
            # We map str() and re.escape() to handle numbers and special chars safely
            safe_vals = [re.escape(str(v)) for v in val]
            return r'(?:%s)' % '|'.join(safe_vals)
        elif isinstance(val, (float, int, str)):
            # Single values match exactly
            return re.escape(str(val))
        return r'.+?' # Fallback

    # 2. Smart Dictionary: Handles both defined filters and undefined wildcards
    class RegexBuilder(dict):
        def __missing__(self, key):
            # If pattern has {key} but filters doesn't, allow ANY content
            return fr'(?P<{key}>.+?)'

    # 3. Build the Regex Pattern
    try:
        # Wrap user filters in named groups: split -> (?P<split>val_regex)
        regex_subs = RegexBuilder()
        for k, v in filters.items():
            regex_subs[k] = fr'(?P<{k}>{reg_formatter(v)})'
            
        # Inject regex parts into the template
        # e.g., "file_{split}.json" -> "file_(?P<split>train|test).json"
        regex_str = pattern.format_map(regex_subs)
        
        # Note: We anchor with ^ and $ to ensure exact matches
        pattern_re = re.compile(f"^{regex_str}$")
        
    except Exception as e:
        print(f"Error constructing regex: {e}")
        return []

    # 4. Filter and Parse
    results = []
    for fname in filenames:
        base_name = os.path.basename(fname)
        match = pattern_re.match(base_name)
        
        if match:
            if filter_only:
                results.append(fname)
            else:
                # Return the filename AND the extracted dictionary
                results.append((fname, match.groupdict()))

    return results


class JobManager:
    """
    Central management class for TextEE experiments.
    Handles: Log parsing, Result aggregation, Job dispatch (Slurm/Local), and Progress tracking.
    """

    def __init__(self, base_dir=None, config_file="global_data/global_mapping.json"):
        self.base_dir = base_dir if base_dir else os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(self.base_dir, config_file)
        self.config = self._load_config()
        
        # Paths
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.slurm_dir = os.path.join(self.base_dir, "jobs")
        self.target_script = os.path.join(self.base_dir, "TextEE", "evaluate_llm.py")

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {self.config_path}")
            return {}

    def _smart_path(self, path):
        if path.startswith("/"): return path
        return os.path.join(self.base_dir, path)

    # =========================================================================
    # MODULE 1: LOG PARSING (Execution Times)
    # =========================================================================
    def parse_logs(self, log_filenames=None, display=False):
        """Parses Slurm logs to calculate execution durations per split."""
        if not log_filenames:
            # Default to all .txt files in log dir if none specified
            log_filenames = [f for f in os.listdir(self.log_dir) if f.endswith('.txt')]

        start_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] - .*? - Processing (split\d+) \(K=(\d+)\)")
        end_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] - .*? - Computing scores for (split\d+)")
        oom_pattern = re.compile(r"torch\.OutOfMemoryError")
        cancelled_pattern = re.compile(r"CANCELLED AT")
        time_fmt = "%Y-%m-%d %H:%M:%S"

        all_results = []

        # print(f"\n{'K-Shot':<8} | {'Split':<10} | {'Duration':<20} | {'Minutes':<10} | {'Log File'}")
        # print("-" * 80)

        for log_file in log_filenames:
            full_path = os.path.join(self.log_dir, log_file)
            current_job = None
            
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # Check Start
                        start_match = start_pattern.search(line)
                        if start_match:
                            current_job = {
                                "start_time": datetime.strptime(start_match.group(1), time_fmt),
                                "split": start_match.group(2),
                                "k_shot": int(start_match.group(3))
                            }
                            continue

                        # Check Errors
                        if current_job and (oom_pattern.search(line) or cancelled_pattern.search(line)):
                            current_job = None
                            continue

                        # Check End
                        end_match = end_pattern.search(line)
                        if end_match and current_job:
                            if end_match.group(2) == current_job['split']:
                                end_time = datetime.strptime(end_match.group(1), time_fmt)
                                duration = end_time - current_job['start_time']
                                seconds = duration.total_seconds()
                                
                                # print(f"{current_job['k_shot']:<8} | {current_job['split']:<10} | {str(duration):<20} | {round(seconds/60, 2):<10} | {log_file}")
                                all_results.append(seconds)
                                current_job = None
            except FileNotFoundError:
                logger.warning(f"File {log_file} not found.")

        # if all_results:
        #     total_hours = sum(all_results) / 3600
        #     print("-" * 80)
        #     print(f"Total Successful Runtime: {total_hours:.2f} hours")
        return all_results
    
    def parse_filename(self, filename):
        """
        Parses filename in format: Task_Dataset_Model_JobId.txt
        Returns a dict of metadata or None if invalid.
        """
        if not filename.endswith(".txt"):
            return None
        
        # Remove extension
        stem = filename[:-4]
        
        parts = stem.split('_')
        
        # We need at least 4 parts: Task, Dataset, Model, JobID
        if len(parts) < 4:
            return None
            
        return {
            "task": parts[0],                 # e.g., EAE
            "dataset": parts[1],              # e.g., phee
            # Model is everything between dataset and the last part (Job ID)
            "model": "_".join(parts[2:-1]),   # e.g., Llama-3.2-11B
            "job_id": parts[-1],              # e.g., 17338821
            "filename": filename
        }

    
    
    def generate_report(self, input_source, output_file=None, filter_task=None, filter_dataset=None, filter_model=None):
        """
        Args:
            input_source: Directory path (str) or list of file paths.
            filter_task: List of allowed tasks (e.g., ['EAE', 'NER'])
            filter_dataset: List of allowed datasets
            filter_model: List of allowed models
        """
        import pandas as pd
        # 1. Normalize Input to List of Files (same as before)
        file_paths = []
        if isinstance(input_source, list):
            file_paths = input_source
        elif isinstance(input_source, str):
            if os.path.isdir(input_source):
                file_paths = [os.path.join(input_source, f) for f in os.listdir(input_source)]
            elif os.path.isfile(input_source):
                file_paths = [input_source]
            else:
                print(f"Error: Input '{input_source}' not found.")
                return

        results = []
        print(f"--- Generating Report ---")
        print(f"Filters -> Task: {filter_task} | Dataset: {filter_dataset} | Model: {filter_model}")

        for full_path in file_paths:
            fname = os.path.basename(full_path)
            meta = self.parse_filename(fname)
            
            if not meta:
                continue 

            # --- UPDATED FILTER LOGIC ---
            # Check if filter exists AND if the current file's value is NOT in that filter list
            
            if filter_task and meta['task'] not in filter_task:
                continue
                
            if filter_dataset and meta['dataset'] not in filter_dataset:
                continue
                
            if filter_model and meta['model'] not in filter_model:
                continue

            # 3. Read Content (same as before)
            try:
                results_df = pd.DataFrame(self.parse_logs(full_path))
                metadata = {
                        "Task": meta['task'],
                        "Dataset": meta['dataset'],
                        "Model": meta['model'],
                        "Job ID": meta['job_id'],
                    }
                results_df = results_df.assign(**metadata)
            except Exception as e:
                print(f"Error reading {fname}: {e}")

        # 4. Output (same as before)
        if not results:
            print("No matching results found.")
            return

        df = pd.DataFrame(results)
        df = df.sort_values(by=['Dataset', 'Model'])
        
        print("\nSummary:")
        print(df.to_markdown(index=False)) 
        
        # Save to CSV
        if output_file:
            csv_name = "experiment_report.csv"
            df.to_csv(csv_name, index=False)
            print(f"\nSaved report to {csv_name}")

    # =========================================================================
    # MODULE 2: RESULT AGGREGATION (CSV Compilation)
    # =========================================================================
    def compile_results(self, output_file, filter_models=None, filter_datasets=None, filter_tasks=None, include_splits=False):
        """Combines JSON result files into a single CSV."""
        root_dir = self.config.get("output_dir", ".")
        
        # Resolve Filters
        target_models = filter_models if filter_models else [m.get("path") for m in self.config.get("models", {}).values()]
        target_tasks = filter_tasks if filter_tasks else self.config.get("task_mapping", {}).keys()
        
        # Resolve Datasets
        config_datasets = self.config.get("dataset_alts", {})
        if filter_datasets:
            target_dataset_keys = [d for d in filter_datasets if d in config_datasets]
        else:
            target_dataset_keys = list(config_datasets.keys())

        logger.info(f"Compiling Results from: {root_dir}")
        all_rows = []

        for task in target_tasks:
            valid_datasets = self.config.get("task_mapping", {}).get(task, [])
            for ds_key in target_dataset_keys:
                if ds_key not in valid_datasets: continue
                
                real_dataset_name = config_datasets.get(ds_key)
                search_pattern = os.path.join(root_dir, "**", f"scores_*{real_dataset_name}*.json")
                files = glob.glob(search_pattern, recursive=True)

                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
                        continue

                    # Validation
                    if data.get("task") != task: continue
                    if data.get("dataset") != real_dataset_name: continue
                    json_model = data.get("model", "Unknown")
                    if json_model not in target_models: continue

                    k_shot = data.get("k_shot", "Unknown")

                    # Helper to extract rows
                    def extract_f1(metric_dict, split_label):
                        for metric, scores in metric_dict.items():
                            if isinstance(scores, dict) and "f1" in scores:
                                all_rows.append({
                                    "Model": json_model, "K_Shot": k_shot, "Dataset": ds_key,
                                    "Task": task, "Split": split_label, "Metric": metric,
                                    "F1_Score": scores["f1"]
                                })

                    if "aggregate" in data: extract_f1(data["aggregate"], "aggregate")
                    if include_splits and "splits" in data:
                        for sk, sd in data["splits"].items(): extract_f1(sd, sk)

        # Write CSV
        if all_rows:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            headers = ["Model", "K_Shot", "Dataset", "Task", "Split", "Metric", "F1_Score"]
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_rows)
            logger.info(f"Success! Processed {len(all_rows)} rows into '{output_file}'.")
        else:
            logger.warning("No matching data found.")

    # =========================================================================
    # MODULE 3: JOB DISPATCH (Slurm / Local)
    # =========================================================================
    def dispatch_jobs(self, action, tasks=None, datasets=None, models=None, few_shot_override=None, restart=False, dry_run=False):
        """Generates Slurm scripts or runs experiments locally."""
        # 1. Setup Filters
        all_tasks = list(self.config.get("task_mapping", {}).keys())
        all_datasets = list(self.config.get("dataset_alts", {}).keys())
        
        # Determine Models list
        if "gpu_mapping" in self.config: all_models = list(self.config["gpu_mapping"].keys())
        elif "models_alts" in self.config: all_models = list(self.config["models_alts"].keys())
        else: all_models = list(self.config.get("models", {}).keys())

        tasks = tasks if tasks else all_tasks
        datasets = datasets if datasets else all_datasets
        models = models if models else all_models
        
        shots_list = [str(x) for x in few_shot_override] if few_shot_override else [str(x) for x in self.config.get("few_shot_sizes", [])]

        if not dry_run and action == "generate":
            os.makedirs(self.slurm_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)

        combinations = list(itertools.product(tasks, datasets, models))
        count = 0

        for task, dataset_key, model_key in combinations:
            # Compatibility Check
            allowed = self.config.get("task_mapping", {}).get(task)
            if allowed is not None and dataset_key not in allowed: continue

            # Get Config Details
            model_cfg = self._get_job_config(model_key, dataset_key, task)
            
            cmd_list = [
                "python", self.target_script,
                "-t", task,
                "-d", model_cfg['dataset_path'],
                "-D", model_cfg['dataset_name'],
                "-m", model_cfg['model_path'],
                "-o", model_cfg['output_path'],
                "-g", model_cfg['gpu_str'],
                "-b", str(model_cfg['batch_size']),
                "-M", str(model_cfg['max_output']),
                "-l", str(model_cfg['model_max']),
                "-F"
            ] + shots_list

            if restart: cmd_list.append("--restart")

            cmd_str = " ".join([str(c) for c in cmd_list])
            safe_model = model_key.replace("/", "_").replace(" ", "")
            job_name = f"{task}_{dataset_key}_{safe_model}"

            if action == "generate":
                self._create_slurm_script(job_name, model_cfg['num_gpus'], cmd_str, dry_run)
            elif action == "run":
                self._run_local_command(job_name, cmd_list, model_cfg['gpu_str'], dry_run)
            
            count += 1
        
        print(f"\nCompleted processing {count} jobs.")

    def _get_job_config(self, model_key, dataset_key, task):
        """Helper to extract paths and params from config."""
        # Model
        base_model_dir = self._smart_path(self.config.get("model_dir", ""))
        if "models" in self.config and model_key in self.config["models"]:
            m_cfg = self.config["models"][model_key]
            model_subpath = m_cfg["path"]
            gpu_list = m_cfg.get("gpu_mapping", [0])
        elif "models_alts" in self.config:
            model_subpath = self.config["models_alts"][model_key]
            gpu_list = self.config.get("gpu_mapping", {}).get(model_key, [0])
        else:
            model_subpath = model_key; gpu_list = [0]

        # Dataset
        base_data_dir = self._smart_path(self.config.get("input_dir") or self.config.get("data_dir", ""))
        d_name = self.config["dataset_alts"].get(dataset_key, dataset_key)
        d_path = os.path.join(base_data_dir, dataset_key) if dataset_key in self.config.get("dataset_alts", {}) else self._smart_path(dataset_key)

        return {
            "model_path": os.path.join(base_model_dir, model_subpath),
            "gpu_str": ",".join(map(str, gpu_list)),
            "num_gpus": len(gpu_list),
            "dataset_path": d_path,
            "dataset_name": d_name,
            "output_path": os.path.join(self._smart_path(self.config.get("output_dir", "results/")), task, dataset_key, model_key),
            "batch_size": self.config.get("eval_batch_size", 1),
            "max_output": self.config.get("max_output_length", 100),
            "model_max": self.config.get("model_max_length", 2048)
        }

    def _create_slurm_script(self, job_name, num_gpus, cmd_str, dry_run):
        slurm_filename = os.path.join(self.slurm_dir, f"{job_name}.sh")
        log_filename = os.path.join(self.log_dir, f"{job_name}_%j.txt")
        
        content = SLURM_TEMPLATE.format(
            num_gpus=num_gpus, job_name=job_name, log_path=log_filename,
            work_dir=self.base_dir, cmd_str=cmd_str
        )

        if dry_run:
            print(f"[DRY-GEN] {slurm_filename}\n{'-'*20}\n{content.strip()}\n{'-'*20}")
        else:
            with open(slurm_filename, "w") as f: f.write(content)
            print(f"[GEN] Created: {slurm_filename}")

    def _run_local_command(self, job_name, cmd_list, gpu_str, dry_run):
        if dry_run:
            print(f"[DRY-RUN] {job_name}: {' '.join(cmd_list)}")
            return

        print(f"[RUN] {job_name}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        try:
            subprocess.run(cmd_list, env=env, check=True)
        except Exception as e:
            logger.error(f"Job {job_name} failed: {e}")

    # =========================================================================
    # MODULE 4: PROGRESS TRACKER (Nested Class)
    # =========================================================================
    # class ProgressTracker:
    #     """
    #     Manages state of experiments to prevent re-running completed splits.
    #     Used by the training script, but defined here for consolidation.
    #     """
    #     def __init__(self, output_dir, model_name, dataset, task, pred_pattern, score_pattern):
    #         self.output_dir = output_dir
    #         self.filepath = os.path.join(output_dir, "progress.json")
    #         self.model_name = model_name
    #         self.dataset = dataset
    #         self.task = task
    #         self.pred_pattern = pred_pattern
    #         self.score_pattern = score_pattern
    #         self.state = self.load_or_reconstruct()

    #     def load_or_reconstruct(self):
    #         if os.path.exists(self.filepath):
    #             try:
    #                 with open(self.filepath, 'r') as f: return json.load(f)
    #             except Exception: pass
    #         return self._reconstruct_from_files()

    #     def _reconstruct_from_files(self):
    #         state = {}
    #         # Reconstruct logic based on globbing existing files
    #         glob_p = self.pred_pattern.format(model=self.model_name, dataset=self.dataset, task=self.task.lower(), split="*", shot="*")
    #         files = glob.glob(os.path.join(self.output_dir, glob_p))
            
    #         # Regex to extract split/shot from filename
    #         regex_str = re.escape(self.pred_pattern.format(model=self.model_name, dataset=self.dataset, task=self.task.lower(), split="___SPLIT___", shot="___SHOT___"))
    #         regex_str = regex_str.replace("___SPLIT___", r"(?P<split>.+?)").replace("___SHOT___", r"(?P<shot>\d+)")
    #         pattern = re.compile(regex_str)

    #         for fpath in files:
    #             match = pattern.match(os.path.basename(fpath))
    #             if match:
    #                 shot = match.group("shot")
    #                 split = match.group("split")
    #                 key = f"k_shot_{shot}"
    #                 if key not in state: state[key] = {}
    #                 state[key][split] = os.path.basename(fpath)
            
    #         # Check for score files (completion)
    #         glob_s = self.score_pattern.format(model=self.model_name, dataset=self.dataset, task=self.task.lower(), shot="*")
    #         s_files = glob.glob(os.path.join(self.output_dir, glob_s))
    #         # ... (Simulated regex for scores similar to above) ...
            
    #         self._save(state)
    #         return state

    #     def _save(self, state):
    #         with open(self.filepath, 'w') as f: json.dump(state, f, indent=4)

    #     def complete_split(self, shot_size, split_name, predfile):
    #         k_key = f"k_shot_{shot_size}"
    #         if k_key not in self.state: self.state[k_key] = {}
    #         self.state[k_key][split_name] = predfile
    #         self._save(self.state)

    #     def is_complete(self, shot_size, split_name):
    #         return split_name in self.state.get(f"k_shot_{shot_size}", {})

# =========================================================================
# CLI ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextEE Job Manager")
    parser.add_argument("-c", "--config", default="global_data/global_mapping.json", help="Path to config file")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: logs
    p_logs = subparsers.add_parser("logs", help="Parse Slurm logs for execution times")
    # p_logs.add_argument("files", nargs="*", help="Specific log files to parse")
    p_logs.add_argument( "-i", "--input", nargs='+', required=True, 
        help="Path to a directory OR specific file(s) to process."
    )
    p_logs.add_argument("-t", "--tasks", nargs="+")
    p_logs.add_argument("-d", "--datasets", nargs="+")
    p_logs.add_argument("-m", "--models", nargs="+")
    

    # Subcommand: report
    p_rep = subparsers.add_parser("report", help="Generate CSV report from JSON scores")
    p_rep.add_argument("-o", "--output", default="analysis/results_summary.csv")
    p_rep.add_argument("-m", "--models", nargs="+")
    p_rep.add_argument("-d", "--datasets", nargs="+")
    p_rep.add_argument("-t", "--tasks", nargs="+")
    p_rep.add_argument("-s", "--splits", action="store_true", help="Include split details")

    # Subcommand: jobs
    p_job = subparsers.add_parser("jobs", help="Generate or Run experiments")
    p_job.add_argument("-a", "--action", choices=["generate", "run"], default="generate")
    p_job.add_argument("-t", "--tasks", nargs="+")
    p_job.add_argument("-d", "--datasets", nargs="+")
    p_job.add_argument("-m", "--models", nargs="+")
    p_job.add_argument("-F", "--few-shot", nargs="+", type=int, help="Override few-shot sizes")
    p_job.add_argument("--restart", action="store_true", help="Pass restart flag to script")
    p_job.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    
    # Initialize Manager
    manager = JobManager(config_file=args.config)

    if args.command == "logs":
        input_source = args.input[0] if os.path.isdir(args.input[0]) and len(args.input) == 1 else args.input
        if len(args.input) == 1:
            # Check if the single item is a directory
            if os.path.isdir(args.input[0]):
                input_source = args.input[0] # Pass as string
            else:
                input_source = args.input # Keep as list (or pass string, generate_report handles both)
        manager.generate_report(
            input_source=input_source,
            filter_task=args.task,
            filter_dataset=args.dataset,
            filter_model=args.model
        )
    elif args.command == "report":
        manager.compile_results(args.output, args.models, args.datasets, args.tasks, args.splits)
    elif args.command == "jobs":
        manager.dispatch_jobs(args.action, args.tasks, args.datasets, args.models, args.few_shot, args.restart, args.dry_run)