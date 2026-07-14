import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def build_llm_cmd_list(target_script, *, task, d_path, d_name, model_path, out_path,
                        gpu_list, eval_bs, max_out, model_max,
                        sim_threshold, skip_threshold, key_map, col_defs,
                        shots, mins, base_cmd_tail=None, skip_cols=None):
    """Build the evaluate_llm.py argv for one (gpu, batch) sub-job.

    Pure argument assembly: every value here is already resolved by the
    caller (run.py, via the active global config) — this is the single
    place that knows the CLI shape, so neither run.py nor job_manager.py
    build argv themselves. Shared by both the "run" path (run_command below)
    and the "generate" path (job_manager.create_grouped_sbatch_parts), so
    a SLURM-generated command and a directly-executed one never drift apart.
    """
    cmd_list = [
        "python", target_script,
        "-t", task, "-d", d_path, "-D", d_name,
        "-m", model_path, "-o", out_path,
        "-g", *[str(x) for x in gpu_list],
        "-b", str(eval_bs),
        "-M", str(max_out),
        "-l", str(model_max),
        "--ac_sim_threshold", str(sim_threshold),
        "--skip_threshold", str(skip_threshold),
        "--key_map", str(key_map),
        "--col_defs", str(col_defs),
        "-F", *shots,
        "--shot_mins", *mins,
    ] + list(base_cmd_tail or [])
    if skip_cols:
        cmd_list += ["--skip_columns", *[str(c) for c in skip_cols]]
    return cmd_list


def run_command(cmd_list, *, env_overrides=None, label=None, dry_run=False, log_path=None):
    """Execute a resolved command (the actual `-a run` executor).

    run_arg.py resolves cmd_list/env_overrides via its argument-handling logic
    (config lookups, gpu/batch grouping); this just dispatches the subprocess.
    When log_path is given, stdout+stderr are teed to both the console and
    that file, mirroring the generate path's `#SBATCH --output=<log_path>`
    (which also merges both streams into one file). Opened in append mode
    since multiple groups belonging to the same job_name/timestamp share one
    log_path, same as how a job's grouped sbatch parts share one log file.
    Returns True on success, False on failure or dry-run (nothing was run).
    """
    cmd_str = " ".join(str(c) for c in cmd_list)
    if dry_run:
        logger.info("  🔍 [DRY] Would execute: %s", label or cmd_str)
        logger.info("    --> Command: %s", cmd_str)
        return False

    logger.info("\n🚀 [RUN] Executing: %s", label or cmd_str)
    env = os.environ.copy()
    env.update(env_overrides or {})
    try:
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as log_file:
                log_file.write(f"\n===== {label or cmd_str} =====\n")
                proc = subprocess.Popen(cmd_list, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log_file.write(line)
                proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd_list)
        else:
            subprocess.run(cmd_list, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("❌ Error running %s: %s", label or cmd_str, e)
        return False
    except KeyboardInterrupt:
        logger.warning("\n🛑 Stopped by user.")
        sys.exit(1)
