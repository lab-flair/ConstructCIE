import os
import logging
import json
import glob
import re
import sys
import argparse
from main.progress_manager import ProgressManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 2. Configuration Mapping
# ==========================================
main_dir = os.path.dirname(os.path.realpath(__file__))

# # Load Global Mapping
# config_path = os.path.join(main_dir, "global_data", "global_mapping.json")
# try:
#     with open(config_path) as f:
#         CONFIG = json.load(f)
# except FileNotFoundError:
#     print(f"❌ Error: Could not find config at {config_path}")
#     sys.exit(1)
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# ==========================================
# 3. Main Execution Script
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Rebuild progress.json files from results directory.")
    
    # Action Flag
    parser.add_argument("-c", "--config_file", default=os.path.join(main_dir, "global_data", "global_mapping.json"), help="Path to the configuration JSON file")
    
    parser.add_argument('-r', '--recreate', action='store_true', 
                        help="If set, deletes existing progress.json files to force a full rebuild based on found prediction files.")
    
    # Filters
    parser.add_argument("-t", '--tasks', nargs='+', help="List of tasks to process (e.g., EAE ED)")
    parser.add_argument("-d", '--datasets', nargs='+', help="List of datasets to process (e.g., ace05 phee)")
    parser.add_argument("-m", '--models', nargs='+', help="List of model folder names to process (e.g., Llama-2-7b Zephyr-7B)")

    args = parser.parse_args()
    
    try:
        CONFIG = load_config(args.config_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find config at {args.config_file}")
        sys.exit(1)

    # Configuration
    ROOT_DIR = "results"
    PRED_PATTERN = "pred_{model}_{dataset}_{task}_{split}_k{shot}.json"
    SCORE_PATTERN = "scores_{model}_{dataset}_{task}_k{shot}.json"

    if not os.path.exists(ROOT_DIR):
        logger.error(f"Root directory '{ROOT_DIR}' does not exist.")
        return

    logger.info(f"🚀 Traversing '{ROOT_DIR}' to rebuild progress.json files...")
    
    if args.tasks: logger.info(f"   Filter Tasks: {args.tasks}")
    if args.datasets: logger.info(f"   Filter Datasets: {args.datasets}")
    if args.models: logger.info(f"   Filter Models: {args.models}")
    
    if args.recreate:
        logger.warning("⚠️  --recreate mode is ON: Existing progress.json files will be deleted and regenerated.")

    # Walk the directory tree
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        # We assume the structure: results/TASK/DATASET/MODEL
        rel_path = os.path.relpath(dirpath, ROOT_DIR)
        parts = rel_path.split(os.sep)
        
        # Check if we are at depth 3 (Task/Dataset/Model)
        if len(parts) == 3:
            task = parts[0]
            dataset = parts[1]
            model_dir_name = parts[2]
            
            # 1. Skip hidden folders
            if model_dir_name.startswith('.'): 
                continue

            # 2. Apply User Filters
            if args.tasks and task not in args.tasks:
                continue
            if args.datasets and dataset not in args.datasets:
                continue
            if args.models and model_dir_name not in args.models:
                continue

            # 3. Validate against config to ensure keys exist
            if model_dir_name not in CONFIG["models"]:
                logger.warning(f"Skipping unknown model in config: {model_dir_name}")
                continue
            if dataset not in CONFIG["dataset_alts"]:
                logger.warning(f"Skipping unknown dataset in config: {dataset}")
                continue

            logger.info(f"Processing: {task} / {dataset} / {model_dir_name}")

            # --- Logic to handle Recreation ---
            if args.recreate:
                progress_file = os.path.join(dirpath, "progress.json")
                if os.path.exists(progress_file):
                    try:
                        os.remove(progress_file)
                        logger.info(f"   -> Deleted existing progress.json")
                    except OSError as e:
                        logger.error(f"   -> Error deleting {progress_file}: {e}")

            # Run the Manager
            ProgressManager(
                output_dir=dirpath,
                model_name=CONFIG["models"][model_dir_name]["path"],
                dataset=CONFIG["dataset_alts"][dataset],
                task=task,
                pred_pattern=PRED_PATTERN,
                score_pattern=SCORE_PATTERN
            )
            
            # Optimization: Stop recursing deeper
            dirnames[:] = []

    logger.info("Done.")

if __name__ == "__main__":
    main()