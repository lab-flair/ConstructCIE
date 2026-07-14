import os, logging, json, random, datetime, pprint, yaml
import numpy as np
import torch
from argparse import Namespace
# from models import *
import ipdb
import csv
from typing import Tuple, List
import re
import stanza

def _get_stanza_en(stanza_dir=None):
    kwargs = dict(lang='en', processors='tokenize', verbose=False,
                  download_method=None)
    if stanza_dir:
        kwargs["dir"] = stanza_dir
    return stanza.Pipeline(**kwargs)

logger = logging.getLogger(__name__)

VALID_TASKS = ["E2E", "ED", "EAE", "EARL", "JHE", "IHE"]


# Process-wide run id for grouping log/output files across multiple calls in

# TRAINER_MAP = {
#     ("DyGIEpp", "E2E"): DyGIEppE2ETrainer,
#     ("OneIE", "E2E"): OneIEE2ETrainer,
#     ("CRFTagging", "ED"): CRFTaggingEDTrainer, 
#     ("CRFTagging", "EAE"): CRFTaggingEAETrainer, 
#     ("EEQA", "ED"): EEQAEDTrainer, 
#     ("EEQA", "EAE"): EEQAEAETrainer, 
#     ("RCEE", "ED"): RCEEEDTrainer, 
#     ("RCEE", "EAE"): RCEEEAETrainer, 
#     ("TagPrime", "ED"): TagPrimeEDTrainer, 
#     ("TagPrime", "EAE"): TagPrimeEAETrainer, 
#     ("QueryAndExtract", "ED"): QueryAndExtractEDTrainer,
#     ("QueryAndExtract", "EAE"): QueryAndExtractEAETrainer,
#     ("Degree", "E2E"): DegreeE2ETrainer,
#     ("Degree", "ED"): DegreeEDTrainer,
#     ("Degree", "EAE"): DegreeEAETrainer,
#     ("UniST", "ED"): UniSTEDTrainer,
#     ("CEDAR", "ED"): CEDAREDTrainer,
#     ("PAIE", "EAE"): PAIEEAETrainer, 
#     ("XGear", "EAE"): XGearEAETrainer,
#     ("BartGen", "EAE"): BartGenEAETrainer,
#     ("Ampere", "EAE"): AmpereEAETrainer,
#     ("AMRIE", "E2E"): AMRIEE2ETrainer,
#     ("LLM", "EAE"): LLMEAETrainer,
# }
# MAPPING: (Model, Task) -> (Module Path, Class Name)
# NOTE: You must verify that the 'module paths' (e.g., 'models.dygiepp') 
# match the actual file location of your classes to avoid importing the main package.
import importlib # Required for dynamic imports
TRAINER_REGISTRY = {
    ("DyGIEpp", "E2E"): ("models.DyGIEpp", "DyGIEppE2ETrainer"),
    ("DyGIEpp", "AC"): ("models.DyGIEpp", "DyGIEppACTrainer"),
    ("OneIE", "E2E"):   ("models.OneIE",   "OneIEE2ETrainer"),
    ("OneIE", "AC"):    ("models.OneIE",   "OneIEACTrainer"),
    ("AMRIE", "AC"):    ("models.AMRIE",   "AMRIEACTrainer"),
    ("Degree", "AC"):   ("models.Degree",  "DegreeACTrainer"),
    ("EEQA", "AC"):     ("models.EEQA",    "EEQAACTrainer"),
    ("RCEE", "AC"):     ("models.RCEE",    "RCEEACTrainer"),
    ("QueryAndExtract", "AC"): ("models.QueryAndExtract", "QueryAndExtractACTrainer"),
    ("TagPrime", "AC"): ("models.TagPrime", "TagPrimeACTrainer"),
    ("TagPrime-C", "AC"): ("models.TagPrime", "TagPrimeACTrainer"),
    ("TagPrime-CR", "AC"): ("models.TagPrime", "TagPrimeACTrainer"),
    # ACH reuses the AC trainer — accident_type is given as input (same as
    # AC). evaluate_supervised's _load_split_data flattens nested constee
    # via convert_predictions("ACH","AC", ...) before training.
    ("TagPrime", "ACH"): ("models.TagPrime", "TagPrimeACTrainer"),
    ("TagPrime-C", "ACH"): ("models.TagPrime", "TagPrimeACTrainer"),
    ("TagPrime-CR", "ACH"): ("models.TagPrime", "TagPrimeACTrainer"),
    # E2AC / E2ACH — TagPrimeE2ACTrainer subclasses TagPrimeACTrainer and
    # adds a `predictE2AC` method (selected via PREDICT_METHOD for the E2
    # task family in evaluate_supervised). predictE2AC runs a per-doc
    # Level-0 probe so Level-1 prompts are built from the PREDICTED
    # accident_type instead of the gold value the eval data still carries
    # (gold is kept only for scoring). Model + training pipeline +
    # `skip_columns`-driven cls_factors logic (accident_type stays in
    # cls_factors for these tasks) are unchanged.
    # ("TagPrime", "E2AC"): ("models.TagPrime", "TagPrimeE2ACTrainer"),
    # ("TagPrime-C", "E2AC"): ("models.TagPrime", "TagPrimeE2ACTrainer"),
    # ("TagPrime-CR", "E2AC"): ("models.TagPrime", "TagPrimeE2ACTrainer"),
    # ("TagPrime", "JHE"): ("models.TagPrime", "TagPrimeE2ACTrainer"),
    ("TagPrime-C", "JHE"): ("models.TagPrime", "TagPrimeE2ACTrainer"),
    ("TagPrime-CR", "JHE"): ("models.TagPrime", "TagPrimeE2ACTrainer"),
    ("BartGen", "AC"):  ("models.BartGen", "BartGenACTrainer"),
    ("XGear", "AC"):    ("models.XGear",   "XGearACTrainer"),
    ("PAIE", "AC"):     ("models.PAIE",    "PAIEACTrainer"),
    ("Ampere", "AC"):   ("models.Ampere",  "AmpereACTrainer"),
    ("CRFTagging", "ED"): ("models.CRFTagging", "CRFTaggingEDTrainer"), 
    ("CRFTagging", "EAE"): ("models.CRFTagging", "CRFTaggingEAETrainer"), 
    ("EEQA", "ED"): ("models.EEQA", "EEQAEDTrainer"), 
    ("EEQA", "EAE"): ("models.EEQA", "EEQAEAETrainer"), 
    ("RCEE", "ED"): ("models.RCEE", "RCEEEDTrainer"), 
    ("RCEE", "EAE"): ("models.RCEE", "RCEEEAETrainer"), 
    ("TagPrime", "ED"): ("models.TagPrime", "TagPrimeEDTrainer"), 
    ("TagPrime", "EAE"): ("models.TagPrime", "TagPrimeEAETrainer"), 
    ("QueryAndExtract", "ED"): ("models.QueryAndExtract", "QueryAndExtractEDTrainer"),
    ("QueryAndExtract", "EAE"): ("models.QueryAndExtract", "QueryAndExtractEAETrainer"),
    ("Degree", "E2E"): ("models.Degree", "DegreeE2ETrainer"),
    ("Degree", "ED"): ("models.Degree", "DegreeEDTrainer"),
    ("Degree", "EAE"): ("models.Degree", "DegreeEAETrainer"),
    ("UniST", "ED"): ("models.UniST", "UniSTEDTrainer"),
    ("CEDAR", "ED"): ("models.CEDAR", "CEDAREDTrainer"),
    ("PAIE", "EAE"): ("models.PAIE", "PAIEEAETrainer"), 
    ("XGear", "EAE"): ("models.XGear", "XGearEAETrainer"),
    ("BartGen", "EAE"): ("models.BartGen", "BartGenEAETrainer"),
    ("Ampere", "EAE"): ("models.Ampere", "AmpereEAETrainer"),
    ("AMRIE", "E2E"): ("models.AMRIE", "AMRIEE2ETrainer"),
    ("LLM", "EAE"): ("models.LLM", "LLMEAETrainer"),
    ("LLM", "ED"): ("models.LLM", "LLMEDTrainer"),
    ("LLM", "JHE"):  ("models.LLM", "LLME2ACTrainer"),
    ("LLM", "IHE"): ("models.LLM", "LLME2AC1Trainer"),
    ("vLLM", "EAE"): ("models.LLM", "vLLMEAETrainer"),
}

def get_trainer_class(model_name: str, task: str):
    """
    Dynamically imports and returns the trainer class based on model and task.
    """
    key = (model_name, task)
    
    if key not in TRAINER_REGISTRY:
        raise ValueError(f"No trainer configuration found for Model: {model_name}, Task: {task}")

    module_path, class_name = TRAINER_REGISTRY[key]
    
    try:
        # This performs the import ONLY when this function is called
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        return trainer_class
    except ImportError as e:
        logger.error(f"Failed to import module '{module_path}'. Check your file structure.")
        raise e
    except AttributeError as e:
        logger.error(f"Class '{class_name}' not found in module '{module_path}'.")
        raise e

# Usage Example:
# TrainerClass = get_trainer_class("DyGIEpp", "E2E")
# trainer = TrainerClass(...)


def update_namespace(target_ns, source_ns):
    """
    Updates the target namespace with values from the source namespace.
    
    Args:
        target_ns (Namespace): The object to be updated (e.g., config).
        source_ns (Namespace): The object containing new values (e.g., args).
        
    Returns:
        The updated target namespace.
    """
    # specific check to ensure we are working with Namespaces or compatible objects
    if not hasattr(source_ns, '__dict__') and not isinstance(source_ns, dict):
        logger.info("Source is not a Namespace or dict. Skipping update.")
        return target_ns

    # Convert source to dict for iteration if it's a Namespace
    source_dict = vars(source_ns) if hasattr(source_ns, '__dict__') else source_ns
    
    logger.info(f"--- Merging Namespaces ---")
    
    for key, value in source_dict.items():
        # Optional: Skip None values if you don't want empty CLI args overwriting config
        if value is None:
            continue
            
        # Update the target
        old_value = getattr(target_ns, key, "Not Set")
        setattr(target_ns, key, value)
        
        # Only print if the value actually changed or is new
        if old_value != value:
            logger.info(f"Updated '{key}': {old_value} -> {value}")
            
    logger.info("--------------------------")
    return target_ns

def save_scores_to_csv(scores, output_path, extra_info=None):
    """
    Converts the scores dictionary to a list of dicts and saves to CSV.
    
    Args:
        scores (dict): The dictionary containing evaluation metrics.
        output_path (str): The file path to save the CSV (e.g., 'results.csv').
        extra_info (dict): Optional. Extra columns to add to every row (e.g., {'k_shot': 5}).
    """
    
    # Map internal keys to the display names used in your print_scores function
    key_mapping = {
        "trigger_id": "Tri-I",
        "trigger_cls": "Tri-C",
        "argument_id": "Arg-I",
        "argument_cls": "Arg-C",
        "argument_attached_id": "Arg-I (attached)",
        "argument_attached_cls": "Arg-C (attached)"
    }

    rows = []
    
    # Iterate through the known keys to maintain order
    for key, display_name in key_mapping.items():
        if key in scores:
            data = scores[key]
            
            # Create the row dictionary
            row = {
                "Metric": display_name,
                "Precision": f"{data['precision']:.2f}",
                "Recall": f"{data['recall']:.2f}",
                "F1": f"{data['f1']:.2f}",
                "Match": data['match_num'],
                "Pred": data['pred_num'],
                "Gold": data['gold_num']
            }
            
            # Add any extra info (like k-shot size) if provided
            if extra_info:
                row.update(extra_info)
                
            rows.append(row)

    # Write to CSV
    if rows:
        # Check if file exists to handle headers (if appending)
        file_exists = os.path.isfile(output_path)
        
        fieldnames = list(rows[0].keys())
        
        # Open in 'a' (append) mode so you can add multiple runs to one file
        # or use 'w' (write) if you want to overwrite every time.
        mode = 'a' if file_exists else 'w'
        
        with open(output_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(rows)
            
        logger.info(f"Scores saved to {output_path}")
    else:
        logger.info("No scores found to save.")

    return rows

# --- Helper: VRAM Monitor ---
def print_vram(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3 
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[{label}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    else:
        logger.error(f"[{label}] CUDA not available.")

def load_config(config_fn, intask = None):
    import _jsonnet
    config = json.loads(_jsonnet.evaluate_file(config_fn))
    config = Namespace(**config)
    if intask is not None:
        config.task = intask
    assert config.task in VALID_TASKS, f"Task must be in {VALID_TASKS}"

    # `cache_dir` is no longer required in supervised JSON configs — models load
    # from a local backbone dir resolved via `model_dir` in run.py. Provide a
    # None default so legacy model code that still passes `cache_dir=` to
    # `from_pretrained` keeps working (HF treats None as "use default cache").
    if not hasattr(config, "cache_dir"):
        config.cache_dir = None

    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

def set_gpu(gpu_device):
    if gpu_device >= 0:
        torch.cuda.set_device(gpu_device)
        
def set_gpus(gpu_device:Tuple[List[int], Tuple[int], int]):
    """
    Sets the GPU device(s) to be used.
    
    Args:
        gpu_device: Can be an integer (e.g., 0) or a list of integers (e.g., [1, 2]).
    """
    if isinstance(gpu_device, list) or isinstance(gpu_device, tuple):
        # Case: List of GPUs -> Set CUDA_VISIBLE_DEVICES
        # This masks the GPUs so PyTorch only sees the specific ones listed.
        # e.g., if input is [1, 2], PyTorch will see them as "cuda:0" and "cuda:1"
        if len(gpu_device) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_device))
            logger.info(f"Visible GPUs restricted to IDs: {gpu_device}")
    
    elif isinstance(gpu_device, int):
        # Case: Single Integer -> Use legacy set_device
        if gpu_device >= 0:
            torch.cuda.set_device(gpu_device)
        
def set_logger(config):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
    output_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                        handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
    logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
    
    # save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
        json.dump(vars(config), fp, indent=4)
        
    config.output_dir = output_dir
    config.log_path = log_path
    
    return config

def parse_unknown_args(unknown_args):
    args = {}
    key = None
    for unknown_arg in unknown_args:
        if unknown_arg.startswith("--"):
            key = unknown_arg[2:]
        else:
            args[key] = unknown_arg
    return args

def print_vram(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[{label}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    else:
        logger.info(f"[{label}] CUDA not available.")

def load_data(task, file, add_extra_info_fn, config):
    
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    objs = [json.loads(line) for line in lines]
    
    if task == "E2E":
        data, type_set = load_E2E_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
            len(data), len(type_set["trigger"]), len(type_set["role"]), file))
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task == "ED":
        data, type_set = load_ED_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
            len(data), len(type_set["trigger"]), file))
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif task == "EAE":
        data, type_set = load_EAE_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
            len(data), len(type_set["trigger"]), len(type_set["role"]), file))
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task == "EARL":
        data, type_set = load_EARL_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
            len(data), len(type_set["trigger"]), len(type_set["role"]), file))
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task in ("AC", "AC1", "E2AC", "E2AC1"):
        # AC / E2AC family: flat AC input (accd). E2AC uses the same input
        # shape as AC; the difference is purely in the trainer's
        # internal_predict logic (predicts accident_type vs reading it).
        data, type_set = load_AC_data(objs, add_extra_info_fn, config)
        if type_set:
            logger.info('Loaded {} AC instances ({} trigger types and {} role types) from {}'.format(
                len(data), len(type_set["trigger"]), len(type_set["role"]), file))
            logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
        else:
            logger.info('Loaded {} AC instances from {}'.format(len(data), file))
    elif task in ("JHE", "IHE"):
        # ACH / E2ACH family: constee records arrive nested
        # ({accident_report: {text, children}, accident_type: {value}}).
        # Flatten once at load so the downstream pipeline (cache builder,
        # prompts, scorer) sees the same flat AC factor shape as accd.
        # The reverse conversion happens at the save step —
        # see evaluate_llm.py for flat→nested before json.dump.
        data = ach_to_flat_ac(objs)
        type_set = {}
        logger.info('Loaded %d %s instances (flattened from nested) from %s',
                    len(data), task, file)
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return data, type_set

def load_all_data(config, add_extra_info_fn):
    
    with open(config.train_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    train_objs = [json.loads(line) for line in lines]
    
    with open(config.dev_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    dev_objs = [json.loads(line) for line in lines]
    
    with open(config.test_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    test_objs = [json.loads(line) for line in lines]
    
    if config.task == "E2E":
        train_data, train_type_set = load_E2E_data(train_objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
            len(train_data), len(train_type_set["trigger"]), len(train_type_set["role"]), config.train_file))
        dev_data, dev_type_set = load_E2E_data(dev_objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
            len(dev_data), len(dev_type_set["trigger"]), len(dev_type_set["role"]), config.dev_file))
        test_data, test_type_set = load_E2E_data(test_objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
            len(test_data), len(test_type_set["trigger"]), len(test_type_set["role"]), config.test_file))
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif config.task == "ED":
        train_data, train_type_set = load_ED_data(train_objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
            len(train_data), len(train_type_set["trigger"]), config.train_file))
        dev_data, dev_type_set = load_ED_data(dev_objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
            len(dev_data), len(dev_type_set["trigger"]), config.dev_file))
        test_data, test_type_set = load_ED_data(test_objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
            len(test_data), len(test_type_set["trigger"]), config.test_file))
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"]}
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif config.task == "EAE":
        train_data, train_type_set = load_EAE_data(train_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
            len(train_data), len(train_type_set["trigger"]), len(train_type_set["role"]), config.train_file))
        dev_data, dev_type_set = load_EAE_data(dev_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
            len(dev_data), len(dev_type_set["trigger"]), len(dev_type_set["role"]), config.dev_file))
        test_data, test_type_set = load_EAE_data(test_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
            len(test_data), len(test_type_set["trigger"]), len(test_type_set["role"]), config.test_file))
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif config.task == "EARL":
        train_data, train_type_set = load_EARL_data(train_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
            len(train_data), len(train_type_set["trigger"]), len(train_type_set["role"]), config.train_file))
        dev_data, dev_type_set = load_EARL_data(dev_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
            len(dev_data), len(dev_type_set["trigger"]), len(dev_type_set["role"]), config.dev_file))
        test_data, test_type_set = load_EARL_data(test_objs, add_extra_info_fn, config)
        logger.info('Loaded {} EARL instances ({} trigger types and {} role types) from {}'.format(
            len(test_data), len(test_type_set["trigger"]), len(test_type_set["role"]), config.test_file))
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return train_data, dev_data, test_data, type_set

def load_text(task, file, add_extra_info_fn, config):
    
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        
    nlp_en = _get_stanza_en(getattr(config, "stanza_path", None))

    objs = []
    offset_map = []
    for i, line in enumerate(lines):
        line = line.strip()
        res_text = nlp_en(line)
        text_tokens = [t.text for s in res_text.sentences for t in s.tokens]
        text_offsets = [(t.start_char, t.end_char) for s in res_text.sentences for t in s.tokens]
        obj = {
            "doc_id": f"DOC_{i:06d}", 
            "wnd_id": f"DOC_{i:06d}", 
            "text": line, 
            "lang": "en", 
            "tokens": text_tokens,
            "entity_mentions": [], 
            "event_mentions": [], 
        }
        objs.append(obj)
        offset_map.append(text_offsets)

    if task == "E2E":
        data, _ = load_E2E_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} E2E instances from {}'.format(len(data), file))
    elif task == "ED":
        data, _ = load_ED_data(objs, add_extra_info_fn, config)
        logger.info('Loaded {} ED instances from {}'.format(len(data), file))
        
    assert len(data) == len(offset_map)
        
    return data, offset_map

def load_E2E_data(data, add_extra_info_fn, config):
    
    instances = []
    for dt in data:

        entities = dt['entity_mentions']

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        events = []
        entity_map = {entity['id']: entity for entity in entities}
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                
                # argument = (start index, end index, role type, text span)
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)

            arguments.sort(key=lambda x: (x[0], x[1]))
            events.append({"trigger": trigger, "arguments": arguments})

        events.sort(key=lambda x: (x['trigger'][0], x['trigger'][1]))
        
        instance = {"doc_id": dt["doc_id"], 
                    "wnd_id": dt["wnd_id"], 
                    "tokens": dt["tokens"], 
                    "text": dt["text"], 
                    "events": events, 
                   }

        instances.append(instance)

    trigger_type_set = set()
    for instance in instances:
        for event in instance['events']:
            trigger_type_set.add(event['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for event in instance['events']:
            for argument in event["arguments"]:
                role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    return new_instances, type_set

def load_ED_data(data, add_extra_info_fn, config):

    instances = []
    for dt in data:

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        triggers = []
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            triggers.append(trigger)

        triggers.sort(key=lambda x: (x[0], x[1]))
        
        instance = {"doc_id": dt["doc_id"], 
                    "wnd_id": dt["wnd_id"], 
                    "tokens": dt["tokens"], 
                    "text": dt["text"], 
                    "triggers": triggers,
                   }

        instances.append(instance)

    trigger_type_set = set()
    for instance in instances:
        for trigger in instance['triggers']:
            trigger_type_set.add(trigger[2])

    type_set = {"trigger": trigger_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    return new_instances, type_set

def load_EAE_data(data, add_extra_info_fn, config):

    instances = []
    for dt in data:
        
        entities = dt['entity_mentions']

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        entity_map = {entity['id']: entity for entity in entities}
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                
                # argument = (start index, end index, role type, text span)
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)

            arguments.sort(key=lambda x: (x[0], x[1]))
            
            instance = {"doc_id": dt["doc_id"], 
                        "wnd_id": dt["wnd_id"], 
                        "tokens": dt["tokens"], 
                        "text": dt["text"], 
                        "trigger": trigger, 
                        "arguments": arguments, 
                       }

            instances.append(instance)
            
    trigger_type_set = set()
    for instance in instances:
        trigger_type_set.add(instance['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for argument in instance["arguments"]:
            role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    return new_instances, type_set

def load_EARL_data(data, add_extra_info_fn, config):

    instances = []
    for dt in data:
        
        entities = dt['entity_mentions']

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])
        
        entity_map = {entity['id']: entity for entity in entities}

        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])
            
            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                
                # argument = (start index, end index, role type, text span)
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)
            
            labeled_entities = set([(a[0], a[1]) for a in arguments]) 
            non_labeled_entities = set([(e['start'], e['end'], None, e['text']) for e in entities if (e['start'], e['end']) not in labeled_entities])
            arguments.extend(list(non_labeled_entities))
            arguments.sort(key=lambda x: (x[0], x[1]))
            
            instance = {"doc_id": dt["doc_id"], 
                        "wnd_id": dt["wnd_id"], 
                        "tokens": dt["tokens"], 
                        "text": dt["text"], 
                        "trigger": trigger, 
                        "arguments": arguments,
                       }

            instances.append(instance)
            
    trigger_type_set = set()
    for instance in instances:
        trigger_type_set.add(instance['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for argument in instance["arguments"]:
            if argument[2] is not None:
                role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    return new_instances, type_set

def convert_ED_to_EAE(data, gold):
    instances = []
    for dt, gd in zip(data, gold):
        for trigger in dt["triggers"]:
            trigger_ = (trigger[0], trigger[1], trigger[2], " ".join(gd["tokens"][trigger[0]:trigger[1]]))
            instance = {"doc_id": gd["doc_id"], 
                        "wnd_id": gd["wnd_id"], 
                        "tokens": gd["tokens"], 
                        "text": gd["text"], 
                        "trigger": trigger_, 
                        "arguments": [], 
                        "extra_info": gd["extra_info"]
                       }
            instances.append(instance)
    
    return instances

def combine_ED_and_EAE_to_E2E(ed_predicitons, eae_predictions):
    e2e_predictions = []
    idx = 0
    for ed_prediciton in ed_predicitons:
        events = []
        for trigger in ed_prediciton["triggers"]:
            eae_prediction = eae_predictions[idx]
            assert ed_prediciton["doc_id"] == eae_prediction["doc_id"]
            assert ed_prediciton["wnd_id"] == eae_prediction["wnd_id"]
            assert trigger[0] == eae_prediction["trigger"][0]
            assert trigger[1] == eae_prediction["trigger"][1]
            assert trigger[2] == eae_prediction["trigger"][2]
            events.append({"trigger": trigger, "arguments": eae_prediction["arguments"]})
            idx += 1
        
        ed_prediciton["events"] = events
        e2e_predictions.append(ed_prediciton)

    return e2e_predictions

def resolve_skip_columns(skip_columns, config):
    """Return the safe-to-drop subset of `skip_columns`.

    The key_map's `task.default.columns` lists structural / source fields
    that the IE converter needs (typically `id`, `accident_report`,
    `accident_type`). Those must never be stripped at LOAD time — doing so
    wipes out the source text and the accident_type trigger, producing
    zero training instances. They CAN still be filtered at SCORE time by
    the scorer's own skip_columns handling, so excluding them here has no
    effect on what gets evaluated.
    """
    key_map = getattr(config, "key_map", None) or {}
    protected_list = ((key_map.get("task") or {})
                      .get("default") or {}).get("columns") or []
    protected = {c for c in protected_list if isinstance(c, str)}
    return {c for c in (skip_columns or [])
            if isinstance(c, str) and c not in protected}


def _strip_skip_columns(data, config):
    """Drop columns listed in `config.skip_columns` from each raw ACCD record.

    Returns a new list (does not mutate the caller's list). Protected
    default keys from `key_map.task.default.columns` are kept in every
    record via `resolve_skip_columns`.

    Effect downstream:
      * convert_AC_to_EAE walks `structure[mf]` for sub-factors but reads
        each field via `flat_data.get(key)` — popped keys yield None and
        are silently skipped, so no event_mentions / entity_mentions are
        produced for skipped roles.
      * load_EAE_data builds `type_set["role"]` from instance arguments;
        skipped roles never appear there → never get a BIO label in
        the model's label_stoi → model has no head class to emit them.
      * The AC scorer iterates over roles present in gold; skipped
        columns produce zero gold counts → excluded from OVERALL F1
        (no 0/0 row drag).
    """
    skip_set = resolve_skip_columns(getattr(config, "skip_columns", None), config)
    if not skip_set:
        return data
    logger.info(f"[skip_columns] stripping {len(skip_set)} role columns: {sorted(skip_set)}")
    return [
        {k: v for k, v in rec.items() if k not in skip_set}
        for rec in data
    ]


def load_AC_data(data, add_extra_info_fn, config, is_llm=True):
    """Load ACCD data for the LLM AC trainer or a supervised IE model.

    is_llm=True  (default) — returns the original flat ACCD records so the LLM
                             AC trainer can process them in its own prompt format.
    is_llm=False           — converts records to standard IE format and returns
                             E2E-formatted instances that preserve the full
                             main-factor → sub-factor extraction hierarchy as
                             joint events.  Use load_AC_supervised_data() as a
                             convenience wrapper.

    Applies `config.skip_columns` first so both paths see the filtered records.
    """
    data = _strip_skip_columns(data, config)

    if is_llm:
        return data, {}

    converted_ie_data = convert_predictions(data, "AC", "IE", config)
    e2e_instances, type_set = load_E2E_data(converted_ie_data, add_extra_info_fn, config)
    return e2e_instances, type_set


def load_gold_data(task, file_path, add_extra_info_fn, config):
    """Load gold data in the format expected by compute_scores.

    For AC, compute_scores needs the original flat ACCD records (is_llm=True).
    For all other tasks, the standard IE instances from load_data are correct.
    """
    return load_data(task, file_path, add_extra_info_fn, config)


def inject_source_text(preds, gold_data):
    """Stitch `accident_report` from gold records onto matching pred records by doc_id.

    Most trainers' `_to_ie_format` drops the source text and only forwards
    {doc_id, wnd_id, tokens, entity_mentions, event_mentions}. Without the
    source string, `ie_preds_to_flat_ac` can't take the offset-map path and
    falls back to `" ".join(tokens)`, which inserts spaces around punctuation
    ("#1" → "# 1", "hammer-drill" → "hammer - drill") and tanks sim=1.0 exact
    match.

    Idempotent: skips preds that already carry `text`/`accident_report`.
    Modifies the pred dicts in place and returns the list.
    """
    src_by_id = {}
    for g in gold_data or []:
        if not isinstance(g, dict):
            continue
        doc_id = str(g.get("id", g.get("doc_id", "")))
        if not doc_id:
            continue
        v = g.get("accident_report") or g.get("text")
        if isinstance(v, list):
            v = v[0] if v else ""
        if v:
            src_by_id[doc_id] = v
    for p in preds or []:
        if not isinstance(p, dict):
            continue
        if p.get("text") or p.get("accident_report"):
            continue
        pid = str(p.get("doc_id", p.get("id", "")))
        if pid in src_by_id:
            p["text"] = src_by_id[pid]
    return preds


def ie_preds_to_flat_ac(preds, acc_types):
    """Convert IE-format predictAC output to flat ACCD dicts for compute_AC_flat_score.

    Input:  [{doc_id, wnd_id, tokens, entity_mentions, event_mentions}, ...]
    Output: [{id, doc_id, accident_type, <factor_key>: "<text>", ...}, ...]

    entity_mentions carry (entity_type=factor_key, start, end); the factor text
    is reconstructed by joining the token span.  Multiple entities with the same
    entity_type are joined with ";".

    acc_types: list of accident type strings from
               key_map["task"]["classification"]["classes"]["accident_type"].
               Compared in lowercase against event_type strings produced by the model.
    """
    if not acc_types:
        raise ValueError("ie_preds_to_flat_ac requires a non-empty acc_types list")
    # `root_types` maps a normalized comparison key (lower + replace -,/ with _)
    # back to the canonical raw label from `acc_types`. Incoming IE event_types
    # come in two forms:
    #   * Level-1 trigger events emit the normalized form because
    #     convert_AC_to_EAE normalizes acc_type when building IE for
    #     training/inference.
    #   * The cls head injects the RAW form (straight from `class_itos`).
    # Matching on the normalized key accepts either form, and the emitted
    # acc_type uses the canonical raw label so the flat output aligns with
    # gold flat AC (ach_to_flat_ac / e2e_to_flat_ac_gold keep raw class
    # strings).
    def _norm_at(s):
        return str(s).lower().replace("-", "_").replace("/", "_")
    root_types = {_norm_at(t): t for t in acc_types}

    flat_preds = []
    for pred in preds:
        tokens  = pred.get("tokens", [])
        doc_id  = pred.get("doc_id", "")
        source_text = pred.get("text") or pred.get("accident_report") or ""
        # Token→char offset map, built once per record. Used to slice the
        # verbatim source span when a model emits (start, end) without `text`,
        # avoiding " ".join(...) detokenizer artifacts ("hammer - drill" instead
        # of "hammer-drill"). None when no source text is available.
        offset_map = generate_offset_map(tokens, source_text) if source_text and tokens else None

        acc_type = "accident_report"
        for em in pred.get("event_mentions", []):
            normed = _norm_at(em.get("event_type", ""))
            if normed in root_types:
                acc_type = root_types[normed]
                break

        factor_texts = {}
        for ent in pred.get("entity_mentions", []):
            key   = ent.get("entity_type", "")
            if not key:
                continue
            # Prefer the generated text (set when span_map is disabled by
            # XGear AC trainer). Otherwise slice the verbatim substring of
            # source_text using offset_map; fall back to whitespace-joined
            # tokens when neither path is available.
            entity_text = ent.get("text")
            if entity_text is None:
                start = ent.get("start", 0)
                end   = ent.get("end", 0)
                if start >= end:
                    continue
                if offset_map is not None and 0 <= start < end <= len(offset_map):
                    entity_text = source_text[offset_map[start][0]:offset_map[end - 1][1]]
                else:
                    entity_text = " ".join(tokens[start:end])
            if not entity_text:
                continue
            if key in factor_texts:
                factor_texts[key] = factor_texts[key] + ";" + entity_text
            else:
                factor_texts[key] = entity_text

        flat_pred = {"id": doc_id, "doc_id": doc_id, "accident_type": acc_type}
        flat_pred.update(factor_texts)
        flat_preds.append(flat_pred)

    return flat_preds


def e2e_to_flat_ac_gold(instances, acc_types):
    """Convert E2E-format instances (from load_AC_supervised_data) to flat AC gold dicts.

    Each E2E instance carries the full event hierarchy:
      Root event   : trigger type = acc_type,     arguments = main factors
      Sub-events   : trigger type = main factor,  arguments = sub-factors
    All argument roles are flattened into the same dict level, matching the
    format expected by compute_AC_flat_score.

    acc_types: list of acc_type class strings from key_map (compared lowercase).
    """
    root_types = {t.lower().replace("-", "_").replace("/", "_") for t in acc_types}
    flat_golds = []
    for inst in instances:
        tokens  = inst.get("tokens", [])
        doc_id  = inst.get("doc_id", "")

        acc_type     = "accident_report"
        factor_texts = {}

        for event in inst.get("events", []):
            trig  = event.get("trigger", ())
            etype = trig[2] if len(trig) > 2 else ""
            if etype.lower() in root_types:
                acc_type = etype

            for arg in event.get("arguments", []):
                start, end, role = arg[0], arg[1], arg[2]
                text = arg[3] if len(arg) > 3 and arg[3] else " ".join(tokens[start:end])
                if not text:
                    continue
                if role in factor_texts:
                    factor_texts[role] += ";" + text
                else:
                    factor_texts[role] = text

        flat_gold = {"id": doc_id, "doc_id": doc_id, "accident_type": acc_type}
        flat_gold.update(factor_texts)
        flat_golds.append(flat_gold)

    return flat_golds


def eae_to_flat_ac_gold(eae_instances, acc_types):
    """Convert EAE instances (grouped by doc) to flat AC gold dicts.

    EAE-based AC training produces one instance per event.  Multiple instances
    may share the same (doc_id, wnd_id) — they are aggregated into one flat
    AC dict per document, matching the format expected by compute_AC_scores.

    acc_types: list of acc_type class strings from key_map (compared lowercase).
    """
    from collections import defaultdict

    root_types = {t.lower().replace("-", "_").replace("/", "_") for t in acc_types}

    doc_map   = defaultdict(list)
    doc_order = []
    for inst in eae_instances:
        key = (inst["doc_id"], inst["wnd_id"])
        if key not in doc_map:
            doc_order.append(key)
        doc_map[key].append(inst)

    flat_golds = []
    for key in doc_order:
        instances    = doc_map[key]
        tokens       = instances[0].get("tokens", [])
        doc_id       = instances[0].get("doc_id", "")
        acc_type     = "accident_report"
        factor_texts = {}

        for inst in instances:
            trig  = inst.get("trigger", ())
            etype = trig[2] if isinstance(trig, (list, tuple)) and len(trig) > 2 else ""
            if etype.lower() in root_types:
                acc_type = etype

            for arg in inst.get("arguments", []):
                start, end, role = arg[0], arg[1], arg[2]
                text = arg[3] if len(arg) > 3 and arg[3] else " ".join(tokens[start:end])
                if not text:
                    continue
                if role in factor_texts:
                    factor_texts[role] += ";" + text
                else:
                    factor_texts[role] = text

        flat_gold = {"id": doc_id, "doc_id": doc_id, "accident_type": acc_type}
        flat_gold.update(factor_texts)
        flat_golds.append(flat_gold)

    return flat_golds


def print_dev_scores(scores):
    """Print score_graphs results in the original DyGIEpp/OneIE tabular format."""
    SEP = "---------------------------------------------------------------------"

    def _row(label, d):
        p  = d.get("prec",      0) * 100
        r  = d.get("rec",       0) * 100
        f  = d.get("f",         0) * 100
        mn = d.get("match_num", 0)
        pn = d.get("pred_num",  0)
        gn = d.get("gold_num",  0)
        print('{:<10} - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            label, p, mn, pn, r, mn, gn, f))

    print(SEP)
    if "entity" in scores:
        _row("Entity",    scores["entity"])
    print(SEP)
    if "trigger_id" in scores:
        _row("Trigger I", scores["trigger_id"])
    if "trigger" in scores:
        _row("Trigger C", scores["trigger"])
    print(SEP)
    if "relation" in scores:
        _row("Relation",  scores["relation"])
    print(SEP)
    if "role_id" in scores:
        _row("Role I",    scores["role_id"])
    if "role" in scores:
        _row("Role C",    scores["role"])
    print(SEP)


def _ach_node_to_flat(children, flat):
    """Recursively flatten constee's `{factor: node}` children dict into
    `flat[factor] = [text/value, ...]`. Sub-factor nesting collapses — every
    descendant node's `.text` (extraction) or `.value` (classification)
    becomes a flat top-level entry under its key, matching the ACCD layout
    the AC scorer iterates.

    Extraction nodes also carry a parallel `keywords` list (one inner list of
    required keywords per span). When present, we forward it as
    `flat["<factor>_keywords"]` aligned with `flat["<factor>"]`, matching
    the parallel-key convention ACCD already uses. The AC scorer ignores
    `*_keywords` keys; the keyword scorer (compute_AC_flat_keyword_score)
    reads them.
    """
    for fkey, node in (children or {}).items():
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        if node_type == "extraction":
            payload = node.get("text") or []
        elif node_type == "classification":
            payload = node.get("value") or []
        else:
            payload = []
        if payload:
            payload_list = payload if isinstance(payload, list) else [payload]
            flat.setdefault(fkey, []).extend(
                v for v in payload_list if v not in (None, "")
            )
            # Forward the parallel keywords list for extraction nodes. Length
            # must match the post-filter span count above so the keyword
            # scorer can align span[i] ↔ keywords[i] — we re-apply the same
            # (None, "") filter to the keywords side using the original
            # payload as the alignment reference.
            if node_type == "extraction":
                kws = node.get("keywords") or []
                if isinstance(kws, list) and len(kws) == len(payload_list):
                    aligned = [k for v, k in zip(payload_list, kws)
                               if v not in (None, "")]
                    flat.setdefault(f"{fkey}_keywords", []).extend(aligned)
        # Recurse into grandchildren regardless of node type — classification
        # nodes typically don't have children, but the walk is defensive.
        _ach_node_to_flat(node.get("children") or {}, flat)


def flat_ac_to_ach(records, key_map):
    """Inverse of `ach_to_flat_ac` — rebuild constee's nested record shape
    from a flat AC dict, using `key_map`'s `structure` to walk children
    and `task.classification.classes` to decide each node's `type`.

    Designed for write-time conversion of LLM predictions: the trainer
    operates on flat AC dicts, and at save time we re-emit nested ACH so
    on-disk pred files match the original constee input shape.
    """
    structure = (key_map or {}).get("structure", {}) or {}
    cls_keys = set(((key_map or {}).get("task", {})
                    .get("classification", {}).get("classes") or {}).keys())

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if x not in (None, "")]
        return [v]

    def _build_node(factor_key, flat_record):
        """Recursively build the nested node for one factor key.

        Type is decided by `cls_keys` membership (classification when in,
        extraction otherwise). Children are walked from `structure`.
        """
        is_cls = factor_key in cls_keys
        node = {}
        vals = _as_list(flat_record.get(factor_key))
        if is_cls:
            node["type"] = "classification"
            node["value"] = vals
        else:
            node["type"] = "extraction"
            node["text"] = vals
        child_keys = structure.get(factor_key, []) or []
        if child_keys:
            node["children"] = {
                k: _build_node(k, flat_record)
                for k in child_keys
                if flat_record.get(k) not in (None, "", [])
            }
        return node

    nested_records = []
    for rec in records:
        doc_children_keys = structure.get("accident_report", []) or []
        nested = {
            "id": rec.get("id"),
            "accident_report": {
                "text": _as_list(rec.get("accident_report")),
                "children": {
                    k: _build_node(k, rec)
                    for k in doc_children_keys
                    if rec.get(k) not in (None, "", [])
                },
            },
            "accident_type": {
                "type": "classification",
                "value": _as_list(rec.get("accident_type")),
            },
        }
        nested_records.append(nested)
    return nested_records


def ach_to_flat_ac(records):
    """Flatten constee-style nested records into AC-scorer-compatible flat dicts.

    Input record:
        {"id": ...,
         "accident_report": {"text": [...], "children": {...}},
         "accident_type":   {"type": "classification", "value": [...]}}

    Output:
        {"id": ...,
         "accident_report": [<doc text>],
         "accident_type":   [<acc_type value>],
         "<factor>":        [<text/value>, ...],
         ...}

    Factor keys retain their long-form names (working_circumstances, etc.).
    """
    flat_records = []
    for rec in records:
        flat = {"id": rec.get("id")}

        report = rec.get("accident_report")
        if isinstance(report, dict):
            text = report.get("text") or []
            flat["accident_report"] = list(text) if isinstance(text, list) else [text]
            _ach_node_to_flat(report.get("children") or {}, flat)

        atype = rec.get("accident_type")
        if isinstance(atype, dict):
            val = atype.get("value") or []
            flat["accident_type"] = list(val) if isinstance(val, list) else [val]
        elif isinstance(atype, list):
            flat["accident_type"] = list(atype)
        elif atype:
            flat["accident_type"] = [atype]

        flat_records.append(flat)
    return flat_records


def convert_predictions(data, source_format, target_format, config):
    """Convert a list of records between data formats.

    Supported conversions:
        (X,  X)    no-op pass-through
        (IE, AC)   IE-format predictions (entity_mentions/event_mentions)
                   → flat ACCD dicts. Used to score supervised AC models
                   against ACCD gold.
        (AC, IE)   flat ACCD records → hierarchical IE format via
                   convert_AC_to_EAE. Used by load_AC_data / load_AC_eae_data
                   when training supervised models on ACCD.
        (ACH, AC)  constee-style nested records (accident_report.children,
                   accident_type.value, recursive sub-factors) → flat ACCD
                   dicts. Lets ACH/ACH1 results be scored through the same
                   compute_AC_flat_score path used by AC.

    Raises ValueError for any other (source, target) pair.
    """
    if source_format == target_format:
        return data

    key_map = getattr(config, "key_map", None) or {}

    if source_format == "IE" and target_format == "AC":
        acc_types = (key_map.get("task", {}).get("classification", {})
                     .get("classes", {}).get("accident_type", []))
        return ie_preds_to_flat_ac(data, acc_types)

    if source_format == "AC" and target_format == "IE":
        structure       = key_map.get("structure", key_map)
        mode            = getattr(config, "span_mode", "individual")
        stanza_pipeline = _get_stanza_en(getattr(config, "stanza_path", None))
        cls_factors     = set((key_map.get("task", {}).get("classification", {})
                              .get("classes", {}) or {}).keys())
        # E2* tasks predict accident_type; AC tasks receive it as input. The
        # distinction is encoded in `skip_columns`: AC lists accident_type
        # (skipped from scoring because it's gold-input), E2 omits it. Using
        # skip_columns here keeps the AC/E2 differentiation a single source
        # of truth (same flag drives the cls head's vocab in
        # TagPrimeACModel._build_cls_vocab).
        skip_set = set(getattr(config, "skip_columns", None) or [])
        predict_acc_type = "accident_type" not in skip_set
        return [convert_AC_to_EAE(item, structure,
                                  stanza_pipeline=stanza_pipeline, mode=mode,
                                  cls_factors=cls_factors,
                                  predict_acc_type=predict_acc_type)
                for item in data]

    if source_format == "ACH" and target_format == "AC":
        return ach_to_flat_ac(data)

    if source_format == "AC" and target_format == "ACH":
        # Inverse of (ACH, AC): rebuild nested constee shape from flat AC
        # predictions for save-time output (e.g. evaluate_llm.py writes
        # pred files in the same nested format as the input dataset).
        return flat_ac_to_ach(data, key_map)

    if source_format == "ACH" and target_format == "IE":
        # ACH → IE chains the existing converters: first flatten the nested
        # constee record into the long-name flat-AC shape, then run
        # convert_AC_to_EAE (which already speaks long names).
        structure       = key_map.get("structure", key_map)
        mode            = getattr(config, "span_mode", "individual")
        stanza_pipeline = _get_stanza_en(getattr(config, "stanza_path", None))
        cls_factors     = set((key_map.get("task", {}).get("classification", {})
                              .get("classes", {}) or {}).keys())
        # E2* tasks predict accident_type; AC tasks receive it as input. The
        # distinction is encoded in `skip_columns`: AC lists accident_type
        # (skipped from scoring because it's gold-input), E2 omits it. Using
        # skip_columns here keeps the AC/E2 differentiation a single source
        # of truth (same flag drives the cls head's vocab in
        # TagPrimeACModel._build_cls_vocab).
        skip_set = set(getattr(config, "skip_columns", None) or [])
        predict_acc_type = "accident_type" not in skip_set
        return [convert_AC_to_EAE(item, structure,
                                  stanza_pipeline=stanza_pipeline, mode=mode,
                                  cls_factors=cls_factors,
                                  predict_acc_type=predict_acc_type)
                for item in ach_to_flat_ac(data)]

    raise ValueError(
        f"convert_predictions: unsupported conversion "
        f"{source_format!r} -> {target_format!r}. "
        f"Supported: same-format pass-through, (IE, AC), (AC, IE), "
        f"(ACH, AC), (AC, ACH), (ACH, IE)."
    )


def load_AC_supervised_data(data, add_extra_info_fn, config):
    """Convenience wrapper — loads ACCD as joint E2E instances for supervised models.

    Each instance contains the full event hierarchy:
      Root event   : acc_type trigger  → main factors as arguments
      Sub-events   : main-factor trigger → sub-factors as arguments
    Both levels are present in the same instance so the model trains on the
    complete main→sub extraction structure without cascading gold-trigger leakage.
    """
    return load_AC_data(data, add_extra_info_fn, config, is_llm=False)


# predict_ac_hierarchical and _enforce_sub_factor_spans were moved to
# TextEE/models/ACtrainer.py — they're only consumed by the supervised AC
# trainer mixins, so they live next to E2EACMixin / EAEACMixin now.


# ---------------------------------------------------------------------------
# AC + EAE support
# ---------------------------------------------------------------------------

def load_AC_eae_data(data, add_extra_info_fn, config):
    """Load flat ACCD records as EAE instances for EAE-based supervised models.

    Converts each record to hierarchical IE format (via convert_AC_to_EAE),
    then calls load_EAE_data so every event in the hierarchy becomes one
    training instance with its gold trigger attached.

    Applies `config.skip_columns` upfront so the skipped role columns never
    flow into training data (no entity_mentions, no BIO labels in type_set).
    """
    data = _strip_skip_columns(data, config)
    converted_ie_data = convert_predictions(data, "AC", "IE", config)
    return load_EAE_data(converted_ie_data, add_extra_info_fn, config)


def save_predictions(file, predictions, data=None, offset_map=None):
    if data:
        assert len(predictions) == len(data)
        
    with open(file, 'w') as fp:
        for i, prediction in enumerate(predictions):
            event_mentions = []
            for event in prediction["events"]:
                arguments = [{"role": a[2], "start": a[0], "end": a[1]} for a in event["arguments"]]
                
                event_mention = {
                    "event_type": event["trigger"][2],  
                    "trigger": {
                        "start": event["trigger"][0], 
                        "end": event["trigger"][1], 
                    }, 
                    "arguments": arguments
                }
                
                if data and offset_map:
                    event_mention["trigger"]["text"] = data[i]["text"][offset_map[i][event_mention["trigger"]["start"]][0]:offset_map[i][event_mention["trigger"]["end"]-1][1]]
                    event_mention["trigger"]["offset_start"] = offset_map[i][event_mention["trigger"]["start"]][0]
                    event_mention["trigger"]["offset_end"] = offset_map[i][event_mention["trigger"]["end"]-1][1]
                    for a in arguments:
                        a["text"] = data[i]["text"][offset_map[i][a["start"]][0]:offset_map[i][a["end"]-1][1]]
                        a["offset_start"] = offset_map[i][a["start"]][0]
                        a["offset_end"] = offset_map[i][a["end"]-1][1]
                else:
                    event_mention["trigger"]["text"] = " ".join(prediction["tokens"][event_mention["trigger"]["start"]:event_mention["trigger"]["end"]])
                    for a in arguments:
                        a["text"] = " ".join(prediction["tokens"][a["start"]:a["end"]])
                
                event_mentions.append(event_mention)
                
            out = {
                "doc_id": prediction["doc_id"], 
                "wnd_id": prediction["wnd_id"], 
                "tokens": prediction["tokens"], 
                "event_mentions": event_mentions, 
            }
                
            if data:
                out["text"] = data[i]["text"]
                
            fp.write(json.dumps(out)+"\n")

import json

import json

def save_all_predictions(file, predictions, data, offset_map=None):
    assert len(predictions) == len(data)
    
    with open(file, 'w', encoding='utf-8') as fp:
        for i, prediction in enumerate(predictions):
            
            tokens = prediction["tokens"]
            raw_text = data[i].get("text", data[i].get("accident_report", ""))
            doc_id = prediction.get("doc_id", "Unknown")
            
            # 1. GENERATE OR USE OFFSET MAP
            if offset_map and i < len(offset_map):
                current_offset_map = offset_map[i]
            else:
                current_offset_map = generate_offset_map(tokens, raw_text)

            events = prediction.get("events", [{
                "trigger": prediction["trigger"],
                "arguments": prediction.get("arguments", [])
            }])

            event_mentions = []
            for event in events:
                arguments = []
                for a in event["arguments"]:
                    
                    # STANDARDIZE BOUNDS TO LISTS
                    starts = a[0] if isinstance(a[0], list) else [a[0]]
                    ends = a[1] if isinstance(a[1], list) else [a[1]]
                    role = a[2]
                    llm_text = a[3] if len(a) > 3 else ""

                    text_frags = []
                    # LOOP THROUGH EACH FRAGMENT INDEPENDENTLY
                    for s, e in zip(starts, ends):
                        if s == 0 and e == 0:  
                            text_frags.append(llm_text)
                            continue

                        # EXACT TEXT SLICE
                        if s < len(current_offset_map):
                            s_char = current_offset_map[s][0]
                            e_char = current_offset_map[min(e - 1, len(current_offset_map) - 1)][1]
                            sliced_raw_text = raw_text[s_char:e_char]
                            
                            # POST-MAPPING VERIFICATION (ARGUMENTS)
                            intended_tokens = "".join(tokens[s:e]).lower()
                            actual_slice = "".join(sliced_raw_text.split()).lower()
                            
                            if intended_tokens != actual_slice and intended_tokens not in actual_slice:
                                logger.warning(
                                    f"[{doc_id}] Offset Mismatch (Arg)!\n"
                                    f"  Tokens Expected: '{' '.join(tokens[s:e])}'\n"
                                    f"  Character Slice: '{sliced_raw_text}'"
                                )
                            
                            text_frags.append(sliced_raw_text)
                        else:
                            text_frags.append(llm_text)
                    
                    is_multi = len(starts) > 1
                    arguments.append({
                        "role": role,
                        "start": starts if is_multi else starts[0],
                        "end": ends if is_multi else ends[0],
                        "text": "; ".join(text_frags) if text_frags else llm_text
                    })
                
                # 3. TRIGGER PROCESSING
                t_s, t_e = event["trigger"][0], event["trigger"][1]
                t_s_char = current_offset_map[t_s][0]
                t_e_char = current_offset_map[min(t_e - 1, len(current_offset_map) - 1)][1]
                trigger_text = raw_text[t_s_char:t_e_char]

                # POST-MAPPING VERIFICATION (TRIGGER)
                intended_trigger = "".join(tokens[t_s:t_e]).lower()
                actual_trigger = "".join(trigger_text.split()).lower()
                
                if intended_trigger != actual_trigger and intended_trigger not in actual_trigger:
                    logger.warning(
                        f"[{doc_id}] Offset Mismatch (Trigger)!\n"
                        f"  Tokens Expected: '{' '.join(tokens[t_s:t_e])}'\n"
                        f"  Character Slice: '{trigger_text}'"
                    )

                event_mentions.append({
                    "event_type": event["trigger"][2],  
                    "trigger": {
                        "start": t_s,
                        "end": t_e,
                        "text": trigger_text
                    }, 
                    "arguments": arguments
                })
                
            out = {
                "doc_id": doc_id, 
                "wnd_id": prediction["wnd_id"], 
                "tokens": tokens, 
                "text": raw_text,
                "event_mentions": event_mentions, 
            }
            fp.write(json.dumps(out) + "\n")

def load_all_predictions(filepath, no_mod = True):
    if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return []

    # 1. Load Raw Data (Handle JSON List or JSON Lines)
    raw_data = []
    try:
        with open(filepath, 'r') as f:
            # Try standard JSON first (List of dicts)
            raw_data = json.load(f)
    except json.JSONDecodeError:
        # Fallback to JSON Lines (One dict per line)
        try:
            with open(filepath, 'r') as f:
                raw_data = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return []
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return []

    # 2. Convert Raw File Data -> Internal Prediction Format
    # Input to save_all_predictions expects: 
    # {'trigger': (start, end, type), 'arguments': [(start, end, role), ...], ...}
    if not no_mod:
        internal_preds = []
        for item in raw_data:
            # item is a document/window object from the file
            if "event_mentions" not in item: 
                continue
            
            for mention in item["event_mentions"]:
                # --- Reconstruct Trigger Tuple: (start, end, type) ---
                trig = mention.get("trigger", {})
                t_start = trig.get("start")
                t_end = trig.get("end")
                e_type = mention.get("event_type")
                
                if t_start is None or t_end is None: 
                    continue

                trigger_tuple = (t_start, t_end, e_type)
                
                # --- Reconstruct Argument Tuples: [(start, end, role), ...] ---
                args_tuples = []
                for arg in mention.get("arguments", []):
                    args_tuples.append((arg.get("start"), arg.get("end"), arg.get("role")))
                
                # --- Build Internal Object ---
                pred_obj = {
                    "doc_id": item.get("doc_id"),
                    "wnd_id": item.get("wnd_id"),
                    "tokens": item.get("tokens"),
                    "trigger": trigger_tuple,
                    "arguments": args_tuples
                }
                internal_preds.append(pred_obj)
    else:
        internal_preds = raw_data
            
    return internal_preds

def generate_offset_map(tokens, raw_text):
    """Safely maps tokens to character offsets."""
    offset_map = []
    current_char_idx = 0
    raw_text_lower = raw_text.lower()
    
    for token in tokens:
        start_char = raw_text_lower.find(token.lower(), current_char_idx)
        if start_char != -1:
            end_char = start_char + len(token)
            offset_map.append((start_char, end_char))
            current_char_idx = end_char 
        else:
            offset_map.append((current_char_idx, current_char_idx + len(token)))
            current_char_idx += len(token)
            
    return offset_map

def convert_AC_to_EAE(flat_data, structure, stanza_pipeline, mode="individual",
                      cls_factors=None, predict_acc_type=False):
    """Convert flat accident JSON to hierarchical EAE format.

    Uses the long-name schema (post-rename): `accident_report` is the doc-body
    field and structure root; `accident_type` is the top-level classification.

    mode="individual" — each fragment = one entity span (one fragment, one role assignment).
    mode="joint"      — all fragments merged into one bounding span (single entity per role).

    cls_factors — optional set of sub-keys whose values are classification labels
    rather than source spans (e.g. ``construction_trade``, ``severity``). For these,
    zero-width arguments are emitted with sentinel offsets (-1, -1) so the
    classification head has access to gold labels even when the label string isn't
    a substring of the report text.

    predict_acc_type — when True (E2* tasks), emit an extra Level-0 event whose
    trigger is ``accident_report`` and whose sole argument is ``accident_type``
    (sentinel offset, gold label). The TagPrime forward primed on this event
    runs the cls head against accident_type. When False (AC tasks where
    acc_type is given), the existing Level-1 event with trigger = acc_type
    carries that information implicitly; no extra event is needed.
    """
    cls_factors = set(cls_factors or ())
    main_factors = structure.get("accident_report", [])

    doc_id = str(flat_data.get("id", "UNKNOWN"))
    wnd_id = f"{doc_id}_wnd"

    # Extract text and overall accident type
    text = flat_data.get("accident_report", [""])[0]
    acc_type = flat_data.get("accident_type", ["Accident"])[0]
    acc_type = acc_type.lower().replace("-", "_").replace("/", "_")

    # 1. Tokenize text and map character spans to token indices
    tokens = []
    tok_spans = []
    _doc = stanza_pipeline(text)
    for sent in _doc.sentences:
        for tok in sent.tokens:
            tokens.append(tok.text)
            tok_spans.append((tok.start_char, tok.end_char))

    def get_token_range(search_text):
        """Finds the start and end token indices for a given string."""
        char_start = text.lower().find(search_text.lower())
        if char_start == -1:
            return 0, 0 # Fallback for classification labels not in text
            
        char_end = char_start + len(search_text)
        
        start_tok, end_tok = -1, -1
        for i, (s, e) in enumerate(tok_spans):
            if start_tok == -1 and (s >= char_start or e > char_start):
                start_tok = i
            if s < char_end:
                end_tok = i + 1
                
        if start_tok == -1: start_tok = 0
        if end_tok == -1 or end_tok <= start_tok: end_tok = start_tok
        return start_tok, end_tok

    entity_mentions = []
    event_mentions = []
    # entity_map: key -> list of {entity_id, text, start, end}
    entity_map = {}

    def _register_fragments(fragments, key=None):
        """Register entity mentions for a key's fragments according to mode.
        Returns list of {entity_id, text, start, end} dicts.

        For classification factors (``key in cls_factors``), zero-width spans
        are kept with sentinel offsets (-1, -1) so the label survives downstream.
        """
        spans = []
        is_cls = key in cls_factors
        if mode == "individual":
            # One entity per fragment (each fragment = one role span)
            for frag in fragments:
                s, e = get_token_range(frag)
                if s >= e:
                    if not is_cls:
                        continue
                    s, e = -1, -1
                ent_id = f"{wnd_id}_ent_{len(entity_mentions)}"
                entity_mentions.append({"id": ent_id, "entity_type": "UNK",
                                        "start": s, "end": e, "text": frag})
                spans.append({"entity_id": ent_id, "text": frag, "start": s, "end": e})
        else:  # joint — bounding span across all fragments
            all_s_e = [(get_token_range(f), f) for f in fragments]
            valid = [((s, e), f) for (s, e), f in all_s_e if s < e]
            if valid:
                s = min(se[0] for se, _ in valid)
                e = max(se[1] for se, _ in valid)
                merged_text = " ".join(tokens[s:e])
                ent_id = f"{wnd_id}_ent_{len(entity_mentions)}"
                entity_mentions.append({"id": ent_id, "entity_type": "UNK",
                                        "start": s, "end": e, "text": merged_text})
                spans.append({"entity_id": ent_id, "text": merged_text, "start": s, "end": e})
            elif is_cls and fragments:
                # Classification label that didn't anchor in source — keep the
                # first fragment's text with a sentinel span so the cls head
                # still sees the gold label.
                frag = fragments[0]
                ent_id = f"{wnd_id}_ent_{len(entity_mentions)}"
                entity_mentions.append({"id": ent_id, "entity_type": "UNK",
                                        "start": -1, "end": -1, "text": frag})
                spans.append({"entity_id": ent_id, "text": frag, "start": -1, "end": -1})
        return spans

    # 2. Extract entities for sub-factors (one or more per key depending on mode).
    all_sub_keys = [sub for mf in main_factors for sub in structure.get(mf, [])]
    for key in all_sub_keys:
        if key not in flat_data or not flat_data[key]:
            continue
        entity_map[key] = _register_fragments(flat_data[key], key=key)

    # 3. One event per main_factor; trigger = first fragment, args = all sub-factor spans.
    # Main_factors with no sub-factors (e.g. `object_involved`) are skipped: a
    # Level-2 event for them would have empty args, no BIO supervision (extraction
    # vocab filters cls factors anyway), and no cls supervision (no factor has them
    # as parent). The doc's oi spans still surface via the Level-1 acc_type event
    # (where main_factor spans are emitted as args).
    for mf in main_factors:
        if mf not in flat_data or not flat_data[mf]:
            continue
        if not structure.get(mf, []):
            continue
        mf_start, mf_end = get_token_range(flat_data[mf][0])
        if mf_start >= mf_end:
            continue

        sub_arguments = []
        for sub_key in structure.get(mf, []):
            for ent in entity_map.get(sub_key, []):
                sub_arguments.append({"entity_id": ent["entity_id"],
                                      "text": ent["text"], "role": sub_key})

        event_mentions.append({
            "id":         f"{wnd_id}_ev_{mf}",
            "event_type": mf,
            "trigger":    {"start": mf_start, "end": mf_end,
                           "text": " ".join(tokens[mf_start:mf_end])},
            "arguments":  sub_arguments,
        })

    # 4b. Top-level acc_type event: trigger = full doc, args = all MF spans.
    mf_arguments = []
    for mf in main_factors:
        if mf not in flat_data or not flat_data[mf]:
            continue
        for ent in _register_fragments(flat_data[mf]):
            mf_arguments.append({"entity_id": ent["entity_id"],
                                  "text": ent["text"], "role": mf})

    if mf_arguments:
        event_mentions.insert(0, {
            "id":         f"{wnd_id}_ev_acc_type",
            "event_type": acc_type,
            "trigger":    {"start": 0, "end": len(tokens), "text": text},
            "arguments":  mf_arguments,
        })

    # 4c. E2-task Level-0 event: trigger = "accident_report", sole argument is
    # `accident_type` with a sentinel-offset cls arg carrying the gold label.
    # Emitted only when the caller signals an E2 task (predict_acc_type=True);
    # under CR mode the TagPrime forward primed on this event runs the cls
    # head against `accident_type` (patterns[dataset]["accident_report"]
    # lists `accident_type` as its sole CR candidate).
    # AC tasks (accident_type given as input) skip this block — the existing
    # Level-1 event with trigger = acc_type carries the gold acc_type
    # implicitly via the event_type, no separate cls-arg needed.
    if predict_acc_type and flat_data.get("accident_type"):
        gold_at = flat_data["accident_type"][0] if isinstance(
            flat_data["accident_type"], list) else flat_data["accident_type"]
        if gold_at:
            at_ent_id = f"{wnd_id}_ent_{len(entity_mentions)}"
            entity_mentions.append({
                "id": at_ent_id, "entity_type": "UNK",
                "start": -1, "end": -1, "text": gold_at,
            })
            event_mentions.insert(0, {
                "id":         f"{wnd_id}_ev_accident_report",
                "event_type": "accident_report",
                "trigger":    {"start": 0, "end": len(tokens), "text": text},
                "arguments":  [{"entity_id": at_ent_id,
                                "text": gold_at, "role": "accident_type"}],
            })

    # 5. Assemble final payload
    return {
        "doc_id": doc_id,
        "wnd_id": wnd_id,
        "text": text,
        "tokens": tokens,
        "event_mentions": event_mentions,
        "entity_mentions": entity_mentions,
        "lang": "en"
    }