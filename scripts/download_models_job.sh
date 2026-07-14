#!/bin/bash

# Usage: bash download_models_job.sh [-M <0|1|2|3>] [--local] [--user <name>] [--model <name1> <name2> ...] [index1 index2 ...]
# -M, --map     selects the global mapping that supplies model_dir / scratch_path
#               (mirror of run.py's -M/--map / reset_env.sh's -M): 0=global_mapping.json
#               (default), 1=global_mapping_1.json, 2=global_mapping_2.json,
#               3=global_mapping_3.json. All live under main/ in the new
#               pointer format (top-level "global"/"args"/"configs" keys).
# --local       skips cluster-only setup (module load WebProxy, conda activate
#               textee3) AND downloads into <repo>/local/models regardless of
#               the chosen mapping's model_dir/scratch_path. Assumes the caller
#               already activated their env.
# --user        explicit username override (string). Default: $USER (env var).
#               Ignored when the resolved BASE_DIR doesn't reference scratch.
# --token       HuggingFace access token, needed for gated repos (e.g. the
#               meta-llama/* models in MODELS below). Falls back to $HF_TOKEN /
#               $HUGGING_FACE_HUB_TOKEN, then to whatever `hf auth login`
#               already cached. Get a token from https://huggingface.co/settings/tokens.
# --model       accepts model basenames (e.g. gemma-4-E4B-it). Values are read until the
#               next --flag or end of args. Positional integer args are MODELS-array indices.

MODELS=(
    "meta-llama/Llama-3.2-11B-Vision-Instruct"   # Index 0  (LLM)
    "meta-llama/Llama-3.2-90B-Vision-Instruct"   # Index 1  (LLM)
    "HuggingFaceH4/zephyr-7b-alpha"              # Index 2  (LLM)
    "mistralai/Mixtral-8x7B-Instruct-v0.1"       # Index 3  (LLM)
    "Qwen/Qwen3.5-9B"                            # Index 4  (LLM)
    "roberta-large"                              # Index 5  (backbone: roberta-based models)
    "facebook/bart-large"                        # Index 6  (backbone: bart-based models)
    "google/gemma-4-E4B-it"                      # Index 7  (LLM)
    "google/gemma-4-26B-A4B-it"                  # Index 8  (LLM)
    "google/gemma-4-31B-it"                      # Index 9  (LLM)
    "Qwen/Qwen3.5-27B"                           # Index 10  (LLM)
    "facebook/mbart-large-50"                    # Index 11  (backbone: bart-based models)
    "meta-llama/Llama-3.2-3B-Instruct"           # Index 12  (LLM)
    "meta-llama/Llama-3.1-70B-Instruct"          # Index 13  (LLM)
)

# --- Parse flags ---
USER_ARG=""
TOKEN_ARG=""
LOCAL_MODE=0
CONFIG_INDEX=0  # default mirrors run.py's -M/--map default: global_mapping.json (override with -M)
MODEL_NAMES=()
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            LOCAL_MODE=1
            shift
            ;;
        -M|--map)
            shift
            CONFIG_INDEX="$1"
            shift
            ;;
        --user)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                USER_ARG="$1"
                shift
            done
            ;;
        --token)
            shift
            TOKEN_ARG="$1"
            shift
            ;;
        --model)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODEL_NAMES+=("$1")
                shift
            done
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${REMAINING_ARGS[@]}"

# Resolve --user: string override. Default: $USER env var.
USERNAME="${USER_ARG:-$USER}"

# Resolve --token: explicit flag > $HF_TOKEN > $HUGGING_FACE_HUB_TOKEN (both
# recognized by huggingface_hub) > whatever `hf auth login` already cached
# locally (~/.cache/huggingface/token). Only the first three are passed
# explicitly via --token; a cached login is picked up by `hf` on its own.
HF_TOKEN_RESOLVED="${TOKEN_ARG:-${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}}"
if [ -n "$HF_TOKEN_RESOLVED" ]; then
    echo "🔑 Using HuggingFace token (${HF_TOKEN_RESOLVED:0:6}...) for gated repos."
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "🔑 No --token/HF_TOKEN given; using cached \`hf auth login\` session."
else
    echo "⚠️  No HuggingFace token found (--token / \$HF_TOKEN / \`hf auth login\`)."
    echo "   Gated repos (e.g. meta-llama/*) will fail to download without one."
fi

# --- CONFIGURATION ---
# Resolve project root from this script's location (scripts/..) so relative
# paths in the chosen global mapping (e.g. "local/models/") anchor here.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Map -M index → mapping filename (mirror of run.py _CONFIG_MAP).
case "$CONFIG_INDEX" in
    0) CONFIG_FILE="global_mapping.json" ;;
    1) CONFIG_FILE="global_mapping_1.json" ;;
    2) CONFIG_FILE="global_mapping_2.json" ;;
    3) CONFIG_FILE="global_mapping_3.json" ;;
    *) echo "❌ Invalid -M $CONFIG_INDEX (expected 0-3)"; exit 1 ;;
esac
CONFIG_PATH="${PROJECT_ROOT}/main/${CONFIG_FILE}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Mapping file not found: $CONFIG_PATH"
    exit 1
fi
echo "📄 Using config: $CONFIG_FILE"

# Pull model_dir + scratch_path out of the chosen JSON. We use python3 rather
# than jq to avoid an extra cluster dependency — python3 is already required
# by the rest of the pipeline.
#
# Mirrors run.py's load_global_config dispatch: the new pointer format
# (global_mapping.json / global_mapping_1.json) carries a top-level "global"
# dict and nests scalar overrides like model_dir/scratch_path under "args";
# the old flat format (e.g. main/old_mapping.json) has no "global" key and
# carries those fields at the top level.
MAPPING_FIELDS="$(python3 -c "
import json, sys
with open('$CONFIG_PATH') as f:
    c = json.load(f)
args = c.get('args') if isinstance(c.get('global'), dict) else c
args = args or {}
print(args.get('model_dir', ''))
print(args.get('scratch_path', ''))
")"
MODEL_DIR_RAW="$(echo "$MAPPING_FIELDS" | sed -n '1p')"
SCRATCH_TPL="$(echo "$MAPPING_FIELDS" | sed -n '2p')"

# Substitute the username placeholder in scratch_path (matches run.py's logic).
SCRATCH_PATH="${SCRATCH_TPL//userid/$USERNAME}"
SCRATCH_PATH="${SCRATCH_PATH%/}"

# Resolve model_dir → BASE_DIR:
#   "/scratch_path/..."  → anchor to resolved SCRATCH_PATH
#   absolute path        → use as-is
#   relative path        → anchor to PROJECT_ROOT
if [[ "$MODEL_DIR_RAW" == /scratch_path* ]]; then
    BASE_DIR="${SCRATCH_PATH}${MODEL_DIR_RAW#/scratch_path}"
elif [[ "$MODEL_DIR_RAW" == /* ]]; then
    BASE_DIR="$MODEL_DIR_RAW"
elif [ -n "$MODEL_DIR_RAW" ]; then
    BASE_DIR="${PROJECT_ROOT}/${MODEL_DIR_RAW}"
else
    BASE_DIR="${PROJECT_ROOT}/local/models"
fi
BASE_DIR="${BASE_DIR%/}"

# --local always downloads into the repo's local models folder, regardless of
# the chosen mapping's model_dir/scratch_path (which normally point at cluster
# scratch space that doesn't exist off-cluster).
if [ "$LOCAL_MODE" -eq 1 ]; then
    BASE_DIR="${PROJECT_ROOT}/local/models"
fi

if [[ "$BASE_DIR" == /scratch/* ]]; then
    echo "👤 User: $USERNAME"
fi
echo "📂 Model dir: $BASE_DIR"

# --- PREPARATION ---
echo "-----------------------------------------"
echo "🚀 Starting Model Download"
echo "📅 Date: $(date)"
echo "💻 Host: $(hostname)"
echo "-----------------------------------------"

# 1. Check and Create Directory
if [ -d "$BASE_DIR" ]; then
    echo "📂 Base Directory exists: $BASE_DIR"
else
    echo "📂 Directory not found. Creating: $BASE_DIR"
    mkdir -p "$BASE_DIR"
fi

# 2. Determine which models to process
TARGET_INDICES=()

# Resolve --model values → indices. Numeric value = direct index; string = basename match.
for name in "${MODEL_NAMES[@]}"; do
    if [[ "$name" =~ ^[0-9]+$ ]]; then
        if [ -n "${MODELS[$name]}" ]; then
            TARGET_INDICES+=("$name")
        else
            echo "⚠️  --model index $name out of range; skipping."
        fi
        continue
    fi
    found=""
    for idx in "${!MODELS[@]}"; do
        base="${MODELS[$idx]##*/}"
        if [[ "$base" == "$name" || "${MODELS[$idx]}" == "$name" ]]; then
            TARGET_INDICES+=("$idx")
            found="1"
            break
        fi
    done
    if [ -z "$found" ]; then
        echo "⚠️  --model '$name' not found in MODELS array; skipping."
    fi
done

# Append any positional integer args as raw indices
for arg in "$@"; do
    TARGET_INDICES+=("$arg")
done

# If user explicitly passed --model or positional args but none resolved, abort
# rather than silently downloading every model.
if [ ${#TARGET_INDICES[@]} -eq 0 ]; then
    if [ ${#MODEL_NAMES[@]} -gt 0 ] || [ $# -gt 0 ]; then
        echo "❌ No valid models matched the given --model / index args. Aborting."
        exit 1
    fi
    echo "ℹ️  No --model or indices provided. Downloading ALL models."
    TARGET_INDICES=("${!MODELS[@]}")
else
    echo "ℹ️  Selected indices: ${TARGET_INDICES[*]}"
fi

echo "-----------------------------------------"

# Cluster-only setup: skipped in --local mode (assumes the caller has already
# activated whatever environment they want to download into).
if [ "$LOCAL_MODE" -eq 0 ]; then
    # Load Proxy (Common on clusters); skipped on hosts with no module system.
    if command -v module &> /dev/null; then
        echo "🌐 Loading WebProxy..."
        module load WebProxy
    fi

    # Activate conda dynamically (mirrors reset_env.sh's activation logic).
    # Different hosts install conda at different paths (e.g. /sw/eb/sw/Miniconda3
    # vs ~/miniconda3), so we never hardcode one. $CONDA_EXE is exported by
    # conda's own hook once `conda init` has run in .bashrc and survives into
    # non-interactive script invocations regardless of whether a module system
    # exists. Override via $CONDA_MODULE / $CONDA_ROOT.
    if [ -n "$CONDA_EXE" ] && [ -x "$CONDA_EXE" ]; then
        eval "$("$CONDA_EXE" shell.bash hook)"
    elif command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    elif command -v module &> /dev/null; then
        echo "🌐 No conda found; loading module (${CONDA_MODULE:-Miniconda3/23.10.0-1})..."
        module load "${CONDA_MODULE:-Miniconda3/23.10.0-1}"
        eval "$(conda shell.bash hook)"
    elif [ -n "$CONDA_ROOT" ] && [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
    else
        echo "❌ conda not found. Set \$CONDA_ROOT (install prefix) or \$CONDA_MODULE, or ensure 'conda'/'module' is on PATH."
        exit 1
    fi
    conda activate textee3
fi

# --- DOWNLOAD LOOP ---
for i in "${TARGET_INDICES[@]}"; do
    
    # Check if input is a valid number
    if ! [[ "$i" =~ ^[0-9]+$ ]]; then
        echo "⚠️  Skipping invalid input: '$i' (Not a number)"
        continue
    fi

    # Check if index exists in the array
    if [ -z "${MODELS[$i]}" ]; then
        echo "⚠️  Skipping Index $i: No model defined at this index."
        continue
    fi

    model_id="${MODELS[$i]}"
    model_name="${model_id##*/}"
    target_folder="$BASE_DIR/$model_name"

    echo "-----------------------------------------"
    echo "Processing Index [$i]"
    echo "⬇️  Source:      $model_id"
    echo "📂 Destination: $target_folder"
    echo "-----------------------------------------"

    # Download using --local-dir
    # Note: Ensure 'hf' (huggingface-cli) is installed in your environment
    if [ -n "$HF_TOKEN_RESOLVED" ]; then
        hf download "$model_id" \
            --local-dir "$target_folder" \
            --token "$HF_TOKEN_RESOLVED"
    else
        hf download "$model_id" \
            --local-dir "$target_folder"
    fi

    if [ $? -eq 0 ]; then
        echo "✅ Success: $model_name is ready."
    else
        echo "❌ Failed: $model_name"
    fi
done

echo "-----------------------------------------"
echo "🎉 Job finished."
echo "-----------------------------------------"