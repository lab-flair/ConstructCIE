#!/bin/bash

# --- ARGS ---
SOFT=0
LOCAL=0
USERNAME="$USER"
CONFIG_INDEX=0  # selects the global mapping that supplies scratch_path
                # (mirror of run.py's -M/--map / download_models_job.sh's -M):
                # 0=global_mapping.json (default), 1=global_mapping_1.json,
                # 2=global_mapping_2.json, 3=global_mapping_3.json
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--soft)   SOFT=1  ; shift ;;
        -l|--local)  LOCAL=1 ; shift ;;
        -u|--user)   USERNAME="$2" ; shift 2 ;;
        -M|--map)    CONFIG_INDEX="$2" ; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- CONFIGURATION ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

# ===========================================================
if [ "$LOCAL" -eq 1 ]; then
# =================== LOCAL PC PATH =========================

VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_VERSION="3.10"

echo "-----------------------------------------"
if [ "$SOFT" -eq 1 ]; then
    echo "Soft Reset [Local PC] — sync packages only"
else
    echo "Hard Reset [Local PC] — delete + recreate"
fi
echo "Target: $VENV_DIR"
echo "Requirements: $REQUIREMENTS"
echo "-----------------------------------------"

if [ "$SOFT" -eq 0 ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "Removing existing .venv..."
        rm -rf "$VENV_DIR"
    fi
    echo "Creating .venv (Python $PYTHON_VERSION)..."
    py -$PYTHON_VERSION -m venv "$VENV_DIR" \
        || python$PYTHON_VERSION -m venv "$VENV_DIR" \
        || python3 -m venv "$VENV_DIR" \
        || python -m venv "$VENV_DIR" \
        || { echo "ERROR: could not create venv. Is Python $PYTHON_VERSION installed?"; exit 1; }
else
    if [ ! -d "$VENV_DIR" ]; then
        echo ".venv not found at $VENV_DIR. Run without --soft to create it first."
        exit 1
    fi
    echo "Skipping recreation."
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: .venv was not created. Aborting."
    exit 1
fi

echo "Activating .venv..."
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: activate script not found in .venv"
    exit 1
fi

echo "Installing uv..."
pip install uv --quiet

if [ "$SOFT" -eq 1 ]; then
    echo "Syncing requirements..."
    uv pip sync "$REQUIREMENTS"
else
    echo "Installing requirements..."
    uv pip install -r "$REQUIREMENTS"
fi

else
# =================== SLURM / CLUSTER PATH ==================

ENV_NAME="textee3"
PYTHON_VERSION="3.10"

# Map -M index -> mapping filename (mirror of run.py's _CONFIG_MAP /
# download_models_job.sh's CONFIG_FILE dispatch).
case "$CONFIG_INDEX" in
    0) CONFIG_FILE="global_mapping.json" ;;
    1) CONFIG_FILE="global_mapping_1.json" ;;
    2) CONFIG_FILE="global_mapping_2.json" ;;
    3) CONFIG_FILE="global_mapping_3.json" ;;
    *) echo "ERROR: Invalid -M $CONFIG_INDEX (expected 0-3)"; exit 1 ;;
esac
CONFIG_PATH="$PROJECT_DIR/main/$CONFIG_FILE"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Mapping file not found: $CONFIG_PATH"
    exit 1
fi

# Pull scratch_path out of the chosen JSON. We use python3 rather than jq to
# avoid an extra cluster dependency. Mirrors download_models_job.sh's
# MAPPING_FIELDS dispatch: the new pointer format (global_mapping.json /
# global_mapping_1.json) carries a top-level "global" dict and nests scalar
# overrides like scratch_path under "args"; the old flat format (e.g.
# main/old_mapping.json) has no "global" key and carries those fields at
# the top level.
SCRATCH_TPL="$(python3 -c "
import json
with open('$CONFIG_PATH') as f:
    c = json.load(f)
args = c.get('args') if isinstance(c.get('global'), dict) else c
args = args or {}
print(args.get('scratch_path', '/scratch/user/userid/'))
")"

# Substitute the username placeholder in scratch_path (matches run.py's logic).
SCRATCH_DIR="${SCRATCH_TPL//userid/$USERNAME}"
SCRATCH_DIR="${SCRATCH_DIR%/}"

export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
export UV_CACHE_DIR="$SCRATCH_DIR/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME"

echo "-----------------------------------------"
if [ "$SOFT" -eq 1 ]; then
    echo "Soft Reset [SLURM Conda] — sync packages only"
else
    echo "Hard Reset [SLURM Conda] — delete + recreate"
fi
echo "Target: $ENV_NAME"
echo "Requirements: $REQUIREMENTS"
echo "Config: $CONFIG_FILE"
echo "Scratch dir: $SCRATCH_DIR"
echo "-----------------------------------------"

# Load cluster modules if the `module` command exists (some hosts, e.g. one
# of ours, have no module system at all - safe to skip since conda/gcc may
# already be on PATH). The conda module is only loaded as a last-resort
# fallback below, since $CONDA_EXE usually makes it unnecessary.
if command -v module &> /dev/null; then
    echo "Loading modules..."
    module load WebProxy
    module load GCC/11.3.0
else
    echo "No 'module' command found; skipping WebProxy/GCC loads."
fi

# --- Activate conda dynamically ---
# Different hosts install conda at different paths (e.g. /sw/eb/sw/Miniconda3
# vs ~/miniconda3) so we never hardcode one. $CONDA_EXE is exported by
# conda's own hook once `conda init` has run in .bashrc, and (unlike PATH
# state from `module load`) survives into non-interactive script
# invocations - so it works whether or not this host has a module system.
# Only fall back to `module`/hardcoded search paths if $CONDA_EXE is unset.
# Override via $CONDA_MODULE / $CONDA_ROOT.
if [ -n "$CONDA_EXE" ] && [ -x "$CONDA_EXE" ]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif command -v module &> /dev/null; then
    echo "No conda found; loading module (${CONDA_MODULE:-Miniconda3/23.10.0-1})..."
    module load "${CONDA_MODULE:-Miniconda3/23.10.0-1}"
    eval "$(conda shell.bash hook)"
elif [ -n "$CONDA_ROOT" ] && [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found. Set \$CONDA_ROOT (install prefix) or \$CONDA_MODULE, or ensure 'conda'/'module' is on PATH."
    exit 1
fi

if [ "$SOFT" -eq 0 ]; then
    if conda info --envs | grep -q "$ENV_NAME"; then
        echo "Deleting existing environment '$ENV_NAME'..."
        conda deactivate
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Environment '$ENV_NAME' not found. Skipping deletion."
    fi
    echo "Creating Conda environment (Python $PYTHON_VERSION)..."
    conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
else
    if ! conda info --envs | grep -q "$ENV_NAME"; then
        echo "Environment '$ENV_NAME' not found. Run without --soft to create it first."
        exit 1
    fi
    echo "Skipping deletion and recreation."
fi

echo "Activating Conda environment..."
conda activate "$ENV_NAME"

echo "Installing uv..."
pip install uv --no-cache-dir

if [ "$SOFT" -eq 1 ]; then
    echo "Syncing requirements..."
    uv pip sync --no-cache "$REQUIREMENTS"
    pip install uv --no-cache-dir
else
    echo "Installing requirements..."
    uv pip install --no-cache -r "$REQUIREMENTS"
fi

fi
# ===========================================================

echo "Done"
