# ConstructCIE

**A Dataset for Extracting Causal Information from Construction Accident Narratives**

[**Datasets**](#supported-datasets) |
[**Settings**](#supported-settings) |
[**Models**](#supported-models) |
[**Setup**](#setup) |
[**Usage**](#usage) |
[**Project Structure**](#project-structure)

---

## Supported Datasets

| Dataset Name   | Short Name     | Supported Tasks |
|----------------|----------------|-----------------|
| `ConstructCIE` | `constructcie` | JHE, IHE        |

## Supported Settings

| Setting Name                                    | Short Name |
|-------------------------------------------------|------------|
| End-to-End Joint Hierarchical Extraction        | `JHE`      |
| End-to-End Individual Hierarchical Extraction   | `IHE`      |

## Supported Models

| Model Name                     | Short Name     | Supported Tasks |
|--------------------------------|----------------|-----------------|
| `TagPrime-C`                   | `TagPrime-C`   | JHE             |
| `TagPrime-CR`                  | `TagPrime-CR`  | JHE             |
| `zephyr-7b-alpha`              | `Zephyr-7B`    | JHE, IHE        |
| `Mixtral-8x7B-Instruct-v0.1`   | `Mixtral-8x7B` | JHE, IHE        |
| `Qwen3.5-9B`                   | `Qwen3.5-9B`   | JHE, IHE        |
| `Qwen3.5-27B`                  | `Qwen3.5-27B`  | JHE, IHE        |
| `gemma-4-E4B-it`               | `gemma4-E4B`   | JHE, IHE        |
| `gemma-4-26B-A4B-it`           | `gemma4-26B`   | JHE, IHE        |
| `Llama-3.2-3B-Instruct`        | `Llama3.2-3B`  | JHE, IHE        |
| `Llama-3.2-11B-Vision-Instruct`| `Llama3.2-11B` | JHE, IHE        |
| `Llama-3.1-70B-Instruct`       | `Llama3.1-70B` | JHE, IHE        |

## Setup

### 1. Configure the scratch path

The **scratch path** is where all large files live — downloaded models, trained checkpoints, Stanza resources — typically `/scratch` on a cluster, or any folder with enough disk space locally.

Create a `/scratch/` folder in the root directory, **or** point the `scratch_path` key in [main/global_mapping.json](main/global_mapping.json) to your preferred location.

### 2. Install dependencies

Using **conda + uv**:

```bash
conda create -n textee3 python=3.10 -y && conda run -n textee3 python -m pip install uv && conda run -n textee3 uv pip install -r requirements.txt
```

Or using **uv** only:

```bash
uv venv .venv --python 3.10 && uv pip install --python .venv -r requirements.txt
```

Or, on a **Slurm cluster** (with write access to your scratch path):

```bash
bash scripts/reset_env.sh
```

### 3. Download models

Download HuggingFace models to:

```
<scratch_path>/models/
```

On the cluster, you can download a model with:

```bash
bash scripts/download_models_job.sh [model_id]
```

## Usage

All experiments are driven by `run.py`, which generates (or directly runs) jobs for every combination of task × dataset × model you select.

```bash
python run.py -a [action] -t [task] -d [dataset_short_name] -m [model_short_name] [options]
```

### Actions (`-a`)

| Action     | What it does                                                                       |
|------------|------------------------------------------------------------------------------------|
| `generate` | (Default) Create Slurm scripts under `jobs/` for the selected combinations          |
| `run`      | Run the selected combinations directly, no Slurm needed ([details](#alternative-run-directly-without-slurm--a-run)) |
| `check`    | Report progress/stored scores; add `--aggregate` for a cross-model F1 comparison    |
| `error`    | Run error analysis and write `errors_*.json` files                                  |
| `analysis` | Pool predictions + golds and emit task-level, few-shot scaling, and heatmap tables  |

### Common options

| Option               | Description                                                                  |
|----------------------|------------------------------------------------------------------------------|
| `-t, --task`         | Task(s), e.g. `JHE` `IHE`                                                     |
| `-d, --dataset`      | Dataset short name(s), e.g. `constructcie`                                    |
| `-m, --model`        | Model short name(s), e.g. `TagPrime-C`, `Qwen3.5-11B`                         |
| `-C, --config`       | Config variant(s), e.g. `-C 2` → `c2`                                         |
| `-F, --few-shot`     | Few-shot size(s) to run                                                       |
| `-g, --gpus`         | Override GPU mapping for every combination                                    |
| `--retrain`          | Force retraining even if a checkpoint exists (also re-predicts)               |
| `--repredict`        | Force re-prediction even if a prediction file exists                          |
| `--dry-run`          | Show what would happen without writing/running anything                       |

Omitting `-t`, `-d`, or `-m` selects **all** tasks, datasets, or models from the config.

### Training supervised models

1. Generate the Slurm scripts:

   ```bash
   python run.py -t [task] -m [model_short_name] -d [dataset_short_name] -C [config_variant] --retrain
   ```

2. Submit the job:

   ```bash
   sbatch jobs/[task]_[dataset_short_name]_[model_short_name]_[config_variant].sbatch
   ```

### Evaluating models (LLMs and supervised) on Slurm

1. Generate the Slurm scripts:

   ```bash
   python run.py -t [task] -m [model_short_name] -d [dataset_short_name] -C [config_variant] --repredict
   ```

2. Submit the job:

   - **LLMs:**

     ```bash
     bash jobs/[task]_[dataset_short_name]_[model_short_name].sh
     ```

   - **Supervised models:**

     ```bash
     sbatch jobs/[task]_[dataset_short_name]_[model_short_name]_[config_variant].sbatch
     ```

### Alternative: run directly without Slurm (`-a run`)

If you are on an interactive GPU node or a machine without `sbatch`, skip the job scripts entirely and execute the same commands directly. `-a run` covers both workflows — training and prediction — controlled by the same flags:

- **Train** a supervised model:

  ```bash
  python run.py -a run -t [task] -m [model_short_name] -d [dataset_short_name] -C [config_variant] --retrain
  ```

- **Predict / evaluate** (LLMs and supervised models):

  ```bash
  python run.py -a run -t [task] -m [model_short_name] -d [dataset_short_name] -C [config_variant] --repredict
  ```

## Project Structure

```
├── run.py            # Main entry point: generates/runs jobs, checks progress, analysis
├── TextEE/           # Core library: models, train/predict/evaluate scripts, scorer
├── main/             # Global configs (global_mapping*.json), job & progress management
├── data/             # Raw data, processed data, and the data processor
├── scripts/          # Cluster helper scripts (env setup, model download, training)
├── jobs/             # Generated Slurm/shell job scripts
├── results/          # Model outputs and evaluation results
└── logs/             # Job logs
```
