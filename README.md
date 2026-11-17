# ConstructCIE

**A Dataset for Extracting Causal Information from Construction Accident Narratives**

> Accepted to **Findings of the Association for Computational Linguistics: EMNLP 2026**

[**Datasets**](#supported-datasets) |
[**Settings**](#supported-settings) |
[**Models**](#supported-models) |
[**Setup**](#setup) |
[**Usage**](#usage) |
[**Project Structure**](#project-structure) |
[**Paper**](https://aclanthology.org/2026.findings-emnlp.537/) 

**Authors**: [Hung Nguyen](https://www.linkedin.com/in/hung-nguyen-3b63191a0/), [Jaehoon Lee](https://jaehoonblog.com/), [Namgyun Kim](https://www.arch.tamu.edu/directory/namgyun-kim/), [Kuan-Hao Huang](https://khhuang.me/)

---

## Introduction

ConstructCIE is a benchmark and codebase for causal information extraction from construction accident reports.
- 530 expert-annotated OSHA accident reports with hierarchical causal annotations.
- Support for accident types, causal factors, sub-causal factors, and long or discontinuous evidence spans.
- Standardized evaluation for accident-conditioned and end-to-end extraction settings.
- Implementations of supervised baselines and instruction-tuned LLM approaches.

---

## Supported Datasets

| Dataset Name   | Short Name     | Supported Tasks | Link |
|----------------|----------------|-----------------|-----------------|
| `ConstructCIE` | `constructcie` | JHE, IHE        | https://huggingface.co/datasets/lab-flair/constructcie |

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

## Citation
```bib
@inproceedings{nguyen-etal-2026-constructcie,
    title = "{C}onstruct{CIE}: A Dataset for Extracting Causal Information from Construction Accident Narratives",
    author = "Nguyen, Hung  and
      Lee, Jaehoon  and
      Kim, Namgyun  and
      Huang, Kuan-Hao",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.537/",
    doi = "10.18653/2026.findings-emnlp.537",
    pages = "11284--11302",
    ISBN = "979-8-89176-462-0",
    abstract = "Construction accident narratives contain rich causal information, but the evidence is often implicit, long-span, and distributed. We introduce ConstructCIE, a manually annotated dataset for Causal Information Extraction from OSHA construction accident reports. The dataset uses a hierarchical schema for accident types, causal factors, sub-causal factors, and supporting evidence spans. We evaluate supervised sequence taggers and instruction-tuned LLMs in an end-to-end hierarchical extraction setting. Results show that most evaluated models achieve strong accident-type prediction and recover broad causal meaning but remain limited in precise span-level extraction. Joint Hierarchical Extraction generally achieves stronger exact and soft matching, while Individual Hierarchical Extraction sometimes achieves higher keyword F1. Error distributions vary by extraction strategy, but evidence-selection and span-boundary errors remain common. These findings show that reliable Causal Information Extraction for construction accidents requires stronger domain grounding and more accurate evidence extraction. The code and data can be found at https://github.com/lab-flair/ConstructCIE."
}
```
