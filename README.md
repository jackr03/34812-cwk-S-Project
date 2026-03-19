# NLI Binary Classification

Binary Natural Language Inference classifier using DeBERTa v3.
Given a premise and hypothesis, predicts entailment (1) or non-entailment (0).

## Setup

### Create and Activate .venv

```bash
python -m venv .venv
source .venv/bin/activate
```

### Mac (MPS / CPU)
```bash
pip install -r requirements-mac.txt
```

### HPC (SLURM + NVIDIA CUDA on Linux)
```bash
pip install -r requirements-hpc.txt
```

### If you already created a .venv
You can run the ```setup.sh``` script to activate the ```.venv``` and setup SSH for GitHub pushes.

## Training

### Submit a Training Job
- Change the ```#SBATCH --partition=gpuL``` for L40s or ```#SBATCH --partition=gpuL``` for A100s
```bash
sbatch jobs/train.job
```
- The job runs src/run_train.py and creates training logs, configs, results and plots under a time-labelled runs directory



Training config (learning rate, epochs, batch size, etc.) is in `src/train_config.py`.
Model config (transformer model, seed) is in `src/config.py`.

```
|-- checkpoints
|   `-- best_model.pt
|-- COMP34812 Coursework AY2025-26-3.pdf
|-- data
|   |-- dev.csv
|   |-- gemini.csv
|   |-- NLI_trial.csv
|   `-- train.csv
|-- evaluate_gemini.py
|-- hyperparameters
|   `-- transformer.json
|-- jobs
|   `-- train.job
|-- logs
|   |-- train_2026-03-19_22-36-18.log
|   |-- train_2026-03-19_22-51-53.log
|   `-- train_2026-03-19_22-59-27.log
|-- models
|   `-- transformer.pt
|-- notebooks
|   |-- exploration.ipynb
|   `-- train_transformer_model.ipynb
|-- README.md
|-- requirements
|   |-- requirements-hpc.txt
|   |-- requirements-mac.txt
|   `-- requirements.txt
|-- runs
|   `-- train_2026-03-19_23-42
|       |-- config.json
|       |-- output.log
|       |-- plots
|       |   `-- training_history.png
|       `-- results.json
|-- setup.sh
`-- src
    |-- config.py
    |-- data
    |   |-- dataset.py
    |   `-- __init__.py
    |-- __init__.py
    |-- models
    |   |-- __init__.py
    |   |-- nli_dataset.py
    |   `-- transformer_classifier.py
    |-- run_train.py
    `-- utils.py
```