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
- The job runs src/run_train.py

### Viewing A Training Run

```
|-- runs
|   -- train_2026-03-19_23-42
|       |-- config.json
|       |-- output.log
|       |-- plots
|       |   -- training_history.png
|       -- results.json
```
The above shows an example of ```runs/``` directory. Each job run creates 
- ```output.log``` - this is where console output is saved
- ```config.json``` - lists out all training and model parameters in ```json``` format
- ```plots/``` - directory with training loss and validation plots.
- ```results.json``` - saves accuracies, losses and f1 scores on training, dev and trial datasets.

Training config (learning rate, epochs, batch size, etc.) is in `src/train_config.py`.
Model config (transformer model, seed) is in `src/config.py`.

