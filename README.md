# NLI Binary Classification

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

---

## Running the LSTM demo notebook
1. Set up the virtual environment (instructions above).
2. Place the test set in `data/`.
3. Download the pre-trained model from **[here](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/zhijie_rong_student_manchester_ac_uk/IQAvXjxVpy5kQoAt6t9ovfjMAQY0HxVSJU1_vdAcb9J5Ao0?e=iEeUTf).**
4. Move the downloaded `lstm.pt` to `models/`.
5. Open `demo_lstm.ipynb` and run from the beginning. 

Predictions will be saved to `output/`.
Note that the first run will take longer as it requires downloading the GloVe vector.

**Existing predictions are stored in the root at Group_40_B.csv**.

---

## Attribution

### Data
- Training, development, and trial data provided as part of the COMP34812 coursework.

### LSTM Solution
- **Paper:** Chen, Q., Zhu, X., Ling, Z., Wei, S., Jiang, H., & Inkpen, D. (2017). *Enhanced LSTM for Natural Language Inference*. ACL 2017. https://arxiv.org/abs/1609.06038
- **Reference implementation:** https://github.com/coetaur0/ESIM — used as a reference for the cross-attention alignment mechanism and overall ESIM architecture. Code was not directly copied; the implementation was written independently based on the paper, with this repository consulted for architectural reference.

---

## Use of Generative AI Tools

**Tool used:** Claude

Claude was used for:
- **Writing this README** — structure, wording, and formatting.
- **Boilerplate code in `src/run_lstm.py`** — config pretty-printing block and JSON serialisation of config/results to run directory.

---
# Transformer-based Solution

Our solution performs parameter-efficient fine-tuning (PEFT) on a Small Language Model (SLM) Qwen-3.5-9B for Natural Language Inference (NLI) tasks using the LoRA (Low-Rank Adaptation) technique in bf16 precision. The workflow includes an optional hyperparameter optimization sweep using Optuna to find the best learning rate and LoRA rank before executing the main training run via the SFTTrainer. To ensure performance improvements, the script tracks validation accuracy and F1 scores through custom callbacks and compares the final fine-tuned model against a zero-shot baseline.
## Training a model 
1. Set up the virtual environment (instructions above).
2. Place the train and validation set in `data/`.
3. Submit a job `train_slm.job` on the csf3 cluster. It runs `run_slm.py`. The main training logic is in that script.
4. Under `runs/slm/`, you will see a new dir with your date and time e.g. `2026-03-30_23-12`
5. When the job is completed, you will see a `config.json` specifying the training configuration, a `results.json` with baseline and post-training accuracies on validation set and training losses, and `output.log`. 
6. The adapter weights are saved under `models/slm_adapter_{datetime}`

## Evaluating a model 
1. Submit a job `eval_slm.job` on the csf3 cluster. It runs `eval_slm.py`. The main evaluation logic is in that script.

## Running the SLM demo notebook
1. Set up the virtual environment (instructions above).
2. Place the test set in `data/`.
3. Download the fine-tuned adapter weights from **[here](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/kwan_yip_student_manchester_ac_uk/IgAnDF5qgZOkSIApFvOgP8ETATsxu3piY1hLpkpO70pb71o?e=Bl3M3K).**
4. Move the downloaded `slm_adapter_demo` to `models/`.
5. Open `demo_slm.ipynb` and run from the beginning. 



Predictions will be saved to `output/`.

**Existing predictions are stored in the root at Group_40_C.csv**.

---

## Attribution

### Data
- Training, development, and trial data provided as part of the COMP34812 coursework.

### SLM Solution
- **Paper:** Chen, Q., Zhu, X., Ling, Z., Wei, S., Jiang, H., & Inkpen, D. (2017). *Enhanced LSTM for Natural Language Inference*. ACL 2017. https://arxiv.org/abs/1609.06038
- **Reference implementation:** https://github.com/coetaur0/ESIM — used as a reference for the cross-attention alignment mechanism and overall ESIM architecture. Code was not directly copied; the implementation was written independently based on the paper, with this repository consulted for architectural reference.