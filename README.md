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

**The main training code is in `src/run_slm.py`**

## Evaluating a model 
1. Submit a job `eval_slm.job` on the csf3 cluster. It runs `eval_slm.py`. The main evaluation logic is in that script.

**The main evaluation code is in `src/slm_utils/eval_slm.py`**

## Running the SLM demo notebook
1. Set up the virtual environment (instructions above).
2. Place the test set in `data/`.
3. Download the fine-tuned adapter weights from **[here](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/kwan_yip_student_manchester_ac_uk/IgAnDF5qgZOkSIApFvOgP8ETATsxu3piY1hLpkpO70pb71o?e=Bl3M3K).**
4. Move the downloaded `slm_adapter_demo` to `models/`.
5. Open `demo_slm.ipynb` and run from the beginning. 

**The demo notebook is in `notebooks/demo_slm.ipynb`**

Predictions will be saved to `output/`.

**Existing predictions are stored in the root at Group_40_C.csv**.

---

## Attribution

### Data
- Training, development, and trial data provided as part of the COMP34812 coursework.

### SLM Solution
- **Paper:** Bai, J., Bai, S., Yang, Y., et al. (2023). *Qwen Technical Report*. https://arxiv.org/abs/2309.16609  
- **Paper:** Hu, E. J., Shen, Y., Wallis, P., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685  
- **Paper:** Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). *A Large Annotated Corpus for Learning Natural Language Inference (SNLI)*. https://arxiv.org/abs/1508.05326  
- **Paper:** Williams, A., Nangia, N., & Bowman, S. R. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MultiNLI)*. https://arxiv.org/abs/1704.05426  
- **Paper:** Brown, T. B., Mann, B., Ryder, N., et al. (2020). *Language Models are Few-Shot Learners*. https://arxiv.org/abs/2005.14165  
- **Paper:** Wei, J., Bosma, M., Zhao, V. Y., et al. (2021). *Finetuned Language Models Are Zero-Shot Learners*. https://arxiv.org/abs/2109.01652  

- **Reference implementation:** https://www.datacamp.com/tutorial/fine-tuning-qwen3-5-small — used as a practical guide for supervised fine-tuning (SFT) of Qwen 3.5 models, including dataset formatting, training pipeline design, and parameter-efficient adaptation using LoRA. The implementation in this project was written independently, but follows a similar high-level fine-tuning workflow.

- **Reference implementation:** https://github.com/huggingface/trl — used as a reference for the `SFTTrainer` training framework, including supervised fine-tuning abstractions, dataset handling, and integration with Hugging Face Transformers. No code was directly copied; the library was used as intended for training.

- **Methodology note:** The model is fine-tuned as a causal language model for Natural Language Inference (NLI), where premise–hypothesis pairs are formatted as instruction-style inputs and the model generates entailment labels. This follows the paradigm of instruction tuning as described in Brown et al. (2020) and FLAN-style fine-tuning (Wei et al., 2021), adapting generative LLMs for classification tasks.

## Use of Generative AI Tools

**Tool used:** Claude

Claude was used for:
- **Writing this README** — structure, wording, and formatting.
- **Writing the Model Card** - structure, wording, formatting, summarising and referencing logs and training/evaluation results.
- **Boilerplate code in `src/run_slm.py`** — config pretty-printing block and JSON serialisation of config/results to run directory.
- **Boilerplate code in job scripts** — config pretty-printing block and creating logs and results with date-appropriate naming to run directory.