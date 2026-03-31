# Model Card: Qwen3.5-9B-Base NLI Classifier (LoRA)

---

## Model Details

| Field | Value |
|---|---|
| **Model name** | Qwen3.5-9B-Base NLI Classifier |
| **Base model** | `Qwen/Qwen3.5-9B-Base` |
| **Total parameters** | 8,953,803,264 (~9B) |
| **Model type** | Causal language model fine-tuned for binary classification |
| **Fine-tuning method** | LoRA (Low-Rank Adaptation) via PEFT |
| **Training framework** | Hugging Face TRL `SFTTrainer` |
| **Task** | Binary Natural Language Inference (NLI) |
| **Labels** | `0` = Non-entailment, `1` = Entailment |
| **Precision** | bfloat16 |
| **Version** | 1.0 |
| **Date** | 2026-03-31 |
| **License** | For academic/coursework use only |

This model is a `Qwen/Qwen3.5-9B-Base` causal language model fine-tuned for binary Natural Language Inference using instruction-style supervised fine-tuning (SFT) with LoRA adapters. Given a premise and hypothesis, the model predicts whether the premise **entails** the hypothesis (`1`) or not (`0`).

### Training Approach

The model was fine-tuned using the instruction-tuning paradigm: premise-hypothesis pairs are formatted as a structured chat prompt, and the model is trained to generate the correct label token (`"0"` or `"1"`) as a single-token completion. LoRA adapters are applied to 7 projection layers (attention and feed-forward), keeping the base model weights frozen.

**Prompt format** — using Qwen3's chat template with `enable_thinking=False`:

- **System:** `"You are a binary Natural Language Inference classifier. Given a premise and a hypothesis, predict whether the premise entails the hypothesis."`
- **User:**
  ```
  Premise:
  {premise}

  Hypothesis:
  {hypothesis}

  Return ONLY the number of the correct label.

  0 = Non-entailment
  1 = Entailment

  Answer:
  ```
- **Assistant (training target):** `"0"` or `"1"` (single token)

At inference time, a `_LabelOnlyLogitsProcessor` masks all vocabulary tokens except `"0"` and `"1"` to `-inf`, ensuring the model always outputs a valid binary label.

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Dropout | 0.0 |
| Bias | none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Task type | `CAUSAL_LM` |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 1 |
| Per-device batch size | 16 |
| Gradient accumulation steps | 4 |
| Effective batch size | 64 |
| Learning rate | 1e-4 |
| LR scheduler | Cosine |
| Max sequence length | 256 tokens |
| Seed | 100 |

An optional Optuna hyperparameter sweep (8 trials) is available, searching over learning rate (1e-5 to 5e-4), LoRA rank ([8, 16, 32]), LoRA alpha ([16, 32, 64]), and dropout (0.0 to 0.15).

### Hardware

| Field | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| VRAM | 79.25 GB |
| Training time | ~57 minutes (3,433 s) |

### Resources

- Training code: [`src/run_slm.py`](src/run_slm.py)
- Evaluation code: [`src/slm_utils/eval_slm.py`](src/slm_utils/eval_slm.py)
- Demo notebook: [`demo_slm.ipynb`](demo_slm.ipynb)
- Mitchell, M. et al. *Model Cards for Model Reporting*. FAccT 2019.

---

## Intended Use

**Primary intended use:** Binary NLI classification — determining whether a given premise entails a given hypothesis in English.

**Primary intended users:** Researchers and students working on NLI tasks in academic settings.

**In scope:**
- English-language premise-hypothesis pairs in academic or general domains
- Inputs formatted using the prompt template described in Model Details

**Out-of-scope use cases:**
- Multi-class NLI (e.g., entailment / neutral / contradiction)
- Non-English text
- Any NLP task other than binary NLI
- High-stakes decision-making without human review
- Raw premise-hypothesis pairs provided without the expected prompt structure

---

## Factors

Factors that may affect model performance:

### Group Factors
- **Text domain:** The training data reflects academic NLI corpora (SNLI/MultiNLI-style). Performance may vary for informal, conversational, or domain-specific text (e.g., legal, medical).
- **Language variety:** Fine-tuning was performed exclusively on English text. Performance on other languages or non-standard English is not evaluated.

### Instrumentation Factors
- **Prompt format:** The model was trained with a specific system/user prompt structure. Deviations from this format — such as omitting the system message or reordering fields — may degrade outputs.
- **Tokenisation:** Labels are predicted via constrained decoding over the tokens `"0"` and `"1"`. Behaviour is sensitive to how the tokeniser encodes these characters.

### Environment Factors
- **Sequence length:** The model was trained with a maximum sequence length of 256 tokens. Very long premise-hypothesis pairs may be truncated, potentially degrading accuracy.
- **Precision:** The model is loaded and run in bfloat16. Running in lower precision (e.g., int8) without additional calibration may affect outputs.

---

## Metrics

**Performance measures:**
- **Accuracy:** Proportion of correctly classified samples over the full evaluation set.
- **Weighted F1-score:** F1 computed per class and weighted by class support, accounting for label imbalance.
- **Macro F1-score:** Unweighted mean F1 across both classes.
- **Per-class precision and recall:** Reported separately for class 0 (Non-entailment) and class 1 (Entailment).

**Decision threshold:** Not applicable — the model generates a single label token (`"0"` or `"1"`) via greedy decoding constrained to those two tokens. There is no soft probability threshold to tune.

**Variation approach:** Performance is compared between zero-shot baseline (unmodified `Qwen3.5-9B-Base`) and the fine-tuned LoRA adapter to measure the impact of supervised fine-tuning.

---

## Evaluation Data

**Dataset:** The same dataset provided as part of the **COMP34812 coursework at the University of Manchester** (dev split used for evaluation; test labels are held out).

| Split | Samples |
|---|---|
| Dev (evaluation) | 6,736 |
| Test (held out) | 3,302 |

**Motivation:** The dev set was used for evaluation to monitor generalisation during training, as test labels are not available to students.

**Label balance (dev set):** Class 0 — 3,258 samples; Class 1 — 3,478 samples (approximately balanced, ~48.4% / 51.6%).

**Preprocessing:** Each row is formatted into the chat prompt template described in Model Details. Sequences are tokenised with left-padding and truncated to 256 tokens. At inference time, only the prompt is provided (no answer token appended).

---

## Training Data

**Dataset:** Dataset provided as part of the **COMP34812 coursework at the University of Manchester**. The dataset is not publicly redistributable and was supplied exclusively for coursework use.

| Split | Samples |
|---|---|
| Train | 24,432 |
| Dev | 6,736 |
| Trial | 50 |
| **Total (labelled)** | **31,218** |

**Format:** CSV with columns `premise`, `hypothesis`, and `label` (0 or 1).

**Preprocessing:** Each training example is converted to the full chat-template conversation (system + user + assistant turns). The prompt tokens are masked with `-100` in the loss computation so that only the label token contributes to training loss. Sequences are padded/truncated to 256 tokens.

**Data provenance:** The dataset is in the style of established NLI corpora (SNLI, MultiNLI), though the exact construction methodology is not disclosed as part of coursework materials.

---

## Quantitative Analyses

### Unitary Results

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Baseline (`Qwen3.5-9B-Base`, zero-shot) | 67.83% | 0.6501 | — |
| Fine-tuned (LoRA, 1 epoch) | **93.29%** | **0.9329** | **0.933** |

Fine-tuning improved accuracy by **+25.46 percentage points** over the zero-shot baseline.

### Intersectional Results (Per-Class, Fine-tuned)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 — Non-entailment | 0.924 | 0.938 | 0.931 | 3,258 |
| 1 — Entailment | 0.941 | 0.928 | 0.935 | 3,478 |
| **Macro avg** | **0.933** | **0.933** | **0.933** | 6,736 |
| **Weighted avg** | **0.933** | **0.933** | **0.933** | 6,736 |

Performance is balanced across both classes (F1 difference of 0.004), suggesting the model does not exhibit strong per-class bias on the dev set.

### Training Loss Progression

Training loss decreased from **1.293** (step 1) and stabilised in the range **0.82–0.87** by the end of the epoch (38 logged steps), indicating convergence without observed divergence or overfitting within the single training epoch.

---

## Ethical Considerations

- **Data sensitivity:** The training data consists of academic NLI sentence pairs and does not contain personal or sensitive information.
- **Bias in training data:** As the dataset is derived from academic NLI corpora, it may reflect the linguistic biases present in those corpora (e.g., over-representation of formal English, particular demographic perspectives in crowdsourced annotations). These biases have not been audited for this model.
- **Misuse potential:** The model is designed for academic NLI classification and is not suitable for making consequential decisions about individuals (e.g., screening applications, legal inference) without expert review.
- **Transparency:** The base model (`Qwen3.5-9B-Base`) is a large pre-trained language model whose full training data and provenance are described in the Qwen Technical Report. Users should be aware that capabilities and biases from pre-training may carry over into the fine-tuned model.

---

## Caveats and Recommendations

- **Single epoch training:** The model was fine-tuned for 1 epoch. Extended training with learning rate tuning (via the provided Optuna sweep) may yield further improvements.
- **Dev-set evaluation only:** Reported metrics are from the dev split. Generalisation to the held-out test set and to out-of-distribution data is not confirmed.
- **Domain generalisation:** Users applying this model outside the academic NLI domain should validate performance on representative in-domain samples before deployment.
- **Prompt sensitivity:** The model must receive inputs formatted in the exact prompt structure described in Model Details. Minor changes to wording or structure may affect predictions.
- **Inference constraint:** The `_LabelOnlyLogitsProcessor` guarantees a valid `0`/`1` output, but this constraint bypasses the model's natural probability distribution. For applications requiring calibrated confidence scores, further calibration work is needed.
- **For interactive use:** See [`demo_slm.ipynb`](demo_slm.ipynb) for a working inference example.

---

## How to Use

```python
from src.config import SLMConfig, SLMFinetuneConfig
from src.models.slm_classifier import load_finetuned_for_inference, predict_batch
from src.slm_utils.slm_dataset import SLMNLIDatasetInference
import pandas as pd
from torch.utils.data import DataLoader

ADAPTER_PATH = "models/slm_adapter_<datetime>"

cfg_slm = SLMConfig()           # model = "Qwen/Qwen3.5-9B-Base"
cfg_ft  = SLMFinetuneConfig()   # LoRA settings

model, tokeniser = load_finetuned_for_inference(ADAPTER_PATH, cfg_slm, cfg_ft)
device = next(model.parameters()).device

df      = pd.read_csv("data/dev.csv")
dataset = SLMNLIDatasetInference(df, tokeniser, max_length=cfg_slm.max_seq_length)
loader  = DataLoader(dataset, batch_size=64)

predictions = []
for batch in loader:
    preds = predict_batch(model, tokeniser, batch["input_ids"], batch["attention_mask"], device)
    predictions.extend(preds)
```

---

## References

- Mitchell, M. et al. *Model Cards for Model Reporting*. FAccT 2019. arXiv:1810.03993.
- Qwen Team. *Qwen Technical Report*. arXiv:2309.16609, 2023.
- Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021.
- Bowman, S. R. et al. *A large annotated corpus for learning natural language inference (SNLI)*. arXiv:1508.05326, 2015.
- Williams, A. et al. *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MultiNLI)*. arXiv:1704.05426, 2018.
- Wolf, T. et al. *Hugging Face Transformers*. 2019. [github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- Von Werra, L. et al. *TRL: Transformer Reinforcement Learning*. [github.com/huggingface/trl](https://github.com/huggingface/trl)
- Hu, E. et al. *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning*. [github.com/huggingface/peft](https://github.com/huggingface/peft)
