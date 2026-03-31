---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/jackr03/34812-cwk-S-Project

---

# Model Card for x06213zr-j84846ky-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a binary Natural Language Inference (NLI) classification model that, given a premise and hypothesis, predicts whether the premise entails the hypothesis.
    It is based on the `Qwen/Qwen3.5-9B-Base` causal language model fine-tuned via supervised instruction tuning with LoRA (Low-Rank Adaptation) adapters.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is `Qwen/Qwen3.5-9B-Base` (~9B parameters) fine-tuned for binary NLI using parameter-efficient fine-tuning (PEFT) with LoRA adapters via the
    Hugging Face TRL `SFTTrainer`. Premise-hypothesis pairs are formatted as structured instruction-tuning prompts using Qwen3's chat template, and the model is trained to generate
    the correct label token (`"0"` or `"1"`) as a single-token completion. LoRA adapters are applied to 7 projection layers
    (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), keeping the base model weights frozen.

    At inference time, a `_LabelOnlyLogitsProcessor` masks all vocabulary tokens except `"0"` and `"1"` to `-inf`, ensuring the model always outputs a valid binary label.

    The prompt format (using Qwen3's chat template with `enable_thinking=False`) is:
    - **System:** "You are a binary Natural Language Inference classifier. Given a premise and a hypothesis, predict whether the premise entails the hypothesis."
    - **User:** Premise: {premise} / Hypothesis: {hypothesis} / Return ONLY the number of the correct label. / 0 = Non-entailment / 1 = Entailment / Answer:
    - **Assistant (training target):** "0" or "1" (single token)

- **Developed by:** Zhijie Rong and Kyan Yip
- **Language(s):** English
- **Model type:** Fine-tuned (supervised instruction tuning)
- **Model architecture:** Transformer (decoder-only, Qwen3.5)
- **Finetuned from model [optional]:** Qwen/Qwen3.5-9B-Base (~9B parameters)

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/Qwen/Qwen3.5-9B-Base
- **Paper or documentation:** https://arxiv.org/abs/2106.09685

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24,432 premise-hypothesis pairs supplied by the COMP34812 coursework dataset (train split).
    Each example is converted to the full chat-template conversation (system + user + assistant turns).
    The prompt tokens are masked with `-100` in the loss computation so that only the label token contributes to training loss.
    Sequences are padded/truncated to 256 tokens.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - seed: 100
      - num_epochs: 1
      - per_device_train_batch_size: 16
      - gradient_accumulation_steps: 4  (effective batch size: 64)
      - learning_rate: 1e-4
      - lr_scheduler: cosine
      - max_seq_length: 256 tokens
      - optimizer: AdamW (default)
      - precision: bfloat16
      - LoRA rank (r): 16
      - LoRA alpha: 32
      - LoRA dropout: 0.0
      - LoRA target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: ~57 minutes (3,433 s)
      - LoRA adapter size: ~135 MB
      - base model size: ~18 GB (bfloat16)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

6,736 premise-hypothesis pairs from the dev split of the COMP34812 coursework dataset.
    Test labels are held out (3,302 samples). The dev set is approximately balanced: class 0 — 3,258 samples; class 1 — 3,478 samples (~48.4% / 51.6%).

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Macro Precision
      - Macro Recall
      - Macro F1
      - Weighted Precision
      - Weighted Recall
      - Weighted F1
      - Per-class Precision, Recall, and F1 (class 0: Non-entailment; class 1: Entailment)
      

### Results


    Given a zero-shot baseline of 67.83% accuracy (macro F1: 0.6501) on the unmodified `Qwen3.5-9B-Base`, fine-tuning with LoRA for 1 epoch achieved:
      - Accuracy:               93.29%  (+25.46 percentage points over baseline)
      - Weighted Precision:     0.9330
      - Weighted Recall:        0.9329
      - Weighted F1:            0.9329
      - Macro Precision:        0.9330
      - Macro Recall:           0.9330
      - Macro F1:               0.9330

    Per-class breakdown (fine-tuned model):
      - Class 0 (Non-entailment): Precision 0.924 | Recall 0.938 | F1 0.931 | Support 3,258
      - Class 1 (Entailment):     Precision 0.941 | Recall 0.928 | F1 0.935 | Support 3,478

    Performance is balanced across both classes (F1 difference of 0.004).
    

## Technical Specifications

### Hardware


      - GPU: NVIDIA A100-SXM4-80GB (79.25 GB VRAM required for full fine-tuning in bf16)
      - RAM: at least 40 GB system RAM
      - Storage: at least 25 GB (base model ~18 GB + adapter + dataset)

### Software


      - PyTorch >=2.0
      - Hugging Face Transformers
      - PEFT (Parameter-Efficient Fine-Tuning)
      - TRL (SFTTrainer)
      - Optuna (optional, for hyperparameter search)
      - pandas, scikit-learn

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->


    1. Inputs (concatenation of premise + hypothesis + prompt overhead) longer than 256 tokens are truncated, which may degrade accuracy on longer inputs.
    2. The model was fine-tuned exclusively on English text and will not work reliably with other languages.
    3. Training data is derived from academic NLI corpora (SNLI/MultiNLI style), so performance may degrade on informal, conversational, or domain-specific text (e.g., legal, medical).
    4. The model is sensitive to the exact prompt format used during training; deviations from the expected system/user structure may affect predictions.
    5. The base model (`Qwen3.5-9B-Base`) was pre-trained on a large web corpus; biases present in that pre-training data may carry over into the fine-tuned model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Hyperparameters (learning rate, LoRA rank, LoRA alpha, dropout) can be selected via an optional Optuna sweep consisting of 8 trials.
    The reported results use 1 training epoch; extended training with learning rate tuning may yield further improvements.
    Reported metrics are from the dev split only; generalisation to the held-out test set and out-of-distribution data is not confirmed.
