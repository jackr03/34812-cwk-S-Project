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


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based on the Enhanced Sequential Inference Model (ESIM; Chen et al., 2017), implemented from scratch in PyTorch. It uses frozen 840B GloVe 300d word embeddings,
    a shared bidirectional LSTM encoder, a soft cross-attention alignment mechanism between premise and hypothesis, and a composition BiLSTM over the enhanced representations. The final classification head
    receives mean and max-pooled outputs from both sequences.

- **Developed by:** Zhijie Rong and Kyan Yip
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** LSTM
- **Finetuned from model [optional]:** N/A

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** N/A
- **Paper or documentation:** https://arxiv.org/abs/1609.06038

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24,000 premise-hypothesis pairs supplied by the COMP34812 coursework dataset.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - seed: 100
      - batch_size: 256
      - num_epochs: 100 (early stopping if no improvement after 10 epochs)
      - learning_rate: 2e-3
      - dropout: 0.174
      - optimizer: AdamW
      - weight_decay: 1.72e-05

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 76s
      - duration per training epoch: ~3.5s
      - model size: 13MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

6700 premise-hypothesis pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Macro Precision
      - Macro Recall
      - Macro F1
      - Weighted Precision
      - Weighted Recall
      - Weighted F1
      

### Results


    Given a coursework baseline at ~66% accuracy and a macro F1 of ~0.66, we achieved:
      - Accuracy:           73.06%
      - Macro Precision:    0.7303
      - Macro Recall:       0.7304
      - Macro F1:           0.7303
      - Weighted Precision: 0.7306
      - Weighted Recall:    0.7306
      - Weighted F1:        0.7306
    

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB
      - GPU: V100

### Software


      - Pytorch 2.10.0+cu124

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->


    1. Any inputs (concatenation of two sequences) longer than 64 tokens are truncated, which may degrade accuracy on longer inputs.
    2. Tokens not found in GloVe are represented as zero vectors.
    3. The model was trained exclusively on English text and will not work with other languages.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Hyperparameters (lr, dropout and weight decay) were selected via Optuna sweeps consisting of 20 trials being performed, each being up to 5 epochs.
