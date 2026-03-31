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