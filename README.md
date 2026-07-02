# Multimodal Transformers Predict Cancer Therapy Response from Tumor Mechanics

Code accompanying the manuscript **"Multimodal Transformers Predict Cancer Therapy
Response from Tumor Mechanics"** (Georgiou, Stylianopoulos & Voutouri).

This repository provides the model definitions, training notebooks, ablation and
generalization experiments, and the stiffness-only baseline used to produce the
results in the paper. It is intended to enable independent verification and reuse.

---

## Overview

We predict preclinical cancer therapy response (responder / stable disease /
non-responder) from **shear wave elastography (SWE)** by fusing two modalities:

- the **SWE image** (spatial elastographic texture), and
- the **scalar tissue stiffness** (elastic modulus, kPa).

The core model is a **multimodal transformer** in which the image is encoded into
patch/CNN tokens and the elastic modulus is embedded as a dedicated **numeric
token**; self-attention then fuses both modalities (token-level cross-modal fusion).

---

## Repository structure

```
.
├── models/                     # Model architectures + training notebooks
│   ├── MM-ViT16.ipynb                     # Vision-Transformer variant (primary model)
│   ├── MM-Transformer.ipynb               # CNN -> 2-token transformer variant
│   ├── MM-Attention-Augmentation.ipynb    # Attention + augmentation variant
│   ├── MM-Augmentation.ipynb              # Augmentation variant
│   ├── MM-Transformer-Attention.ipynb     # Transformer + attention-layer variant
│   └── MM-Simple-Baseline.ipynb           # Simple multimodal CNN baseline
├── experiments/                # Reproduces the paper's validation analyses
│   ├── MM-ViT16_ablation_study.ipynb      # Fusion ablation (Reviewer #2.3, #2.4)
│   ├── MM-ViT16_leave_one_tumor_out.ipynb # LOTMO generalization (Reviewer #2.2)
│   └── Model_Comparison.ipynb             # Cross-architecture comparison
├── baselines/
│   └── stiffness_only_logistic_regression.py   # Stiffness-only (kPa) baseline
├── requirements.txt
└── README.md
```

---

## Data availability

The SWE/B-mode images, segmentation masks, and the per-image elastic-modulus (kPa)
table are **available from the corresponding author on reasonable request**, subject
to a data-use agreement consistent with institutional and funder policies. Raw data
are therefore **not** included in this repository.

The notebooks expect:

- an image directory `Elastography_images/` with one subfolder per class
  (`response/`, `stable/`, `non-response/`), and
- an Excel file (`clustering_all_v5.xlsx`) mapping each image filename to its
  elastic modulus (kPa), response label, and tumor model (cell line).

Paths are configurable at the top of each notebook (`IMAGES_DIR`, `KPA_EXCEL`).
The stiffness-only script reads `KPA_EXCEL` from the environment
(`export KPA_EXCEL=/path/to/clustering_all_v5.xlsx`) or a local default.

---

## Reproducibility

- **Fixed seeds.** All experiments run over seeds `[42, 1, 2, 3, 4]`; results are
  reported as mean ± 95% CI across seeds.
- **Fixed splits.** Each run writes its exact train/validation/test file lists to a
  `config/splits/` manifest, so the partitions are recoverable.
- **Preprocessing.** Images are resized to 304×400, scaled to `[0, 1]`; the kPa
  value is rank-normalized (percentile within the training distribution), fit on the
  training partition only.
- **Environment.** See `requirements.txt` for pinned versions (TensorFlow 2.15).
- **Determinism.** `tf.config.experimental.enable_op_determinism()` is enabled where
  supported; note that exact per-seed reproducibility is not guaranteed on all GPUs,
  which is why confidence intervals over seeds are reported.

### Reproducing the main analyses

| Result | Notebook |
|--------|----------|
| Primary model (image + kPa) | `models/MM-ViT16.ipynb` |
| Fusion ablation (image-only, token-removed, shuffled, late-fusion, stiffness-only) | `experiments/MM-ViT16_ablation_study.ipynb` |
| Leave-one-tumor-model-out generalization | `experiments/MM-ViT16_leave_one_tumor_out.ipynb` |
| Stiffness-only baseline | `baselines/stiffness_only_logistic_regression.py` |

Each experiment notebook saves per-run metrics, per-image predictions, training
histories, trained models, split manifests, and figures under a local results
directory (git-ignored).

---

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for the transformer models. The stiffness-only
baseline runs on CPU in seconds.

---

## Citation

If you use this code, please cite the accompanying paper (full reference to be added
upon publication).

## License

Released for academic use. See `LICENSE` (to be added) for terms.
