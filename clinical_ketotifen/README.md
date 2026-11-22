# Clinical application: ketotifen soft-tissue sarcoma study

Code used to apply the preclinical **MM-ViT16** model (PROGNOSTIC) to shear-wave elastography (SWE) images from the
ketotifen clinical study, without retraining. Predictions are compared with RECIST 1.1 per treatment arm and time point.

Patient images and clinical data are **not** included. They are available from the corresponding author on reasonable
request, subject to ethics approval and a data-use agreement.

## Contents

| File | Purpose |
|---|---|
| `01_predictions_by_group_and_timepoint.ipynb` | Main analysis. Predictions for the Experimental arm (baseline, one week of ketotifen, mid-treatment) and the Control arm (baseline, mid-treatment), plus one confusion matrix per arm and time point |
| `02_swe_patient_BOC_image_plus_kPa.ipynb` | Earlier analysis: one segmented image per patient, image + rescaled elastic modulus |
| `03_mid_treatment_image_only_and_kPa_check.ipynb` | Mid-treatment, image only; also checks every elastic-modulus input with the image + kPa model |
| `boc_lib.py` | Model loading and preprocessing helpers |

## Model weights

From the MM-ViT16 training notebook (`models/MM-ViT16.ipynb`), run 1 (seed 42):

- `best_model_1.keras`: image + elastic modulus
- `best_model_image_only_1.keras`: image only (the elastic-modulus token is replaced by a learnable class token)

Load the **`.keras`** files, not `.weights.h5`. The positional embedding is stored only in the `.keras` configuration.
`boc_lib.load_keras_model` handles two TF 2.15 deserialisation issues without changing the model's outputs:

- the `cls_pick` Lambda layer is replaced by an equivalent `TokenSlice` layer;
- the constant positional-embedding input is replaced by an `AddConstant` layer that holds the same values.

## Method summary

- **Images:** the tumour region is segmented automatically from co-registered B-mode images. The segmented tumour is placed on a white 400 × 300 canvas, resized to 304 × 400 and scaled to [0, 1].
- **Patient prediction:** the mean class probability over all of that patient's frames at that session; the predicted class is the one with the highest mean probability.
- **Elastic modulus:** mapped onto the murine training scale with a label-free distribution match:
  `kPa_scaled = mu_mouse + sd_mouse * (kPa - mu_clinical) / sd_clinical`
  where `mu_clinical` and `sd_clinical` come from all clinical SWE measurements and `mu_mouse` and `sd_mouse` from the 1,578 murine training images (11.76 + 2.54 × kPa).
- **Inputs by time point:** image + elastic modulus at baseline and at one week; image only at mid-treatment.

## Environment

TensorFlow / Keras **2.15** (the version the models were saved with), Python 3.11, `numpy<2`, pandas, openpyxl,
scikit-learn, matplotlib and pillow.

## Expected local data layout (not in the repository)

```
<root>/prognostic-ml/03_models/MM-ViT16/run1/              model weights
<root>/prognostic-ml/04_artifacts/MM-ViT16/run1/           model_config.json (kPa standardisation)
<root>/prognostic-ml/02_data/clustering_all_v5.xlsx        murine kPa table
<root>/prognostic-ml/02_data/artifact source/              RECIST labels; one segmented image per patient
<root>/Export/new_clustering/mid/p<N>/                     mid-treatment segmented frames
<root>/Export/experimental_basline_crop/p<N>/              Experimental-arm baseline segmented frames
<root>/Export/...                                          clinical stiffness tables;
                                                           mid_visit_dates.csv and colourbar_reference_frame.png (notebook 03)
```
