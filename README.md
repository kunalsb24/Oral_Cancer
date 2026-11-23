# Oral Cancer (OSCC) Histopathology: Foundation Models and Clean Evaluation

Two Kaggle notebooks that (1) benchmark pathology foundation models, calibrated uncertainty and concept-based explanations on a public oral squamous cell carcinoma (OSCC) histopathology dataset, and (2) show how much of the 97-99% accuracy commonly reported on this dataset depends on the evaluation protocol rather than the model.

> **Main finding:** the same models score 97-99% under a random split of the pooled (augmented) images, but 88-92% on a clean test set of original, unaugmented images. Test images under the random split have many more near-duplicates in the training set than the clean test images do.

---

## Contents

| File | What it is |
|---|---|
| [`oscc-pipeline.ipynb`](oscc-pipeline.ipynb) | Foundation-model pipeline: frozen probes, fine-tuning, conformal prediction, concept bottleneck, stain robustness, leakage audit |
| [`oscc-clean-protocol-benchmark.ipynb`](oscc-clean-protocol-benchmark.ipynb) | Protocol comparison: the same two models under three evaluation protocols (A, B, C) plus a near-duplicate audit |

Both notebooks are self-contained, each is one main code cell, and both run on Kaggle with the dataset attached (the first cell is the default Kaggle file-listing boilerplate).

## Dataset

[Histopathological Imaging Dataset for Oral Cancer](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset) (CC0 on Kaggle), derived from the Rahman et al. repository (*Data in Brief*, 2020).

- 5,192 H&E images in `train/`, `val/`, `test/`, each with `Normal/` and `OSCC/` subfolders.
- The original source has 1,224 images (290 normal, 934 OSCC) from 230 patients at 100x and 400x. The Kaggle copy adds augmented images (filenames starting with `aug_`) to the train split only.
- Val and test contain originals only. Counts from the notebook output:

| Split | Normal | OSCC | Augmented? |
|---|---|---|---|
| train | 2,435 (231 original) | 2,511 (747 original) | mostly yes (`aug_*`: 2,204 + 1,764) |
| val | 28 | 92 | no |
| test | 31 | 95 | no |

No patient IDs are provided, so patient-level splitting is not possible. Near-duplicate grouping is used as a proxy where noted.

---

## Notebook 1: `oscc-pipeline.ipynb`

**Goal:** build a foundation-model OSCC classifier and evaluate it beyond accuracy, on the official Kaggle split (126 test images).

**Pipeline**

1. **Data and caching.** Images are cached at 448x672 (uint8) and as a 224x224 whole-image view.
2. **Frozen foundation-model probes.** Features are the mean over six 224x224 tiles, fed to a logistic-regression probe. Models: Phikon, Phikon-v2, DINOv2-base, PLIP. UNI is skipped unless a Hugging Face token (`HF_TOKEN` Kaggle secret) with accepted license is available.
3. **Fine-tuned baselines.** ImageNet ViT-B/16 and a full fine-tune of Phikon at 224x224 (best epoch chosen on the val split).
4. **Metrics.** Accuracy (with 95% bootstrap CI), precision, sensitivity, specificity, F1, AUC, ECE.
5. **Stain/colour robustness.** Test accuracy under a fixed ColorJitter perturbation.
6. **Split-conformal prediction.** Set-valued predictions at a 95% coverage target, calibrated on the val split.
7. **Concept bottleneck.** PLIP text prompts for 15 morphology concepts, with a classifier that sees only the concept scores, plus remove/boost intervention tests.
8. **Leakage audit.** DINOv2 test-to-train similarity with flip/rotation averaging, and near-duplicate-aware group cross-validation versus random cross-validation.

**Key results** (official test split, n=126, 31 normal; single run)

| Model | Acc | 95% CI | Spec | AUC | Acc under stain shift |
|---|---|---|---|---|---|
| Phikon + linear probe | 86.5 | 80.2-92.1 | 77.4 | 93.1 | 87.3 |
| Phikon-v2 + linear probe | 90.5 | 84.9-95.2 | 77.4 | 92.8 | 86.5 |
| DINOv2 + linear probe | 88.1 | 82.5-93.7 | 74.2 | 92.7 | 87.3 |
| PLIP + linear probe | 83.3 | 76.2-89.7 | 58.1 | 90.2 | 80.2 |
| ViT-B/16 fine-tuned | 88.1 | 82.5-92.9 | 74.2 | 95.0 | 88.9 |
| **Phikon fine-tuned (full)** | **92.1** | 87.3-96.8 | 80.6 | **96.3** | 89.7 |
| PLIP concept bottleneck | 72.2 | 65.1-79.4 | 29.0 | 72.4 | 73.0 |

- **Fine-tuning beats frozen features.** Full-fine-tuned Phikon is the best model here, but its CI overlaps most of the others, so differences within a couple of points are not significant on 126 images.
- **Conformal prediction.** Coverage was 92.9-97.6% against the 95% target. It is noisy because the val set used for calibration is small.
- **Random vs group-aware cross-validation.** Random folds score 2-4.5 points higher than near-duplicate-aware folds for every frozen probe (for example DINOv2: 93.1 vs 88.6).
- **The concept bottleneck did not work.** Accuracy is 72% and several malignant concepts (keratin pearls, nuclear pleomorphism) have negative coefficients. It is included as a negative result, not as validated explainability.

**Outputs written to `/kaggle/working`:** `results.csv`, `results.json`, `leakage_hist.png`, `concept_interventions.csv`, `concept_coefficients.png`, `group_assignments.csv`.

---

## Notebook 2: `oscc-clean-protocol-benchmark.ipynb`

**Goal:** quantify how the evaluation protocol changes reported accuracy, using two models trained identically under three protocols.

| Protocol | Train | Test | Purpose |
|---|---|---|---|
| **A. Random pooled** | 85% of all 5,192 images (4,413) | 15% random (779) | Mimics a random split of the augmented pool |
| **B. Official train, clean test** | Kaggle train folder (4,946) | Original val+test images only (246: 59 normal, 187 OSCC) | Clean evaluation |
| **C. Originals-only train** | Original train images only (978) | Same clean test as B | Tests whether augmented training images help or create a shortcut |

**Models:** EfficientNet-B3 (ImageNet, 8 epochs, lr 3e-4) and Phikon with full fine-tuning (6 epochs, lr 1.5e-5), both at 224x224 with flip/rotation augmentation. Every model is evaluated at its **last epoch**, with no checkpoint selection on test data.

**Results** (seed 42, single run)

| Model | A: random pooled (n=779) | B: clean originals (n=246) | C: originals-only train (n=246) |
|---|---|---|---|
| EfficientNet-B3 | 97.43 (CI 96.3-98.5) | 88.62 (CI 85.0-92.3) | 87.80 (CI 83.7-91.5) |
| Phikon (full FT) | **98.72** (CI 97.8-99.5) | **92.28** (CI 89.0-95.1) | 92.28 (CI 88.6-95.5) |

Specificity (normal class) in the clean protocol is the weak point: 72.9% for EfficientNet-B3 and 79.7% for Phikon, versus about 98-99% under protocol A.

**Near-duplicate audit** (DINOv2 embeddings averaged over 8 flip/rotation views; share of test images whose nearest training image has cosine similarity above the threshold):

| Test set | > 0.95 | > 0.98 |
|---|---|---|
| A: random split, all (779) | 75.0% | 33.4% |
| A: random split, originals only (160) | 76.2% | 35.6% |
| B/C: clean originals (246) | 56.5% | 8.1% |

**Interpretation**

- Under the random split, both models match the 97-99% range reported in the literature for this dataset.
- On clean originals, accuracy drops by 6.4 points (Phikon) and 8.8 points (EfficientNet-B3).
- The 160 original images inside protocol A's test set score 95.6% (EfficientNet-B3) and 97.5% (Phikon), higher than the clean test, and they have far more close training neighbors. This is consistent with near-copies in the training set inflating the random-split numbers.
- Training on only 978 originals (C) performs about the same as training on all 4,946 (B), so the augmented training images add little for clean-image performance, and there is no sign of a "clean-image means OSCC" shortcut.

**Outputs:** `protocol_runs.csv`, `protocol_summary.csv`, `near_duplicate_audit.csv`, `protocol_comparison.png`, `protocol_results.json`.

---

## How to run

1. Create a Kaggle notebook, attach the dataset above, set **Accelerator = GPU** (T4 or P100) and **Internet = On** (model weights are downloaded from Hugging Face).
2. Open one of the notebooks and run all cells. The dataset root is detected automatically under `/kaggle/input`.
3. Approximate runtime on a T4: about 30-40 minutes for `oscc-pipeline`, about 40-60 minutes for `oscc-clean-protocol-benchmark` (one seed).
4. Optional: for UNI, accept the license at `huggingface.co/MahmoodLab/UNI` and add a Kaggle secret named `HF_TOKEN`.
5. For mean ± std over splits, set `SEEDS=[42, 43, 44]` in the second notebook's `CFG` (three times the runtime).

Environment used for the reported numbers: `torch 2.11`, `timm 1.0.30`, a recent `transformers`, scikit-learn.

## Limitations

- **Small clean test set.** The clean evaluation has 246 images (59 normal), so one error is about 0.4 points and one normal-class error is about 1.7 points of specificity. Read the confidence intervals, not the second decimal. Differences between Phikon and EfficientNet-B3 under protocol B overlap and were not tested for significance.
- **Single seed.** Reported numbers come from one run (seed 42). Protocol A uses a random split and could vary.
- **No patient IDs.** Near-duplicate clustering is a proxy for patient-level grouping, not a replacement. Single center, one microscope.
- **Literature comparison is inferential.** I believe published results at 97-100% on this dataset commonly use random splits of the pooled images, but this repository does not verify the protocol of any specific paper.
- **Resolution.** Images were resized to 224x224 (whole image) or 448x672 (tiles), which discards fine nuclear detail.
- **Concept bottleneck.** The PLIP prompts did not produce faithful explanations here.
- **Not a clinical tool.** This is a research benchmark and should not be used for diagnosis.

## Suggested next steps

- Break results down by magnification (`100x` vs `400x`, encoded in filenames) and by class.
- Cross-validate on the 1,224 original images with near-duplicate groups to get a larger clean test set.
- Add paired significance tests (McNemar) and multiple seeds.
- Try LoRA or higher-resolution fine-tuning on tiles, and a stronger concept-grounding model than PLIP.

## Citation and credits

- Dataset: T. Y. Rahman, L. B. Mahanta, A. K. Das, J. D. Sarma, "Histopathological imaging database for oral cancer analysis," *Data in Brief*, 2020. Kaggle copy by Ashenafi Fasil Kebede.
- Models: Phikon / Phikon-v2 (Owkin), DINOv2 (Meta), PLIP, EfficientNet and ViT via `timm`.

## License

Add your license here (for example MIT for code). The dataset is CC0 on Kaggle; please also credit the original authors above.
