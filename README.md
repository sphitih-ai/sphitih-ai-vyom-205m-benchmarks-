# Vyom-205M: Official Benchmark Verification & Evaluation Suite

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Parameters](https://img.shields.io/badge/Parameters-204.96M-green.svg)](#subsystem-parameter-accounting)
[![Verification](https://img.shields.io/badge/Verification-Passing-brightgreen.svg)](#reproducing-results-in-10-seconds)
[![Organization](https://img.shields.io/badge/Organization-sphitih--ai-orange.svg)](https://github.com/sphitih-ai)
[![Preprint](https://img.shields.io/badge/Preprint-PDF_Available-red.svg)](paper/VYOM_205M_PREPRINT.pdf)

Official evaluation suite, publication preprint, and standalone verification harness for **Vyom-205M** (*Sovereign Multi-Task Agronomic Foundation Model*), developed by the **Sphitih AI Research Team**.

📄 **Read the Research Preprint:** [`paper/VYOM_205M_PREPRINT.pdf`](paper/VYOM_205M_PREPRINT.pdf) (Compiled PDF, 15 Pages)  
📝 **Preprint Markdown Source:** [`paper/VYOM_205M_PREPRINT.md`](paper/VYOM_205M_PREPRINT.md)  
🚀 **GitHub Organization:** [https://github.com/sphitih-ai](https://github.com/sphitih-ai)  
🔬 **Public Repository:** [https://github.com/sphitih-ai/vyom-205m-benchmarks](https://github.com/sphitih-ai/vyom-205m-benchmarks)

---

## Executive Summary

Existing Earth Observation Foundation Models (e.g., NASA/IBM Prithvi, SatMAE, CROMA, Clay, DOFA) demonstrate impressive continental-scale classification, but exhibit systemic degradation when applied to smallholder agrarian systems across the Global South. Over 500 million smallholder farms feed more than 2 billion people, yet operate in fragmented landscapes characterized by:
1. **Sub-Pixel Earthen Bunds:** Micro-plots (<0.4 ha) demarcated by 0.5–1.5 m earthen embankments invisible to standard 10 m satellite boundary detectors.
2. **Monsoonal Cloud Gaps:** >75% optical cloud occlusion during peak Kharif phenology, interrupting optical time series for 30–75 consecutive days.
3. **Complex Crop Stress:** Confounded spectral responses between nitrogen chlorosis, water deficit, and pest defoliation.
4. **Modality Disconnect:** Siloed operational data across drone/aerial surveys (0.5–2 m), Sentinel-2 optical, Sentinel-1 SAR, and gridded soil chemistry.

**Vyom-205M** bridges this gap with an end-to-end 204.96M-parameter multi-stream architecture fusing sub-meter cadastral optical imagery, 12-band Sentinel-2 time series, Sentinel-1 C-band SAR (VV, VH, ratio, RVI), and gridded soil chemistry vectors through a pre-layer-normalized spatio-temporal Swin Transformer core.

---

## Model Architecture

![Vyom-205M Architecture](paper/figures/fig_vyom_architecture.png)
*Figure 1: High-level architectural schematic of Vyom-205M. Four front-end encoders (3.99M params) project high-resolution cadastral optical data, multi-temporal 12-band Sentinel-2 imagery, multi-temporal Sentinel-1 SAR, and soil covariates into a unified token space. The 4-stage Hierarchical Swin Transformer core (190.59M params) applies Pre-LayerNorm attention with Query-Key Normalization ($\text{QK-Norm}$). Four task decoders (10.38M params) execute cadastral delineation (Head A), SAR-to-optical synthesis (Head B), phenology clustering (Head C), and parcel stress diagnostics (Head D).*

---

## Benchmark Scoreboard & SOTA Comparisons

### 1. Global Flood Inundation (Sen1Floods11 Official Test Split)
Evaluated following the official PANGAEA benchmark protocol across 30 held-out test scenes spanning 10 countries (Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA):

| Foundation Model | Parameters | Pretraining Domain | Water IoU (%) | Mean IoU (%) | Pixel Acc (%) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **SatMAE (Base)** [3] | 86M | S2 Optical (Global) | 61.20% | 76.50% | 91.80% |
| **Prithvi-100M** [1] | 100M | NASA/USGS HLS Optical | 68.40% | 81.20% | 93.80% |
| **CROMA (Large)** [4] | 300M | S2 Optical + S1 SAR (Global) | 71.40% | 83.10% | 94.60% |
| **Clay (v1 Base)** [5] | 100M | Multi-Modal Optical/SAR (Global) | 72.10% | 83.60% | 94.90% |
| **DOFA (Large)** [6] | 300M | Multi-Spectral Global Waveband | 73.50% | 84.40% | 95.30% |
| **Prithvi-EO-2.0** [2] | 300M | NASA HLS + Multi-Sensor (Global) | **77.80%** | **87.10%** | **96.80%** |
| **Vyom-205M (Ours)** | **205M** | **Sovereign Multi-Stream Agronomic** | **75.81%** | **85.83%** | **96.33%** |

*Vyom-205M outperforms Prithvi-100M by **+7.41% Water IoU** and CROMA-Large by **+4.41%**, approaching within 1.99% of Prithvi-EO-2.0 while utilizing 31.7% fewer parameters.*

### 2. Smallholder Cadastral Boundary Delineation (349 Held-Out Scenes)
Evaluated across sovereign validation scenes with average plot sizes of 0.28 hectares and sub-meter earthen bunds:

| Architecture | Backbone | Input Modality | Parcel Extent IoU | Parcel Extent F1 | Bund 2-px F1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **U-Net** | ResNet-34 | High-Res Optical | 0.6920 | 0.8180 | 0.1640 |
| **ResU-Net** [13] | ResNet-50 | High-Res Optical | 0.7340 | 0.8465 | 0.1980 |
| **FracTAL-ResUNet** [12] | FracTAL Core | High-Res Optical | 0.7610 | 0.8643 | 0.2240 |
| **SegFormer-B4** [15] | MiT-B4 | High-Res Optical | 0.7850 | 0.8795 | 0.2390 |
| **Vyom-205M (Head A)** | **Hierarchical Swin** | **Cadastral + S2 + S1 Multi-Modal** | **0.8230** | **0.9029** | **0.2783** |

### 3. All-Weather SAR-to-Optical Cloud Synthesis (116 Agronomic Cubes)
Evaluated under the SEN12MS-CR protocol translating Sentinel-1 SAR into cloud-free Sentinel-2 optical imagery:

| Model | Conditioning Input | Target Modality | SSIM ($\uparrow$) | MAE / L1 ($\downarrow$) |
| :--- | :--- | :--- | :---: | :---: |
| **Pix2Pix** [16] | S1 SAR (VV, VH) | S2 Optical (RGB) | 0.6840 | 0.0680 |
| **CycleGAN** [17] | S1 SAR (VV, VH) | S2 Optical (RGB) | 0.6420 | 0.0790 |
| **DSen2-CR** [10] | S1 SAR + S2 Cloud Mask | S2 Optical (12-Band) | 0.8120 | 0.0410 |
| **GLF-CR** [11] | S1 SAR + S2 Optical | S2 Optical (12-Band) | 0.8540 | 0.0350 |
| **Vyom-205M (Head B)** | **Multi-Temporal S1 SAR** | **S2 Optical (Spectral Bands)** | **0.8924** | **0.0298** |

### 4. 6-Class Biophysical Crop Stress Diagnostics (550 Held-Out Parcels)
Evaluated across 550 ground-truthed smallholder parcels:

| Class ID & Diagnostic Category | Validation Parcels ($N$) | Precision (%) | Recall (%) | F1-Score | Biophysical Indicator |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Class 0: Healthy Canopy** | 186 | 63.33% | 71.51% | **0.6717** | Normal vegetative vigor and canopy density |
| **Class 1: Moisture Deficit / Drought** | 122 | 54.91% | 77.87% | **0.6441** | Early water deficit detected via NDMI divergence |
| **Class 2: Nutrient Deficiency / Chlorosis** | 35 | 37.50% | 17.14% | **0.2353** | Stunted peak vigor and red-edge depression |
| **Class 3: Pest / Disease Defoliation** | 179 | 56.52% | 36.31% | **0.4422** | Sudden intra-seasonal vegetative collapse |
| **Class 4: Flood Inundation / Waterlogging** | 16 | 86.67% | 81.25% | **0.8387** | High-confidence SAR-optical flood detection |
| **Class 5: Soil Salinity / Usar Stunting** | 12 | 47.62% | 83.33% | **0.6061** | Persistent soil-induced vegetative stunting |
| **Overall Dataset Metrics** | **550** | — | — | **0.5730** | **Overall Accuracy: 58.55%** |

---

## Visual Qualitative Evaluation & Radiometric Verification

### 1. Smallholder Cadastral Parcel Delineation (Head A)
![Cadastral Parcel Boundary Evaluation](paper/figures/fig_cadastral_publication.png)
*Figure 2: Multi-stage cadastral parcel boundary evaluation: (a) Input sub-meter aerial optical imagery; (b) Ground truth surveyed cadastre mask; (c) Head A predicted parcel interior extent probability (Viridis colormap: purple denotes background, transitioning through green to yellow for high-confidence field interiors); (d) Head A predicted earthen bund boundary probability (Magma colormap: black denotes background, fiery yellow denotes ridge peaks); (e) Boundary distance transform field; (f) Watershed extracted individual parcel instances.*

### 2. All-Weather SAR-to-Optical Cloud Penetration (Head B)
![SAR-to-Optical Cloud Penetration Evaluation](paper/figures/fig_sar2opt_publication.png)
*Figure 3: SAR-to-optical cloud penetration evaluation: (a) Input Sentinel-1 dual-polarization C-band SAR false-color composite ($\text{Red} = \sigma^0_{VV}, \text{Green} = \sigma^0_{VH}, \text{Blue} = \sigma^0_{VH}/\sigma^0_{VV}$). Vegetated terrain naturally appears in yellow-green hues due to simultaneous volumetric scattering in cross-polarization ($VH$) and surface scattering in co-polarization ($VV$); (b) Ground truth cloud-free Sentinel-2 optical L2A image showing agricultural field chlorophyll reflectance; (c) Head B reconstructed optical surface reflectance ($\text{SSIM} = 0.892$); (d) Absolute surface reflectance error map ($\text{MAE} = 0.0298$).*

### 3. Biophysical Stress Diagnostics Confusion Matrix (Head D)
![Biophysical Crop Stress Confusion Matrix](paper/figures/fig_stress_confusion_matrix.png)
*Figure 4: Heatmap of the Head D 6-class biophysical crop stress confusion matrix evaluated across 550 held-out smallholder parcels. Overall Accuracy: 58.55% | Macro-F1: 0.5730.*

---

## Subsystem Parameter Accounting

Vyom-205M's modular parameter budget balances multi-modal representation capacity with edge inference efficiency:

| Subsystem Component | Operational Function | Parameter Count | Share (%) |
| :--- | :--- | :---: | :---: |
| **Input Modality Front-Ends** | Cadastral ConvNet, S2 3D-Patch, S1 SAR, Soil Projection | 3,989,632 | 1.95% |
| **Hierarchical Swin Backbone** | 4-Stage Spatio-Temporal Core with Pre-LN & QK-Norm | 190,589,184 | 92.99% |
| **Head A: Cadastral Decoder** | Dual-Stream Multi-Scale Extent & Bund Boundary Module | 4,218,880 | 2.06% |
| **Head B: SAR2Opt Synthesizer** | Multi-Scale Conditional Optical Reconstruction Module | 3,145,728 | 1.53% |
| **Head C: Phenology Clusterer** | Self-Supervised Temporal Prototype Decoder | 1,048,576 | 0.51% |
| **Head D: Biophysical Decoder** | Field-Pooled Feature Extractor & Diagnostic Classifier | 1,966,733 | 0.96% |
| **Total Model Footprint** | **End-to-End Foundation Model** | **204,958,733** | **100.00%** |

---

## Reproducing Results in 10 Seconds

To enable zero-friction peer review, this repository includes a standalone verification script that mathematically re-computes all metrics directly from raw confusion counts and ground truth arrays in under **0.1 seconds**, with zero external deep learning dependencies:

```bash
# Clone the repository
git clone https://github.com/sphitih-ai/vyom-205m-benchmarks.git
cd vyom-205m-benchmarks

# Run the standalone verification suite (requires standard Python 3.8+)
python verify_benchmarks.py
```

---

## Citation

If you find Vyom-205M useful in your research or applications, please cite:

```bibtex
@article{sphitih2026vyom205m,
  title={{Vyom-205M: A Sovereign Multi-Resolution Spatio-Temporal Foundation Model for Smallholder Cadastral Delineation, SAR-to-Optical Cloud Synthesis, and Biophysical Crop Stress Diagnostics}},
  author={{Sphitih AI Research Team}},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/sphitih-ai/vyom-205m-benchmarks}
}
```

---

## License

This repository and evaluation suite are licensed under the [Apache 2.0 License](LICENSE).  
For commercial partnerships, enterprise sovereign deployments, or model access inquiries, contact: **`contact@sphitih.com`**.
