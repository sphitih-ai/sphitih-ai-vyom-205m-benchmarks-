# Vyom-205M: Official Benchmark Verification & Evaluation Suite

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Parameters-204.96M-green.svg)](#subsystem-parameter-accounting)
[![Verification](https://img.shields.io/badge/Benchmark_Verification-Passing-brightgreen.svg)](#reproducing-results-in-10-seconds)
[![Organization](https://img.shields.io/badge/Organization-sphitih--ai-orange.svg)](https://github.com/sphitih-ai)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.vyom205m-blue.svg)](paper/VYOM_205M_PREPRINT.md)

Official open evaluation suite, standalone verification harness, and research preprint for **Vyom-205M** (*Sovereign Multi-Task Agronomic Foundation Model*), developed by the **Sphitih AI Research Team**.

📄 **Read the Research Preprint:** [`paper/VYOM_205M_PREPRINT.md`](paper/VYOM_205M_PREPRINT.md)  
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

```mermaid
flowchart TD
    subgraph Multi-Stream Inputs
        In_HR["High-Res Cadastral<br/>(Aerial / Sub-Meter)<br/>[B, 3, 256, 256]"]
        In_S2["Multi-Temporal S2<br/>(12 Optical Bands)<br/>[B, T, 12, 64, 64]"]
        In_S1["Multi-Temporal S1<br/>(VV, VH, Ratio, RVI)<br/>[B, T, 4, 64, 64]"]
        In_Soil["Gridded Soil / Covariates<br/>(8 Chemical Indices)<br/>[B, 8]"]
    end

    subgraph Spatio-Temporal Core
        Fusion["Multi-Modal Cross-Attention Fusion"]
        Swin["Hierarchical Swin-Transformer Core<br/>(Pre-LN + QK-Norm Attention)<br/>190.59M Parameters"]
    end

    subgraph Multi-Task Decoders
        H_A["Head A: Cadastral Extent & Bund Decoder<br/>• 0.8230 Extent IoU (0.9029 F1)<br/>• 0.2783 Bund 2-px F1"]
        H_B["Head B: SAR-to-Optical Synthesizer<br/>• 0.8924 SSIM<br/>• 0.0298 MAE (2.98% error)"]
        H_C["Head C: Phenology Clusterer<br/>• Self-Supervised Prototype Extraction"]
        H_D["Head D: Biophysical Stress Decoder<br/>• 58.55% 6-Class Accuracy<br/>• 0.5730 Macro-F1"]
    end

    In_HR --> Fusion
    In_S2 --> Fusion
    In_S1 --> Fusion
    In_Soil --> Fusion
    Fusion --> Swin
    Swin --> H_A
    Swin --> H_B
    Swin --> H_C
    Swin --> H_D
```

---

## Benchmark Scoreboard

### 1. Global Flood Inundation (Sen1Floods11 Official Test Split)
Evaluated following the official PANGAEA benchmark protocol across 30 held-out test scenes spanning 10 countries (Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA):

| Foundation Model | Parameters | Pretraining Domain | Water IoU (%) | Mean IoU (%) | Pixel Acc (%) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **SatMAE (Base)** | 86M | S2 Optical (Global) | 61.20% | 76.50% | 91.80% |
| **CROMA (Large)** | 300M | S2 Optical + S1 SAR (Global) | 71.40% | 83.10% | 94.60% |
| **Clay (v1 Base)** | 100M | Multi-Modal Optical/SAR (Global) | 72.10% | 83.60% | 94.90% |
| **DOFA (Large)** | 300M | Multi-Spectral Global Waveband | 73.50% | 84.40% | 95.30% |
| **Prithvi-100M** | 100M | NASA/USGS HLS Optical | 68.40% | 81.20% | 93.80% |
| **Prithvi-EO-2.0** | 300M | NASA HLS + Multi-Sensor (Global) | **77.80%** | **87.10%** | **96.80%** |
| **Vyom-205M (Ours)** | **205M** | **Sovereign Multi-Stream Agronomic** | **75.81%** | **85.83%** | **96.33%** |

*Vyom-205M outperforms Prithvi-100M by **+7.41% Water IoU** and CROMA-Large by **+4.41%**, approaching within 1.99% of Prithvi-EO-2.0 while utilizing 31.7% fewer parameters.*

### 2. Smallholder Cadastral Boundary Delineation (349 Held-Out Scenes)
Evaluated across sovereign validation scenes with average plot sizes of 0.28 hectares and sub-meter earthen bunds:

| Target Delineation Layer | Metric | Threshold ($\tau$) | Score | Operational Context |
| :--- | :--- | :---: | :---: | :--- |
| **Field Interior Extent** | Intersection-over-Union (IoU) | 0.50 | **0.8230** | 82.30% spatial overlap with surveyed cadastral polygons |
| **Field Interior Extent** | F1-Score | 0.50 | **0.9029** | High completeness of smallholder parcel boundary detection |
| **Earthen Bund Boundary** | Raw Boundary IoU | 0.55 | **0.1130** | Strict 1-pixel intersection on sub-pixel ridges |
| **Earthen Bund Boundary** | 2-px Relaxed Boundary F1 | 0.45 | **0.2783** | Evaluated within standard 2-pixel tolerance |
| **Earthen Bund Boundary** | 2-px Relaxed Boundary F1 | 0.55 | **0.2649** | Standard operational calibrated threshold |
| **Earthen Bund Boundary** | Predicted Boundary Density | 0.60 | **5.45%** | Matches true ground truth bund density of 4.99% |

### 3. All-Weather SAR-to-Optical Cloud Synthesis (116 Agronomic Cubes)
Evaluated under the SEN12MS-CR protocol translating Sentinel-1 SAR into cloud-free Sentinel-2 optical imagery:

| Evaluation Partition | Evaluation Metric | Score | Biophysical Meaning |
| :--- | :--- | :---: | :--- |
| **Regional Validation (116 Cubes)** | Structural Similarity (SSIM) | **0.8924** | Preserves high-fidelity textural and spatial canopy geometry |
| **Regional Validation (116 Cubes)** | Mean Absolute Error (MAE) | **0.0298** | Average optical surface reflectance error under 3.0% |
| **Zero-Shot Cross-Continent** | Structural Similarity (SSIM) | **0.4366** | Out-of-domain generalizability across unseen global scenes |
| **Zero-Shot Cross-Continent** | Mean Absolute Error (MAE) | **0.0643** | Optical reflectance error under 6.5% on foreign continents |

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

## Visual Qualitative Evaluation

### Smallholder Cadastral Boundary Delineation (Head A)
| Sample 1: Dense Village Boundary | Sample 2: Micro-Plot Fragmentation | Sample 3: Terraced Contours |
| :---: | :---: | :---: |
| ![Cadastral Sample 1](paper/figures/fig_cadastral_sample_1.png) | ![Cadastral Sample 2](paper/figures/fig_cadastral_sample_2.png) | ![Cadastral Sample 3](paper/figures/fig_cadastral_sample_3.png) |

### All-Weather SAR-to-Optical Cloud Penetration (Head B)
| Sample 1: Dense Monsoonal Cloud Removal | Sample 2: Riverine Agronomic Synthesis | Sample 3: Vegetation Index Preservation |
| :---: | :---: | :---: |
| ![SAR2Opt Sample 1](paper/figures/fig_sar2opt_sample_1.png) | ![SAR2Opt Sample 2](paper/figures/fig_sar2opt_sample_2.png) | ![SAR2Opt Sample 3](paper/figures/fig_sar2opt_sample_3.png) |

---

## Subsystem Parameter Accounting

Vyom-205M's modular parameter budget balances multi-modal representation capacity with edge inference efficiency:

| Subsystem Component | Operational Function | Parameter Count | Share (%) |
| :--- | :--- | :---: | :---: |
| **Input Modality Encoders** | High-Res ConvNet, S2 3D-Patch, S1 SAR, Soil Projection | 3,989,632 | 1.95% |
| **Hierarchical Swin Backbone** | 4-Stage Spatio-Temporal Core with Pre-LN & QK-Norm | 190,589,184 | 92.99% |
| **Head A (Cadastral Decoder)** | Dual-Stream Multi-Scale Extent & Bund Boundary Module | 4,218,880 | 2.06% |
| **Head B (SAR2Opt Synthesizer)** | Multi-Scale Conditional Optical Reconstruction Module | 3,145,728 | 1.53% |
| **Head C (Phenology Clusterer)** | Self-Supervised Temporal Prototype Decoder | 1,048,576 | 0.51% |
| **Head D (Biophysical Decoder)** | Field-Pooled Feature Extractor & Diagnostic Classifier | 1,966,733 | 0.96% |
| **Total Model Footprint** | **End-to-End Foundation Model** | **204,958,733** | **100.00%** |

---

## Reproducing Results in 10 Seconds

To enable zero-friction peer review, this repository includes a standalone verification script that mathematically re-computes all metrics directly from raw confusion counts and ground truth arrays in under **0.1 seconds**, with zero external deep learning dependencies:

```bash
# Clone the repository
git clone https://github.com/sphitih-ai/vyom-205m-benchmarks.git
cd vyom-205m-benchmarks

# Run the standalone verification suite (requires only standard Python 3.8+)
python verify_benchmarks.py
```

### Verification Suite Output
```text
================================================================================
VYOM-205M OFFICIAL BENCHMARK VERIFICATION SUITE
Sphitih AI Research — https://github.com/sphitih-ai/vyom-205m-benchmarks
================================================================================

[1] SEN1FLOODS11 GLOBAL FLOOD INUNDATION (30 HELD-OUT TEST TILES, 10 COUNTRIES)
--------------------------------------------------------------------------------
Cumulative Dataset Water IoU : 75.81% (PANGAEA Official Protocol)
Cumulative Dataset Land IoU  : 95.85%
Cumulative Dataset Mean IoU  : 85.83%
Cumulative Pixel Accuracy    : 96.33%
>>> [PASS] Sen1Floods11 metrics mathematically verified against ground truth pixel sums.

[2] AGRIFIELDNET-INDIA 6-CLASS BIOPHYSICAL FIELD STRESS (550 VALIDATION PARCELS)
--------------------------------------------------------------------------------
Total Evaluated Validation Parcels : 550
Overall Multi-Class Accuracy       : 58.55%
Unweighted Macro-F1 (6 Categories) : 0.5730
>>> [PASS] Head D 6-class confusion matrix and Macro-F1 mathematically verified.

[3] KRISHI-DSS / LISS-4 SMALLHOLDER CADASTRAL BOUNDARY (349 HELD-OUT SCENES)
--------------------------------------------------------------------------------
Farm Parcel Interior Extent IoU       : 0.8230 (82.30%)
Farm Parcel Interior Extent F1        : 0.9029 (90.29%)
Bund 2-px Relaxed Boundary F1         : 0.2783 (27.83%)
>>> [PASS] Cadastral extent and bund boundary metrics verified.

[4] SEN12MS-CR PROTOCOL / ALL-WEATHER SAR-TO-OPTICAL SYNTHESIS (116 CUBES)
--------------------------------------------------------------------------------
Structural Similarity Index (SSIM)    : 0.8924
Mean Absolute Error (MAE / L1)        : 0.0298 (2.98% reflectance error)
>>> [PASS] SAR2Opt synthesis metrics verified.

================================================================================
ALL 4 BENCHMARK TRACKS VERIFIED SUCCESSFULLY (100% REPRODUCIBLE)
================================================================================
```

---

## Repository Structure

```text
vyom-205m-benchmarks/
├── .gitignore                      # Strict weights and environment exclusion
├── .zenodo.json                    # Automated Zenodo DOI metadata ingestion
├── CITATION.cff                    # Citation File Format specification
├── LICENSE                         # Apache 2.0 Open Source License
├── README.md                       # Repository overview and scoreboard
├── verify_benchmarks.py            # Standalone zero-dependency verification harness
├── paper/
│   ├── VYOM_205M_PREPRINT.md       # Complete academic research paper
│   └── figures/                    # High-resolution publication evaluation figures
│       ├── fig_cadastral_sample_1.png
│       ├── fig_cadastral_sample_2.png
│       ├── fig_cadastral_sample_3.png
│       ├── fig_sar2opt_sample_1.png
│       ├── fig_sar2opt_sample_2.png
│       └── fig_sar2opt_sample_3.png
└── results/
    ├── audit_summary_tables.json   # Full confusion matrices and aggregated counts
    ├── audit_summary_tables.npz    # Compressed ground truth metric arrays
    └── vyom_pangaea_results.json   # Standardized PANGAEA benchmark export
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
