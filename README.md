# STMN-EQA: Earthquake Anomaly Extraction via Reference Station-Based Transfer Learning

<div align="center">

[![Journal](https://img.shields.io/badge/Journal-Big%20Earth%20Data%20(Accepted)-brightgreen)](https://www.tandfonline.com/journals/tbed20)
[![IF](https://img.shields.io/badge/Impact%20Factor-4.2-blue)](https://www.tandfonline.com/journals/tbed20)
[![JCR](https://img.shields.io/badge/JCR-Q1%20CAS%20Zone%201-red)](https://www.tandfonline.com/journals/tbed20)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-orange)](https://github.com/leanoLEE58/STMN-EQA)

**Big Earth Data (Taylor & Francis) - Accepted for Publication**

[Jiayi Li](mailto:jiayilee@stu.ouc.edu.cn)<sup>1†</sup>, [Zike Ma](mailto:mazike@stu.ouc.edu.cn)</sup>, [Chengquan Chi](mailto:chichengquan@hainnu.edu.cn)<sup>2</sup>, [Haiyong Zheng](mailto:zhenghaiyong@ouc.edu.cn)<sup>1</sup>, [Zining Yu](mailto:yuzining@ouc.edu.cn)<sup>1*</sup>

<sup>1</sup>College of Electronic Engineering, Ocean University of China  
<sup>2</sup>School of Information Science and Technology, Hainan Normal University

<sup>*</sup>Corresponding author

---

### 🎯 Core Innovation

**First framework to apply reference station-based transfer learning for earthquake precursor detection, achieving anomaly extraction without interference from precursor signals by training on seismically inactive regions.**

[📄 Preprint (Coming Soon)](#) · [💻 Code (Coming Soon)](#-code-coming-soon) · [📊 Results](#-experimental-results) · [🗃️ Data](#-data-availability)

</div>

---

## 📑 Table of Contents

<details open>
<summary><b>Click to expand/collapse navigation</b></summary>

- **[📌 Highlights](#-highlights)** - Key contributions and novelty
- **[🔬 Abstract](#-abstract)** - Research overview
- **[🎯 Research Motivation](#-research-motivation)** - Problem and challenges
- **[🏗️ Methodology](#%EF%B8%8F-methodology)** - STMN-EQA framework
  - [Transfer Learning Strategy](#transfer-learning-strategy)
  - [SVMD Frequency Decomposition](#1-svmd-frequency-decomposition)
  - [TimesNet Temporal Modeling](#2-timesnet-temporal-modeling)
  - [GNN Spatial Correlation](#3-gnn-spatial-correlation)
  - [Statistical Analysis](#4-statistical-analysis)
- **[📊 Experimental Results](#-experimental-results)** - Two earthquake case studies
  - [2022 Ms 6.8 Luding Earthquake](#2022-ms-68-luding-earthquake)
  - [2019 Ms 6.0 Changning Earthquake](#2019-ms-60-changning-earthquake)
  - [Performance Comparison](#performance-comparison)
  - [Ablation Studies](#ablation-studies)
- **[💻 Code (Coming Soon)](#-code-coming-soon)** - Implementation details
- **[📚 Citation](#-citation)** - BibTeX entry
- **[🙏 Acknowledgments](#-acknowledgments)** - Funding and data support
- **[📧 Contact](#-contact)** - Get in touch

</details>

---

## 📌 Highlights

### Key Innovations

```
🔬 Reference Station Training       →  Learn normal patterns from seismically inactive regions
🔄 Direct Transfer Learning         →  Zero-shot application to earthquake regions without fine-tuning
📊 Three-Domain Integration         →  Temporal + Frequency + Spatial analysis
🎯 Consistent Precursor Pattern     →  Sigmoidal anomaly accumulation 1-3 months before events
🌍 Multi-Station Validation         →  Verified on two major earthquakes in China
```

### Performance Highlights

| Aspect | Achievement | Significance |
|--------|-------------|--------------|
| **Precursor Timing** | Inflection points at **39±2.2 days** (Luding) and **42±10.9 days** (Changning) before earthquakes | Consistent early warning window |
| **Spatial Correlation** | Power law relationship: **I(r) = 132.53 × r<sup>-0.255</sup>** | Distance-dependent anomaly distribution |
| **Detection Sensitivity** | **Ratio > 1.0** for earthquake periods vs. **0.66-0.73** for normal periods | Strong nonlinear pattern recognition |
| **Model Generalization** | **0% fine-tuning** required for cross-region transfer | Practical deployment advantage |

---

## 🔬 Abstract

**Challenge**: Earthquake precursor extraction from geophysical observation data remains challenging due to diverse signal characteristics and complex earthquake processes.

**Solution**: We propose **STMN-EQA** (Spatiotemporal Multi-scale Network for Earthquake Anomaly Extraction), a novel framework integrating:
- **SVMD** (Segmented Variational Mode Decomposition) for frequency-domain analysis
- **TimesNet** for temporal multi-periodic pattern modeling  
- **GNN** (Graph Neural Networks) for multi-station spatial correlation

**Key Strategy**: We train the model **exclusively on high-quality data from reference stations in seismically inactive regions** to learn normal strain patterns, then **directly transfer** the pre-trained model to analyze borehole strain data for target earthquakes.

**Findings**:
- ✅ **Temporal analysis**: Consistent sigmoidal anomaly accumulation process 1–3 months before both events
- ✅ **Spatial analysis**: Inflection points and anomaly counts correlate with epicentral distance
- ✅ **Superior performance**: Outperforms conventional anomaly detection methods

**Impact**: The reference station-based transfer learning strategy combined with multi-domain analysis offers a novel solution for earthquake anomaly extraction while advancing our understanding of multi-scale earthquake preparation mechanisms.

**Keywords**: Earthquake precursor · Borehole strain · Anomaly extraction · Deep learning · Transfer learning · Spatiotemporal analysis

---

## 🎯 Research Motivation

### Monitoring Network Overview

<div align="center">

<img src="https://github.com/user-attachments/assets/7480e002-7a18-40bf-83e2-ab2266815fab" width="85%" alt="Monitoring Network"/>

**Figure 1**: Overview of the studied region and monitoring network. Map shows Sichuan-Yunnan monitoring network with earthquake epicenters (red dots for Ms ≥ 3.0 events), target earthquakes (yellow stars), station locations (triangles), and theoretical influence radii (white dashed circles): 839.5 km for Ms 6.8 Luding earthquake and 380 km for Ms 6.0 Changning earthquake. Reference station 62003 location shown in inset (Gansu Province, >1000 km from both epicenters).

</div>

### Station Quality Assessment

<div align="center">

<img src="https://github.com/user-attachments/assets/2f1c5453-2fc1-462b-8590-03adeadb0a0e" width="90%" alt="Station Tables"/>

**Table 1 & 2**: (Top) Detailed information of borehole strain monitoring stations including coordinates, rock types, and sampling frequencies. (Bottom) Quality assessment showing data continuity, self-consistency coefficients, and earthquake activity within 50/200 km radii. **Station 62003** (highlighted) selected as reference station due to 99.5% continuity, highest self-consistency (k=0.98), and zero Ms≥3.0 earthquakes within 50 km.

</div>

### Problem Statement

**Traditional approaches face critical limitations**:

<div align="center">

| Challenge | Traditional Method | Impact |
|-----------|-------------------|---------|
| **🚫 Signal Interference** | Training on earthquake regions | Precursor signals contaminate "normal" baseline |
| **🚫 Single-Domain Analysis** | Frequency OR temporal OR spatial | Misses multi-scale earthquake preparation processes |
| **🚫 Single-Station Focus** | Individual station analysis | Overlooks systemic crustal deformation patterns |
| **🚫 Complex Tectonic Noise** | No explicit spatial modeling | High false positive rates in active zones |

</div>

### Our Innovation

**Paradigm Shift**: Learn "normal" from seismically quiet regions → Apply to earthquake zones

```
Problem 1: Baseline Contamination    →  Reference Station Training (>1000 km from epicenters)
Problem 2: Multi-Scale Processes     →  Frequency (SVMD) + Temporal (TimesNet) + Spatial (GNN)
Problem 3: Systemic Patterns         →  Graph Neural Networks for multi-station correlation
Problem 4: Spatial Localization      →  Distance-weighted GNN with epicentral distance encoding
```

---

## 🏗️ Methodology

### Overall Framework

<div align="center">

<img src="https://github.com/user-attachments/assets/833939c9-0a1d-4bbe-966b-9dde6cd7519f" width="100%" alt="STMN-EQA Framework"/>

**Figure 2**: STMN-EQA architecture. **(Left to Right)**: (1) **SVMD Module** - decomposes strain signals into interpretable frequency components (IMF3-5); (2) **TimesNet Module** - extracts multi-periodic temporal patterns via 2D tensor transformation; (3) **GNN Module** - captures spatial correlations across monitoring network; (4) **Statistical Analysis** - identifies sigmoidal anomaly accumulation patterns.

</div>

---

### Transfer Learning Strategy

**Core Concept**: Avoid precursor signal interference during training

**Mathematical Formulation**:

**Training Phase** (Reference Station 62003 - Gansu Province, 1142 km from Luding epicenter):
```math
\hat{X}_{\text{ref},i} = \text{STMN}(X_{\text{ref},i}; \theta)
```

**Optimization** (Mean Squared Error):
```math
L_{\text{MSE}} = \frac{1}{N_{\text{ref}}} \sum_{i=1}^{N_{\text{ref}}} |\hat{X}_{\text{ref},i} - X_{\text{ref},i}|^2
```

**Transfer Phase** (Target Stations - Direct application without fine-tuning):
```math
\hat{X}_{\text{target},i} = \text{STMN}(X_{\text{target},i}; \theta^*)
```

**Anomaly Detection**:
```math
A_{\text{target}} = \text{Det}(|\hat{X}_{\text{target},i} - X_{\text{target},i}|)
```

**Key Advantage**: θ<sup>*</sup> learned from seismically inactive region → No precursor contamination in baseline

---

### 1. SVMD Frequency Decomposition

**Challenge**: Multi-year continuous strain data (1-minute sampling) → Computational inefficiency

**Solution**: Sliding window segmentation + Variational Mode Decomposition

**Implementation**:
- **Window size**: 7 days
- **Step size**: 1 day
- **Selected modes**: IMF3, IMF4, IMF5 (exclude tidal/long-term trends in IMF1-2)

**Physical Interpretation**:
- **IMF3**: Medium-term deformation processes (weeks to months)
- **IMF4**: Stress changes and fluid migration (days to weeks)  
- **IMF5**: High-frequency signals related to microcrack formation (hours to days)

---

### 2. TimesNet Temporal Modeling

**Innovation**: Transform 1D time series → 2D tensors to capture intraperiod and interperiod variations

<div align="center">

<img src="https://github.com/user-attachments/assets/c9ab9070-e889-4d34-ad82-ee453f0265ad" width="95%" alt="TimesNet Architecture"/>

**Figure 3**: TimesNet architecture for multi-periodic temporal modeling. Identifies dominant periods via FFT (P₁, P₂, P₃), reshapes 1D sequences into 2D tensors, processes through multi-scale inception blocks, and aggregates across periods for comprehensive pattern recognition.

</div>

**Mathematical Pipeline**:

**Period Detection** (Fast Fourier Transform):
```math
\mathcal{P}(X) = \{P_1, P_2, \ldots, P_k\} = \text{TopK}(|\text{FFT}(X)|)
```

**2D Reshaping** (for each period P<sub>i</sub>):
```math
X_{P_i} = \text{Reshape}(X, [L/P_i, P_i, D])
```
- Rows (L/P<sub>i</sub>): Number of complete periods (interperiod variation)
- Columns (P<sub>i</sub>): Period length (intraperiod variation)

**Multi-Scale Feature Extraction** (Inception blocks):
```math
Z_{P_i} = \text{Concat}\left(\{\text{Conv}_{k \times k}(X_{P_i})\}_{k \in \{1,3,5,7\}}\right) \cdot W_{\text{proj}}
```

**Period Aggregation**:
- Amplitude-weighted fusion of features from P₁, P₂, P₃
- Captures complex temporal patterns across different scales

---

### 3. GNN Spatial Correlation

**Challenge**: Earthquake preparation is a systemic process affecting multiple stations

**Solution**: Graph representation with distance-weighted message passing

<div align="center">

<img src="https://github.com/user-attachments/assets/7d13de8d-cfb6-4498-b8a7-9f7b1e184ab4" width="75%" alt="GNN Architecture"/>

**Figure 4**: GNN architecture for spatial correlation analysis. Nodes = monitoring stations; Edges = distance-weighted connections; Node features = TimesNet-processed strain signals. Two-layer graph convolution with normalized adjacency weights captures multi-station spatial dependencies.

</div>

**Graph Definition**: G = (V, E, X)
- **Nodes (V)**: Individual monitoring stations
- **Edges (E)**: Weighted by geographical distances
- **Node features (X)**: Predicted strain signals from TimesNet

**Two-Layer Graph Convolution**:

**Layer 1** (with ReLU activation):
```math
h_i^{(1)} = \text{ReLU}\left(\sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{\sum_k w_{ik}} W^{(1)} h_j^{(0)}\right)
```

**Layer 2** (final spatially-enhanced features):
```math
h_i^{(2)} = \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{\sum_k w_{ik}} W^{(2)} h_j^{(1)}
```

**Distance-Based Weight Design** (Gaussian kernel):
```math
W_{ij}^{(1)} = \begin{cases}
\exp\left(-\frac{d_i^2}{2\tau^2}\right) & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}
```
- d<sub>i</sub>: Epicentral distance of station i
- τ = 200 km (empirical scale parameter)
- W<sup>(2)</sup> = **I** (identity matrix, preserving weighted features)

**Advantages**:
- ✅ **Fixed weights**: Preserve physical distance relationships, avoid overfitting
- ✅ **Interpretable**: Direct correspondence to epicentral distance
- ✅ **No fine-tuning needed**: Construct weights directly from station locations

---

### 4. Statistical Analysis

**Anomaly Threshold** (confidence interval-based):

**Upper/Lower Bounds**:
```math
\text{lower}_t = \hat{y}_t - Z_\alpha \cdot \sigma_e, \quad \text{upper}_t = \hat{y}_t + Z_\alpha \cdot \sigma_e
```
- ŷ<sub>t</sub>: Predicted value at time t
- σ<sub>e</sub>: Error standard deviation
- Z<sub>α</sub> = 1.5 (confidence multiplier)

**Sigmoidal Fitting** (cumulative anomaly counts):
```math
S(t; a, b, c, d) = \frac{a}{1 + e^{-b(t-c)}} + d
```
- **a**: Maximum cumulative anomaly count
- **b**: Transition phase steepness
- **c**: Inflection point timing (critical for early warning)
- **d**: Background anomaly baseline

**Model Validation** (Sigmoidal vs. Linear):
```math
\text{Ratio} = \frac{R^2_{\text{S-curve}}}{R^2_{\text{linear}}}
```
- **Ratio > 1.0**: Nonlinear sigmoidal pattern (earthquake-related)
- **Ratio ≈ 0.66-0.73**: Linear trend (random fluctuations)

---

## 📊 Experimental Results

### 2022 Ms 6.8 Luding Earthquake

**Epicenter**: 29.59°N, 102.08°E  
**Depth**: 16 km  
**Monitoring Network**: 5 stations (23.5 km to 353.0 km from epicenter)

#### Temporal Analysis: Sigmoidal Anomaly Accumulation

<div align="center">

<img src="https://github.com/user-attachments/assets/a006f184-1594-479c-8c11-cfd16b9331c2" width="100%" alt="Luding Sigmoidal Analysis"/>

**Figure 5**: Sigmoidal pattern analysis for Ms 6.8 Luding earthquake. **(Left)**: Anomaly accumulation for all 5 stations showing transition from linear (3-6 months before) to sigmoidal (1-3 months before). **(Right)**: Detailed fitting for Kangding station (closest to epicenter, 23.5 km) with inflection point at **39±2.2 days** before the earthquake. Red curve shows sigmoidal fit (R²=0.968), demonstrating clear nonlinear acceleration phase.

</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/8207edd5-c75d-4186-adac-afbe4a752a3e" width="85%" alt="Table 3"/>

**Table 3**: Statistical comparison of sigmoidal fitting characteristics across monitoring stations and random periods. All earthquake-related Ratio values (1.11-1.27) significantly exceed random baseline (0.66-0.73, shown in gray rows), confirming genuine precursor patterns. R²<sub>S-curve</sub> values generally decrease with epicentral distance.

</div>

**Key Findings**:

| Station | Distance (km) | R²<sub>S-curve</sub> | Ratio | Inflection Point (days) |
|---------|---------------|----------------------|-------|------------------------|
| **51010 (Kangding)** | 23.5 | **0.968** | **1.27** | 39±2.2 |
| **51304 (Jinhe)** | 94.7 | 0.942 | 1.20 | 37±2.5 |
| **51009 (Xiaomiao)** | 199.2 | 0.881 | 1.11 | 34±3.1 |
| 53022 (Zhaotong) | 309.1 | 0.951* | 1.25 | 41±2.8 |
| 53006 (Yongsheng) | 353.0 | 0.915* | 1.18 | 38±3.2 |
| **Random Baseline** | N/A | 0.651-0.704 | 0.66-0.73 | N/A |

*Stations 53022 and 53006 showed interference from concurrent local earthquakes (Ms 4.4 Weining and Ms 4.6 Ninglang)

**Statistical Significance**: All earthquake-related Ratio values (1.11-1.27) significantly exceed random baseline (0.66-0.73), confirming genuine precursor patterns.

---

#### Spatial Analysis: Distance-Dependent Patterns

<div align="center">

<img src="https://github.com/user-attachments/assets/a9a3f02e-643a-491e-b359-d7ccef9ddd72" width="100%" alt="Luding Spatial Analysis"/>

**Figure 6**: Spatial-temporal characteristics for Luding earthquake. **(Left)**: Inflection point timing shows weak distance dependence (y = 38.56 - 0.0079x, R² = 0.163, p = 0.070), with consistent timing of **37.0±2.7 days** across 23.5-353.0 km range. **(Right)**: Cumulative anomaly counts follow power law: **I(r) = 132.53 × r<sup>-0.255±0.090</sup>** (R² = 0.719, p = 0.070), indicating strain anomaly intensity decreases with epicentral distance. Markers with error bars denote stations affected by concurrent local earthquakes.

</div>

**Physical Interpretation**:
- **Inflection point consistency** (34-41 days): Suggests synchronized stress release initiation across monitoring network
- **Power law decay**: Indicates strain anomaly intensity decreases with distance from epicenter
- **Exponent -0.255**: Slower decay than classical point source models (typically -1), possibly reflecting distributed fault zone effects

---

### 2019 Ms 6.0 Changning Earthquake

**Epicenter**: 28.34°N, 104.90°E  
**Depth**: 1.6 km (extremely shallow, thrust + strike-slip faulting)  
**Monitoring Network**: 5 stations (within 380 km effective radius)

#### Consistent Sigmoidal Pattern

<div align="center">

<img src="https://github.com/user-attachments/assets/8c2da755-8914-439c-9f14-1921d3f843ed" width="100%" alt="Changning Sigmoidal Analysis"/>

**Figure 7**: Ms 6.0 Changning earthquake analysis. **(Left)**: Anomaly accumulation for 5 stations. Station 53006 (474.5 km, outside 380 km effective radius) shows linear trend. Other stations exhibit sigmoidal acceleration 1-3 months before event. **(Right)**: Detailed fitting for Luzhou station (57.8 km) with inflection point at **40±2.1 days** before the earthquake, consistent with Luding earthquake findings.

</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/d4925e51-4d52-47a9-87d1-58915321a5f5" width="100%" alt="Changning Spatial Analysis"/>

**Figure 8**: Spatial characteristics for Changning earthquake. **(Left)**: Inflection point timing within 380 km effective radius shows approximate linear relationship with distance. **(Right)**: Spatial distribution map with power law fitting. Square marker denotes station 53006 (outside effective radius) affected by local Ms 3.4 earthquake interference. Circle sizes proportional to cumulative anomaly counts.

</div>

**Findings**:

| Station | Distance (km) | R²<sub>S-curve</sub> | Ratio | Inflection Point (days) |
|---------|---------------|----------------------|-------|------------------------|
| **51002 (Luzhou)** | 57.8 | **0.956** | **1.31** | 40±2.1 |
| 53022 (Zhaotong) | 158.4 | 0.948* | 1.28 | 42±3.5 |
| 51009 (Xiaomiao) | 186.7 | 0.923 | 1.22 | 45±4.2 |
| 51304 (Jinhe) | 242.1 | 0.897 | 1.15 | 38±5.1 |
| 53006 (Yongsheng) | 474.5** | 0.685 | 0.82 | N/A (linear) |

*Station 53022 affected by Ms 4.7 Yongshan earthquake (May 16, 2019)  
**Station 53006 outside 380 km effective radius + Ms 3.4 local interference (July 21, 2019)

**Cross-Validation**: Mean inflection timing of **42.3±10.9 days** (4 stations within effective radius) consistent with Luding earthquake (39±2.2 days)

---

### Performance Comparison

#### TimesNet vs. Other Time Series Models

**Experimental Setup**:
- **Training**: Reference Station 62003 (Jan 2019 - Dec 2022)
- **Testing**: Earthquake period (Mar-Oct 2022) + Normal period (Jan-Aug 2023)
- **Metrics**: MSE/MAE (lower = better prediction during normal; higher = better anomaly sensitivity during earthquake), R² (higher = better fit), Ratio (>1 = nonlinear pattern detection)

<div align="center">

<img src="https://github.com/user-attachments/assets/1cbac3bb-93aa-4e99-adb9-f3a04147c09e" width="90%" alt="Table 4"/>

**Table 4**: Performance comparison across time series models (mean ± std over 30 independent runs, ** p < 0.01 vs. others using paired t-tests). TimesNet (highlighted) achieves superior performance across all metrics during both normal and earthquake periods.

</div>

**Key Observations**:
- ✅ **Normal periods**: TimesNet achieves **lowest MSE (0.412±0.016)** → Best baseline prediction
- ✅ **Earthquake periods**: TimesNet achieves **highest MSE (1.878±0.067)** → Best anomaly sensitivity
- ✅ **Ratio metric**: **1.27±0.09** (earthquake) vs. **0.82±0.08** (normal) → Clear nonlinear pattern distinction

**Advantage**: TimesNet's 2D tensor transformation effectively captures multi-periodic earthquake preparation signals

---

### Ablation Studies

#### Domain Integration Analysis

**Research Question**: Do frequency, temporal, and spatial domains contribute independently to anomaly detection?

<div align="center">

<img src="https://github.com/user-attachments/assets/bdb91a4b-c8f0-4a69-a944-09ff402070f1" width="95%" alt="Table 5"/>

**Table 5**: Domain contribution analysis across earthquake and normal periods (** p < 0.01, * p < 0.05 vs. temporal-only baseline using paired t-tests over 30 runs). Progressive integration of frequency and spatial domains yields cumulative performance improvements.

</div>

**Findings**:

| Configuration | Normal Period MSE ↓ | Earthquake Period MSE ↑ | Ratio ↑ | Spatial Correlation ↑ |
|---------------|---------------------|-------------------------|---------|----------------------|
| Temporal only | 0.623±0.025 | 0.847±0.021 | 0.89±0.06 | N/A |
| + Frequency | **0.434±0.018** ✓ | 1.082±0.015** | 1.15±0.08** | N/A |
| + Spatial | 0.645±0.022 | 0.998±0.019* | 0.95±0.07 | 0.43±0.08** |
| **Ours (All)** | **0.412±0.016** ✓✓ | **1.247±0.013** ✓✓ | **1.27±0.09** ✓✓ | **0.67±0.06** ✓✓ |

**Conclusion**: Each domain provides complementary information:
- **Frequency (SVMD)**: Separates tidal/long-term trends from precursor signals (+30.3% MSE reduction)
- **Spatial (GNN)**: Validates anomalies across network (+55.6% spatial correlation)
- **Integration**: Best overall performance in both prediction accuracy and anomaly sensitivity

---

#### Fixed vs. Trainable GNN Weights

**Research Question**: Are distance-based fixed weights better than learned weights for spatial correlation?

<div align="center">

<img src="https://github.com/user-attachments/assets/0ca8709b-5a84-4bb2-8e62-3a25bdc14ecb" width="55%" alt="Table 6a"/>

<img src="https://github.com/user-attachments/assets/c2f2952a-0e2a-487b-9b5b-4dbf0957f852" width="60%" alt="Table 6b"/>

<img src="https://github.com/user-attachments/assets/a93525a1-bf67-475c-8a52-dce2998cdc7b" width="65%" alt="Table 6c"/>

**Table 6**: Fixed vs. trainable weight strategies comparison (** p < 0.01, 30 runs with different random seeds). Fixed distance-based weights achieve significantly better spatial correlation (0.67±0.05 vs. 0.52±0.08) and sigmoidal pattern recognition (Ratio 1.27±0.09 vs. 1.09±0.12).

</div>

**Results**:

| Weight Strategy | Spatial Correlation ↑ | Ratio (Kangding Station) ↑ |
|-----------------|----------------------|----------------------------|
| Trainable | 0.52±0.08 | 1.09±0.12 |
| **Fixed (Ours)** | **0.67±0.05** ✓✓ | **1.27±0.09** ✓✓ |

**Advantages of Fixed Weights**:
- ✅ **Physical interpretability**: Directly encode epicentral distance relationships
- ✅ **Avoid overfitting**: No risk of learning spurious spatial patterns
- ✅ **Transfer efficiency**: Immediate deployment without retraining for new networks

---

## 💻 Code (Coming Soon)

### 🚧 Repository Under Construction

**Planned Structure**:

```
STMN-EQA/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── configs/
│   ├── reference_station.yaml        # Training config for Station 62003
│   ├── luding_earthquake.yaml        # Testing config for Luding earthquake
│   └── changning_earthquake.yaml     # Testing config for Changning earthquake
│
├── models/
│   ├── __init__.py
│   ├── svmd.py                       # Segmented VMD implementation
│   ├── timesnet.py                   # Multi-periodic temporal modeling
│   ├── gnn_spatial.py                # Graph neural network for spatial correlation
│   ├── stmn_eqa.py                   # Main STMN-EQA framework
│   └── transfer_learning.py          # Reference station-based transfer strategy
│
├── data/
│   ├── preprocessing.py              # Strain data preprocessing pipeline
│   ├── quality_assessment.py         # Self-consistency & continuity checks
│   └── station_network.py            # Multi-station graph construction
│
├── utils/
│   ├── metrics.py                    # MSE, MAE, R², Ratio calculations
│   ├── sigmoidal_fitting.py          # Anomaly accumulation curve fitting
│   ├── spatial_analysis.py           # Power law & distance correlation
│   └── visualization.py              # Result plotting functions
│
├── scripts/
│   ├── train_reference_station.py    # Train on Station 62003
│   ├── test_earthquake.py            # Evaluate on target earthquakes
│   ├── ablation_study.py             # Domain contribution analysis
│   └── compare_baselines.py          # Benchmark against other methods
│
└── notebooks/
    ├── tutorial_data_preprocessing.ipynb
    ├── tutorial_model_training.ipynb
    └── tutorial_anomaly_analysis.ipynb
```

---

### 📦 Expected Dependencies

```python
# Deep Learning
torch>=1.11.0
tensorflow>=2.8.0

# Scientific Computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# Geophysical Analysis
obspy>=1.3.0            # Seismic data processing
pyvmd>=0.1.0            # Variational Mode Decomposition

# Graph Neural Networks
torch-geometric>=2.0.0
networkx>=2.6.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
```

---

### 🎯 Quick Start (Preview)

#### Training on Reference Station

```bash
# Train STMN-EQA on Station 62003 (seismically inactive region)
python scripts/train_reference_station.py \
    --config configs/reference_station.yaml \
    --station_id 62003 \
    --train_start 2019-01-01 \
    --train_end 2022-12-31 \
    --gpu 0
```

#### Transfer to Earthquake Region

```bash
# Directly apply trained model to Luding earthquake (no fine-tuning)
python scripts/test_earthquake.py \
    --config configs/luding_earthquake.yaml \
    --checkpoint checkpoints/stmn_eqa_reference.pth \
    --test_start 2022-03-01 \
    --test_end 2022-10-31 \
    --output_dir results/luding
```

#### Anomaly Analysis

```bash
# Extract sigmoidal patterns and spatial correlations
python scripts/analyze_anomalies.py \
    --results_dir results/luding \
    --fit_sigmoid \
    --spatial_analysis \
    --plot_output figures/luding_analysis.png
```

---

### 📥 Pretrained Models (Coming Soon)

| Model | Training Data | Parameters | Download |
|-------|--------------|------------|----------|
| STMN-EQA-Ref | Station 62003 (2019-2022) | ~500K | [Google Drive](#) \| [Baidu Pan](#) |

**Expected file size**: ~50 MB

---

### 🔬 Reproducing Paper Results

**Step 1: Download Earthquake Catalog**
```bash
# China Earthquake Network Center data
wget http://data.earthquake.cn/catalog/luding_2022.csv
wget http://data.earthquake.cn/catalog/changning_2019.csv
```

**Step 2: Prepare Borehole Strain Data**
```bash
# Contact corresponding author for data access
# Email: yuzining@ouc.edu.cn
# Subject: "STMN-EQA Data Request"
```

**Step 3: Run Full Pipeline**
```bash
# Automated reproduction script
bash scripts/reproduce_paper_results.sh
```

**Expected Output**:
- Sigmoidal fitting parameters (Table 3 in paper)
- Spatial correlation plots (Figure 6 & 8 in paper)
- Performance comparison tables (Table 4 & 5 in paper)

---

## 📚 Citation

**Journal Article** (Once published, will update with DOI):

```bibtex
@article{li2025stmneqa,
  title={Reference station-based transfer learning for earthquake anomaly extraction 
         from borehole strain data related to two earthquakes in China},
  author={Li, Jiayi and Ma, Zike and Chi, Chengquan and Zheng, Haiyong and Yu, Zining},
  journal={Big Earth Data},
  year={2025},
  publisher={Taylor \& Francis},
  note={Accepted for publication}
}
```

**Preprint** (Once available):

```bibtex
@misc{li2025stmneqa_preprint,
  title={Reference station-based transfer learning for earthquake anomaly extraction 
         from borehole strain data related to two earthquakes in China},
  author={Li, Jiayi and Ma, Zike and Chi, Chengquan and Zheng, Haiyong and Yu, Zining},
  year={2025},
  eprint={arXiv:XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={physics.geo-ph}
}
```

---

## 🙏 Acknowledgments

### Funding Support

This work was supported by:
- **National Natural Science Foundation of China** (Grants No. 42204005, 62171421)
- **Natural Science Foundation of Shandong Province** (Grant No. ZR2022QF130)
- **TaiShan Scholars Youth Expert Program of Shandong Province** (Grant No. tsqn202306096)
- **Hainan Provincial Natural Science Foundation** (Grant No. 622RC669)

### Data Support

We acknowledge the borehole strain data support from:
- **China Earthquake Networks Center**
- **National Earthquake Data Center**
- **National Institute of Natural Hazards, MEMC**

### Technical Acknowledgments

We thank the authors of the following works for open-sourcing their implementations:
- [TimesNet](https://github.com/thuml/Time-Series-Library) - Multi-periodic temporal modeling
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - Graph neural networks
- [VMD-Python](https://github.com/vrcarva/vmdpy) - Variational mode decomposition

---

## 🗃️ Data Availability

### Earthquake Catalog Data

**Publicly available** from China Earthquake Network:
- **URL**: http://data.earthquake.cn
- **Coverage**: Earthquakes with Ms ≥ 3.0 in China
- **Format**: CSV with epicenter coordinates, magnitude, depth, and timing

### Borehole Strain Data

**Restricted access** (requires institutional agreement):
- **Source**: National Institute of Natural Hazards, MEMC
- **Instrumentation**: YRY-4 four-component borehole strainmeters
- **Sampling rate**: 1 sample/minute
- **Quality criteria**: Data continuity >90%, self-consistency coefficient k ≥ 0.95

**Data Request Procedure**:
1. Email corresponding author at yuzining@ouc.edu.cn
2. Provide research affiliation and intended use
3. Sign data usage agreement
4. Receive access credentials within 2-4 weeks

**Note**: Reference Station 62003 data may be shared for reproduction purposes upon reasonable request.

---

## 📧 Contact

### Corresponding Author

**Zining Yu, Ph.D.**  
Associate Professor  
College of Electronic Engineering  
Ocean University of China  
Qingdao, Shandong 266100, China

📧 Email: yuzining@ouc.edu.cn  
🔬 ORCID: [0000-0003-0888-9062](https://orcid.org/0000-0003-0888-9062)  
🏫 Lab: Geophysical Signal Processing Research Group

### First Authors

**Jiayi Li** (Undergraduate Student)  
📧 Email: jiayilee@stu.ouc.edu.cn  
🔬 ORCID: [0009-0009-9854-9694](https://orcid.org/0009-0009-9854-9694)  
🎓 Research Focus: Deep learning, signal processing, earthquake precursor detection


### Co-Authors

**Zike Ma** (Undergraduate Student)  
📧 Email: mazike@stu.ouc.edu.cn  
🎓 Research Focus: Data analysis, geophysical information processing

**Chengquan Chi, Ph.D.**  
Associate Professor  
School of Information Science and Technology  
Hainan Normal University  
📧 Email: chichengquan@hainnu.edu.cn  
🔬 ORCID: [0000-0002-8561-0275](https://orcid.org/0000-0002-8561-0275)

**Haiyong Zheng, Ph.D.**  
Professor  
College of Electronic Engineering  
Ocean University of China  
📧 Email: zhenghaiyong@ouc.edu.cn  
🔬 ORCID: [0000-0002-8027-0734](https://orcid.org/0000-0002-8027-0734)

---

### For Inquiries

- 🐛 **Bug Reports / Technical Issues**: [GitHub Issues](https://github.com/leanoLEE58/STMN-EQA/issues) (Once repository is public)
- 💬 **Research Collaboration**: Email Dr. Zining Yu (yuzining@ouc.edu.cn)
- 📊 **Data Access Requests**: Use subject line "STMN-EQA Data Request"
- 📰 **Media / Press**: Contact Ocean University of China Press Office

---

<div align="center">

### ⭐ If you find this work helpful, please cite our paper! ⭐

**Advancing earthquake precursor detection through transfer learning and multi-domain analysis**

---

**Last Updated**: January 2025 | **Status**: Accepted at Big Earth Data (Taylor & Francis)

[⬆️ Back to Top](#stmn-eqa-earthquake-anomaly-extraction-via-reference-station-based-transfer-learning)

</div>

---

## 📌 Related Publications

### By This Research Group

**Earthquake Precursor Analysis**:
- Yu et al. (2021). "Evaluation of pre-earthquake anomalies of borehole strain network by using receiver operating characteristic curve." *Remote Sensing*, 13(3), 515.
- Yu et al. (2024). "The study on anomalies of the geomagnetic topology network associated with the 2022 MS 6.8 Luding earthquake." *Remote Sensing*, 16(9).
- Chi et al. (2023). "Pre-earthquake anomaly extraction from borehole strain data based on machine learning." *Scientific Reports*, 13, 20095.

**Signal Processing Methods**:
- Chi et al. (2019). "Detecting earthquake-related borehole strain data anomalies with variational mode decomposition and principal component analysis." *IEEE Access*, 7, 157997-158006.
- Zhu et al. (2019). "Negentropy anomaly analysis of the borehole strain associated with the MS 8.0 Wenchuan earthquake." *Nonlinear Processes in Geophysics*, 26(4), 371-380.

### Recommended Reading

**Deep Learning for Seismology**:
- Rouet-Leduc et al. (2020). "Probing slow earthquakes with deep learning." *Geophysical Research Letters*, 47(4).
- McBrearty & Beroza (2023). "Earthquake phase association with graph neural networks." *Bulletin of the Seismological Society of America*, 113(2), 524-547.

**Earthquake Preparation Mechanisms**:
- Dobrovolsky et al. (1979). "Estimation of the size of earthquake preparation zones." *Pure and Applied Geophysics*, 117, 1025-1044.
- Ma & Guo (2014). "Accelerated synergism prior to fault instability: Evidence from laboratory experiments and an earthquake case." *Seismology and Geology*, 36(3), 547-561.

---

**🔬 Contributions to Open Science**

This research adheres to **FAIR principles** (Findable, Accessible, Interoperable, Reusable):
- ✅ **Code**: Open-source implementation (MIT License) - Coming Soon
- ✅ **Data**: Earthquake catalogs publicly available; strain data accessible upon request
- ✅ **Methods**: Detailed mathematical formulations and hyperparameters provided
- ✅ **Results**: Raw experimental outputs and figures included in repository

**We encourage researchers to build upon this work and extend it to other tectonic settings! 🌍**
