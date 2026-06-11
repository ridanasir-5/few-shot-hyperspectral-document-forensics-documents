# Few-Shot-Hyperspectral-Document-Forensics
Code for "Few-Shot Hyperspectral Representation Learning for Multi-Task Forensic Document Analysis"

Spectral--spatial representation learning framework for hyperspectral forensic document analysis using episodic prototypical learning, supporting writer identification, writer verification, ink mismatch detection, forgery detection, and demographic inference.

# Few-Shot Hyperspectral Representation Learning for Multi-Task Forensic Document Analysis

This repository contains the implementation, preprocessing pipeline, evaluation framework, representative metadata, and experimental materials associated with the research study:

## "Few-Shot Hyperspectral Representation Learning for Multi-Task Forensic Document Analysis"

The proposed framework investigates whether shared spectral--spatial representations learned through episodic prototypical optimization can support diverse forensic document analysis tasks under few-shot settings and across different hyperspectral datasets.

---

# Repository Overview

This repository provides:

* Hyperspectral preprocessing and tensor generation workflows
* Episodic sampling and pair generation procedures
* Few-shot prototypical training notebooks
* Multi-task forensic evaluation pipelines
* Cross-dataset and zero-shot evaluation frameworks
* Ablation study implementations
* ROC curves and confusion matrix visualizations
* Representative metadata samples
* Experimental outputs and performance analyses

---

# Repository Structure

```text
few-shot-hyperspectral-document-forensics/
│
├── code/
│   ├── hyperspectral_forensics_pipeline_v3.py
│   ├── hyperspectral_forensics_pipeline_v4.py
│   ├── hyperspectral_forensics_ablation.py
│   └── hyperspectral_forensics_ablation_uwa.py
│
├── checkpoints/
│   ├── best_protonet_crosssplit.pt
│   └── best_protonet_uwa.pt
│
├── results/
│   ├── ivision_hhid/
│   ├── uwa_wihsi/
│   ├── zero_shot/
│   └── ablation/
│
├── figures/
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── CITATION.cff
```

```

```


---

# Included Materials

The repository currently includes:

* Dataset indexing notebooks
* Hyperspectral preprocessing pipelines
* Tensor generation procedures
* Episodic sampling and pair generation frameworks
* Prototypical network training implementations
* Evaluation and testing notebooks
* Cross-dataset and zero-shot evaluation scripts
* ROC curve generation
* Confusion matrix visualization
* Precision--recall analysis
* Representative metadata examples
* Ablation study outputs
* Experimental performance summaries

---

# Methodology Summary

The proposed framework combines hyperspectral document imaging with episodic prototypical learning to investigate transferable spectral--spatial representations for forensic analysis.

The workflow includes:

1. Hyperspectral handwriting acquisition
2. Spectral preprocessing and tensor generation
3. Episodic support-query construction
4. Few-shot prototypical training
5. Multi-task forensic evaluation and visualization

The framework supports:

* Writer identification
* Writer verification
* Ink mismatch detection
* Forgery detection
* Gender prediction
* Age estimation

---

# Dataset Availability

Experiments were conducted using the UWA-WIHSI and iVision HHID hyperspectral datasets.

Due to storage requirements and dataset redistribution restrictions, the complete datasets are not directly hosted within this repository. Representative metadata samples, preprocessing examples, generated outputs, and evaluation resources are publicly provided.

Researchers interested in accessing the original datasets should obtain them from their respective providers and follow the preprocessing procedures included in this repository.

---

# Reproducibility

All experiments were conducted using fixed preprocessing configurations, controlled episodic sampling strategies, and standardized evaluation settings to support reproducibility.

Representative notebooks, evaluation scripts, metadata samples, and generated outputs are included to facilitate verification of the reported methodology and findings.

---

# Code Availability

The repository provides access to:

* Hyperspectral preprocessing pipelines
* Tensor generation workflows
* Episodic sampling procedures
* Prototypical network implementations
* Training notebooks
* Evaluation frameworks
* Cross-dataset testing scripts
* Zero-shot evaluation procedures
* Ablation analyses
* Visualization resources

---

# Installation

Recommended environment:

* Python 3.10+
* Jupyter Notebook
* CUDA-enabled GPU (recommended for training)

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision jupyter opencv-python scipy seaborn tqdm spectral
```

---

# Usage

Typical workflow:

1. Run dataset indexing notebook
2. Generate hyperspectral tensors
3. Construct episodic samples
4. Train the prototypical framework
5. Perform evaluation and visualization

Suggested execution order:

### Suggested Execution Order

```text
1. hyperspectral_forensics_pipeline_v3.py
   ↓
   Data preprocessing, episodic sampling, model training, and evaluation
   on the iVision HHID dataset.

2. hyperspectral_forensics_pipeline_v4.py
   ↓
   Data preprocessing, episodic sampling, model training, and evaluation
   on the UWA-WIHSI dataset.

3. best_protonet_crosssplit.pt / best_protonet_uwa.pt
   ↓
   Load the best-performing checkpoints for reproducibility and
   inference without retraining.

4. hyperspectral_forensics_ablation.py
   ↓
   Perform ablation studies on the iVision HHID dataset to investigate
   the influence of architectural and optimization choices.

5. hyperspectral_forensics_ablation_uwa.py
   ↓
   Perform ablation studies on the UWA-WIHSI dataset and compare the
   effects of different design decisions under controlled conditions.

6. Review the generated outputs in:
   ↓
   results/ivision_hhid/
   results/uwa_wihsi/
   results/zero_shot/
   results/ablation/
```


---

# Experimental Outputs

The repository includes representative:

* ROC curves
* Precision--recall curves
* Confusion matrices
* Prototype similarity analyses
* Score distribution visualizations
* Zero-shot evaluation outputs
* Ablation study visualizations

These outputs support the findings reported in the associated manuscript.

---

# Citation

If you use this repository or associated materials in your research, please cite the corresponding manuscript.

```text
R. Nasir et al.,
"Few-Shot Hyperspectral Representation Learning for Multi-Task Forensic Document Analysis,"
submitted for publication.
```

---

# License

This project is distributed under the MIT License.

---

# Contact

For questions regarding datasets, experimental materials, or reproducibility resources, please contact the corresponding author. 
edit the reposity structure accordinly
