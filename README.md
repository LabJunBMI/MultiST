# MultiST: A Cross-Attention–Based Multimodal Model for Spatial Transcriptomics

---

## Implementation

MultiST is implemented in the **PyTorch** framework and supports **GPU acceleration via CUDA**.
For optimal performance, we recommend running MultiST on a CUDA-enabled GPU.

---

## Requirements

To run all components in this repository, the following environment is required:

```text
python==3.9.13
numpy==1.24.3
scipy==1.10.1
pandas==1.5.3
scikit-learn==1.2.2
tqdm==4.65.0
scanpy==1.9.3
anndata==0.9.1
matplotlib==3.7.1
seaborn==0.12.2
rpy2==3.5.12

torch==2.1.0+cu118
torchvision==0.16.2+cu118
torchaudio==2.1.0+cu118
torch-geometric==2.6.1
torch-scatter==2.1.2+pt21cu118
torch-sparse==0.6.18+pt21cu118
torch-cluster==1.6.3+pt21cu118
torchmetrics==0.11.4
```

---

## Installation

Clone the repository and install MultiST in editable mode:

```bash
git clone https://github.com/LabJunBMI/MultiST.git
cd MultiST
pip install -e .
```

---

## Usage

A step-by-step tutorial demonstrating how to run MultiST is provided in the following notebook:

```text
MultiST/MultiST/test.ipynb
```

The tutorial includes:
- Data preprocessing
- Multimodal feature construction
- Model training
- Downstream analysis and visualization

---

## Contact

For questions or issues, please contact the corresponding authors listed in the paper
or open an issue on GitHub.
