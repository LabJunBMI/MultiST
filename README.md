
MultiST: A cross-attention based multimodal model for spatial transcriptomics

MultiST is implemented in the pytorch framework. Please run SEDR on CUDA. The following packages are required to be able to run everything in this repository:
'''
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
'''

cd MultiST
pip install -e .

Tutorials can be found here: MultiST/MultiST/test.ipynb
