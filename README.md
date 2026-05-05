# segmango

Notebook-first mango yield prediction project using segmentation-derived features and tabular regression.

## Installation

### Setup environment

Create and activate a conda environment:

```bash
conda create -n "openmmlab" python=3.8 -y
conda activate openmmlab
```

### Install PyTorch on GPU platform

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

Test the installation:

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.cuda)'
```

### Install MMCV using MIM

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
```

### Clone mmsegmentation repository

```bash
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
```

### Install remaining dependencies

```bash
pip install -v -e .
pip install ftfy
pip install regex
pip install "mmdet>=3.0.0rc4"
```

### Install project-specific packages

From the project root, install the Python packages used by the notebooks:

```bash
pip install pandas numpy scikit-learn torch matplotlib python-dotenv jupyter
```

## Data preparation flow

1. Run `data/dataset_download.ipynb` to download image and weather data from the Google Drive source.
2. Install all imports required by that notebook in the environment first.
3. Run `data/dataset_preprocessing.ipynb` to generate CSV files in `data/tabular_data/`.

## Notebook workflow

- `models/approach_1.ipynb` is the main notebook for approach-1 from the Segmango WACV paper.
- It contains train/test data preparation, feature generation, and linear regression baseline results.
- The repository is notebook-based and intentionally kept minimal; the notebooks are self-explanatory.

## Project structure

- `Depth-Anything-V2/`: segmentation/depth model code.
- `data/`: raw weather files, generated CSVs, model weights, and split files.
- `data/dataset_download.ipynb`: download dataset assets.
- `data/dataset_preprocessing.ipynb`: preprocess images and weather into tabular CSVs.
- `models/approach_1.ipynb`: approach-1 experiment.
- `utils/`: helper utilities.

## Next steps

- `approach-1` is complete for the current baseline.
- Work on `approach-2` is planned next.

## Notes

- Keep the conda environment active when running notebooks.
- Use the `.env` file for `ROOT_DIR` and `DATA_DIR` paths if needed.
- This README is intentionally concise because the notebooks contain the detailed workflow.




