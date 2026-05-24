# 🥭SegMango

Mango yield prediction project using segmentation-derived features and tabular regression.

## Enviroment Setup: 
Use the UV or conda to setup the environment using environment.yml requirement.txt files
```
conda env create -f environment.yml
```
```
conda activate segmango
pip install -r requirements.txt
```

## Dataset Download

- Run `data/dataset_download.ipynb` to download image, manual count for final visit and weather data from the Google Drive source. 
- It also downloads the required deplth anything model weights from the hugging face.

## .env file setup
- check the .env.example and rename it to .env
- Also, update the data and repo dir

## Dataset Preprocess
- Run `models/dataset_preprocessing.ipynb` to generate CSV files in `data/tabular_data/`.
- It contains train/test data preparation and feature generation.

## Approach 1
Run the
```
python segmango_ssh/models/approach_1.py
``` 
is the main python file for approach-1 from the Segmango WACV paper.

## Approach 2

### Step-1 (Segformer finetunning)
- We will use the mmsegmantation library for the segformer finetunning on our dataset
- Data preparation, model config setup and required commands for environments are given in the `segmango_ssh/models/segformer_training/main.ipynb`

### Step-2 (Image based model: Segformer encoder + Regression Model)
- This is stage-1 of the model training, here model will trained on images level data.
- In the `segmango_ssh/models/approach_2` folder the dataloader, model and training script is written.
1. To train the only image based input model:
```
python train_segmango.py --fold 1 --variant b1 --unfreez_epoch 30
``` 
2. To train the image, weather and scale based input model:
```
python train_segmango.py --weather --scale --fold 1 --variant b1 --unfreez_epoch 30
``` 

### Step-3 (Tree based model: Segformer encoder + Regression + Regression Model)
- This is stage-2 of the model training, here model will trained on tree level data (8 images of one tree as input).
- In the `segmango_ssh/models/approach_2` folder the dataloader, model and training script is written.
1. To train the only image based input model:
```
python train_segmango_per_tree.py --fold 1 --variant b1 --unfreez_epoch 50
``` 
2. To train the image, weather and scale based input model:
```
python train_segmango_per_tree.py --weather --scale --fold 1 --variant b1 --unfreez_epoch 50
``` 



## Project structure


## Inference Setup: 
Make sure to download the model weights for inference and save in the `segmango_ssh/data/Model_weights/approach-1/segmango` path.
1. Image based model: 
```
# python segmango_ssh/models/approach_2/inference_segmango_image.py --image "segmango_ssh/02_10_02.jpg" --fold 1 --variant b1 --time 64
```

2. Per Tree based model:
```

```

## Notes

- Keep the conda environment active when running notebooks.
- Use the `.env` file for `ROOT_DIR` and `DATA_DIR` paths if needed.
- This README is intentionally concise because the notebooks contain the detailed workflow.

## Citation
```
@InProceedings{Ven_2026_WACV,
    author    = {Ven, Janaksinh and Sharma, Charu and Syed, Azeemuddin},
    title     = {SegMango: Early Deep Mango Yield Prediction based on Flower Segmentation and Weather Data},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {4984-4993}
}
```
<!-- ```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
``` -->


