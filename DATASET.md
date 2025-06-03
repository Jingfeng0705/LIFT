# Evaluation Datasets

## SugarCrepe

Please first download SugarCrepe's 7 [JSON files](https://github.com/RAIVNLab/sugar-crepe/tree/main/data) of augmented captions and put them under the folder `extra_zeroshot_datasets/sugar_crepe`. SugarCrepe uses the images from COCO Val2017. Please set the `data_path` in the [evaluation config](evaluation/configs) to your COCO2017 folder path. This folder should have the structure:
```
COCO2017/
    val2017/
    ...
```

## ImageNet-1K
We use ImageNet-1K validation set (50,000 images) to evaluate our models. Please set the `data_path` in the [evaluation config](evaluation/configs) to your ImageNet-1K folder path. This folder should have the structure:
```
imagenet1k\
    val\
        n01440764\
        n01698640\
        n01860187\
        ...
```
The names for each ImageNet-1K class are provided in this [JSON file](extra_zeroshot_datasets/imagenet_class_index.json).

## COCO

We use COCO Val2017 split (5,000 images) to evaluate our models. Please set the `data_path` in the [evaluation config](evaluation/configs) to your COCO2017 folder path. This folder should have the structure:
```
COCO2017/
    val2017/
    annotations/
    ...
```

## Flickr30K

We use the test set (1,000 images) of Flickr30K to evaluate our models. Please set the `data_path` in the [evaluation config](evaluation/configs) to your Flickr30K folder path. This folder should have the structure:
```
flickr30k/
    flickr_annotations_30k.csv/
    flickr30k-images/
    ...
```

<br/><br/><br/>
# Training / Embeddings Generation Datasets
We download our training images and captions from [Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B), which includes both the original DataComp-1B captions and the re-captioned versions in its metadata. We use the release containing 3,550 subfiles. Due to some download issues, we were only able to collect a subset of 400 million text-image pairs. However, given the scale and distribution of the dataset, we believe that training on a different split would yield models with performance very close to what we report.

We are working on the dataloader that supports DataComp-1B's data organization.

## The Raw Caption Folder
The raw caption folder has the structure of Recap-DataComp-1B's metadata file:
```
raw_captions\
    train-00000-of-03550.parquet
    train-00001-of-03550.parquet
    ...
    train-03550-of-03550.parquet
```
The raw caption folder is used in offline embeddings generation and the training of CLIP.

## The Caption Embedding Folder
This is the destination folder of caption embeddings after runnning [embed.sh](scripts/embed.sh). It has the structure
```
caption_embeddings\
    train-00000-of-03550\
        0.parquet
        1.parquet
        2.parquet
        ...
    ...
    train-03550-of-03550\
        0.parquet
        ...
```
For some performance reasons, we split each big caption parquet file into smaller parquets (each with 100 caption embeddings) and organize them under a folder named after the original big caption parquet. 

The caption embedding folder is used in the training of LIFT.

## The Image Folder
The image folder has the structure of Recap-DataComp-1B's image folder:
```
images/
    train-00000-of-03550\
        000000000001.jpg
        000000000002.jpg
        000000000003.jpg
        ...
    ...
    train-03550-of-03550\
        000000000001.jpg
        ...
```
The image folder is used in the training of CLIP and LIFT.