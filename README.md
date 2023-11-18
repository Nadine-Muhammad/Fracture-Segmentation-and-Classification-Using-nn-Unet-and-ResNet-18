# Fracture-Segmentation-and-Classification-Using-nnUnet-and-ResNet18

## Table of Contents

- [Overview](#overview)
- [Summary of My Work](#summary-of-my-work)
- [Installation and Usage](#installation-and-usage)
- [Enhancements and Future Work](#enhancements-and-future-work)

## Overview
In this project I used the FracAtlas dataset, a collection of fractured and non-fractured X-ray images, to perform both segmentation and classification tasks. The primary goals are to identify and segment fractures within X-ray images using a nnUnetV2 and subsequently classify images as fractured or non-fractured using a PyTorch ResNet18 model.

## Summary of My Work

### Gathering Data and Environment Setup:
- Downloading the FracAtlas dataset, adding it to my Google Drive, carefully reading the [article](https://www.nature.com/articles/s41597-023-02432-4) published by data creators and performing exploratory data analysis to get to understand the data.
- Following the nnUnet repository's instructions on how to setup the environment for nnUnet and correctly install required packages.

### The trickiest part was Data Preprocessing and Conversion into the required folder structure to fit the nnUNetv2 segmentation framework, and I did the following:
- Converted the COCO JSON annotations into labeled images, by reading the annotations file and mapping segmentation masks to their images, also covered the case of having several masks for the same image.
- Labeled areas with the fracture with ones and everything else (background) with zeros.
- Saved labels as NIfTI images as it's often a standard format when it comes to medical image segmentation tasks, resized them to 224x224, set color channel to grayscale, and renamed them to match nnUNetv2's naming convention.
- Loaded training images into their folder and applied the same processing as label images, as images and their labels need to have the same spatial dimensions during both training and inference. Also renamed them to expected name format.
- Splitted training and testing data a 80:20 split resulting in 573 training images and 144 test images, and moved all instances to their right folders.
(Find in [Data Preprocessing](Data_Preprocessing.ipynb))
- Generated a dataset.JSON file as required by the nnUNetv2 framework.
(Find in [Dataset.JSON](datasetJSON.ipynb))

### Training The Model:
- Segmentation with nnUNetv2: A segmentation model is trained using the nnUNetv2 framework to identify and segment fractures within X-ray images. This model produces segmented masks for each image.
Classification with PyTorch ResNet18: The segmented data is utilized to train a PyTorch ResNet18 model for binary classification. The objective is to classify images into fractured and non-fractured categories based on the segmented information.

## Installation and Usage

At first, please add a shortcut of this Google Drive [folder](https://drive.google.com/drive/folders/1wmULTo-87FWcIvIN-YeSgEyH1838fymj?usp=sharing) to your drive, as it has both the original and the preprocessed versions of dataset ready for the model, so there will be no need for you to run the preprocessing steps again.
Second, open the [notebook](data/your_file.txt) notebook in Google Colab. Make sure to change runtime type to T4 GPU utilize Colab's GPU, and to mount your drive.

## Enhancements and Future Work

I definetily couldn't get the best out of this data and these models due to time and resources limitations. Improvements can be done that would increase accuracy and performance such as using the Pro+ version of Google Colab to utilize several GPU's to train the different data folds as recommended by nnUNet creators and avoid that your session times out. Also allowing the training to run for a longer time making more epoches.

Regarding classification, grid search can be applied for better hyperparameter tuning, and generative super resolution techniques can be used as a preprocessing step to enhance training data and therefore achieve better performance.
