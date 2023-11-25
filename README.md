# Fracture-Segmentation-and-Classification-Using-nnUnet-and-ResNet18

## Table of Contents

- [Overview](#overview)
- [Summary of My Work](#summary-of-my-work)
- [Installation and Usage](#installation-and-usage)
- [Enhancements and Future Work](#enhancements-and-future-work)

## Overview
In this project I used the [FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012) dataset, a collection of fractured and non-fractured X-ray images, to perform both segmentation and classification tasks. The primary goals are to identify and segment fractures within X-ray images using [nnUnetV2](https://github.com/MIC-DKFZ/nnUNet) framework and subsequently classifying images as fractured or non-fractured using a PyTorch ResNet18 model.

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
- Generated a dataset.JSON file as required by the nnUNetv2 framework. Set the labels to 0 for background and 1 for fracture, and set channel names (normalization method) to be rescale_to_0_1 as it's a common practice when working with X-ray images.
(Find in [Dataset.JSON](datasetJSON.ipynb))

### Training Segmentation Model:
- Setup the environment and installed nnUNetv2 correctly, initialized environment variables, and ran the dataset integrity command to ensure processed data is a good fit for nnUNet.
- Due to training the nnUNetv2 framework for 5 folds cross validation being a very computationally intensive task to be done on the free version of Google Colab, I switched between two accounts and could only run training for about 50 epochs for the first fold, made sure it was working correctly, prepared the notebook to be used for training the rest of the folds, and progress was saved in nnUNet different folders. Note that number of epochs here is hardcoded and can only be changed using Colab's Terminal that's also only available in the pro version, so I wasn't able to reduce the number of epochs for each fold and complete training. (Find in [nnUNet_Training](nnUNet_Training.ipynb))

### Segmentation Model Inference:
- I prepared a notebook for inference with clear instructions to be run after completing training process. (Find in [nnUNet_Inference](nnUNet_Inference.ipynb))

### Classification:
- Created a custome version of original dataset ([See Here](ResNetPreprocessing.ipynb)) to ensure balanced class distribution and keep the model equally sensitive to all classes, as in this case we prioritize the consequences of false negatives. Also, approximately followed the same split as dataset creators(80:12:6).
- Used Pytorch framework to build a simple ResNet-18 binary classification model, used data augmentation techniques and learning rate scheduler to ensure best performance, and finally trained the model for 50 epochs ([See Here](MyResNet18.ipynb)).
- Plotted training and validation loss and accuracy metrics at the end of training for a better understanding of how the model is performing to help us do the correct hyperparameter tuning and figure out what needs improvement.
- Ran the model on test data and calculated overall model performance
- Saved the model and its weights for later use. (Find in [models](https://drive.google.com/drive/folders/1au_VcDwEMy183qL6zGQc2KuBve-Cwzp1?usp=sharing) folder)

## Installation and Usage

At first, please add a shortcut of this Google Drive [folder](https://drive.google.com/drive/folders/1wmULTo-87FWcIvIN-YeSgEyH1838fymj?usp=sharing) to your drive, as it has both the original and the preprocessed versions of dataset ready for the model, so there will be no need for you to run the preprocessing steps again.

Second, open the [training notebook](nnUNet_Training.ipynb) in Google Colab. Make sure to change runtime type to T4 GPU to utilize Colab's GPU and to mount your drive, and repeat these two steps for all upcoming notebooks. Run all cells as they are. You might just add a little modification if you have access to multiple GPU's for better performance ([See Here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#using-multiple-gpus-for-training)).

Next, you should open [inference notebook](nnUNet_Inference.ipynb) and follow its instructions. Finally, you can use resulting inference data to train the ResNet18 model by changing root folders in [this notebook](MyResNet18.ipynb) and training again for best results.


## Enhancements and Future Work

I definetily couldn't get the best out of this data and these models due to time and resources limitations. Improvements can be done such as using the Pro+ version of Google Colab to utilize high RAM and several GPU's to train the different data folds in parallel as recommended by nnUNet creators and avoid that your session times out. Also allowing the training to run for more epoches until it's done and run the other 4 folds as well. Then, moving further and running the inference notebook to obtaing segmentation results from nnUNet and use it for classification.

Regarding classification, grid search can be applied for better hyperparameter tuning, and generative super resolution techniques can be used as a preprocessing step to enhance training data and therefore achieve better performance. Also we can consider using more complex ResNet architectures than the ResNet-18 and running more epochs.

And of course, the main goal of this project, using inference data of nnUNetv2 segmentation to train the classifier by only changing the paths to the data in the ResNet notebook and running again.
