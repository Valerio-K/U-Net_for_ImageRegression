# U-Net_for_ImageRegression

A convolutional neural network for image binary segmentation and image regression.

# Requirements
* albumentations
* matplotlib
* numpy
* os
* PIL
* time
* torch
* torchvision
* torchmetrics
* tqdm

# Usage
Very easy to use.
The file 'model.py' contains the model structure. Nothing has to change in it.
The file 'dataset.py' contains the dataset acquisition:
	for binary segmentation, put the images you want your model to train from, in 'data\dataBinary\train_images' and relative target in 'data\dataBinary\train_masks'
			 	 then put the images you want the model to test, in 'data\dataBinary\test_images' and ground truth in 'data\dataBinary\test_masks';
	for regression, put the images you want your model to train from, in 'data\dataRegression\train_images' and relative target in 'data\dataRinary\train_masks'
			then put the images you want the model to test, in 'data\dataRegression\test_images' and ground truth in 'data\dataRegression\test_masks'.
The file 'utils.py' contains some utils function like saving images or cheking accuracy.
The most important file 'train.py' contains the whole training of model in 'model.py':
	use parameter 'TYPE_DATASET' to switch the task you want to perform: 0 for binary segmentation, 1 for regression;
	set the right 'IMAGE_HEIGHT' and 'IMAGE_WIDTH' according to images dimensions;
	set the right hyperparameters like 'LEARNING_RATE', 'WEIGHT_DECAY', and 'EPOCH' to perform your chosen task; 
	use boolean parameter 'LOAD_MODEL' to load a checkpoint previously created.

# Description
This set of Python script perform a binary segmentation task or regression task, taking some images with relative target in input, to train the model e testing it on others images with relative ground truth.
'saves/' folder contains the result of test images with ground truth too and loss function plot for epochs, then the algorithm outputs the accuracy trhought dice score for binary segmentation and r-squared and SSIM for regression.