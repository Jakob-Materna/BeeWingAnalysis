# BeeWingAnalysis

Repository for my Master's thesis in bioinformatics. The semi-automated wing segmentation tool developed for this project can be found [here](https://github.com/Jakob-Materna/SegmentAnywing). This repository is used for segmenting wing cells and age and weight prediction. The preprocessing uses [Segment Anything](https://github.com/facebookresearch/segment-anything#installation), followed by training and prediction of CNN models.

## Usage

Install the dependencies using conda and activate the environment. There is a separate environment for model training and prediction:

```
conda env create -f environment.yml
conda activate bee-wings
```

The [Segment-Anything](https://github.com/facebookresearch/segment-anything#installation)  checkpoints used during preprocessing must be downloaded separately:

  - `vit_h`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
  - `vit_l`: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
  - `vit_b`: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

The semantic segmentation models can be used with the following commands:
```
python main.py --train --dataPath /path/to/data/
python main.py --predict --checkpoint /path/to/checkpoint/ --dataPath /path/to/data/
```

## Workflow

The scripts in this repository are prefixed by their role in the workflow. Scripts prefixed with 0 are helper tools. Prefix 1 indicates preprocessing steps for scanned bee wings, 2 refers to preprocessing of live bee wing photographs. Scripts starting with 3 handle postprocessing of semantic segmentations. Scripts with prefix 4 are optional scripts for preparing various input data sets for age and weight predictions. Scripts with 5 include visualization and model evaluation scripts.

![workflow](https://github.com/user-attachments/assets/ee3d12c0-e329-40e2-87a6-feb05b522619)

## Scripts

The subdirectories contain the code for the convolutional neural networks. The main directory contains preprocessing and visualization scripts. 

### 0: Helper scripts
 - **0-GetCoordinates.ipynb:** Tool for getting image coordinates with a mouse click. 
 - **0-ImageSorter.ipynb:** Tool for manually sorting images into two categories using keyboard input.
 - **0-MeasureDistance.ipynb:** Tool for manual distance measurement.
 - **0-TrainValTestSplit.ipynb:** Splits the data set into training, validation and test set.
 
### 1: Preprocessing wing scans
 - **1-1-ScansWingCropping.ipynb:** Loops through all Wing Scans, segments them into the individual wings and uses an OCR to extract the label number.
 - **1-2-ScansManualImprovements.ipynb:** Renames wings with incorrect OCR results, adjust left/right wing identification, and deformed wings marked with an 'x'.
 - **1-3-ScansWingAlignment.ipynb:** Identifies the background using segment-anything and determines the wing orientation and height.
 - **1-4-ScansWingOrientation.ipynb:** Correctly orients the images by detecting the black color at the top and base of the wing.

### 2: Preprocessing live bee wings
 - **2-1-LiveBeeLabelCropper.ipynb:**  Identifies the labels on the scanned sheets and saves them.
 - **2-2-LiveBeeContourFinder.ipynb:** Identifies wing and marker contours on the label. The marker length is measured and the wing is cropped and saved.
 - **2-3-LiveBeeManualImprovements.ipynb:** Introduces manual corrections to wing cropping and marker length.
 - **2-4-LiveBeeBackgroundRemoval.ipynb:** Removes the background using segment-anything.
 - **2-5-LiveBeeWingOrientation.ipynb:** Correctly orientates the images by detecting the black color at the top and base of the wing.
 - **2-6-LiveBeeSegmentation.ipynb:** Identifies wing segments using segment-anything and returns masks for UNet training and validation.
 - **2-7-LiveBeeTrainValSplit.ipynb:** Turns successfully segmented wings into training and validation sets.

### 3: Process segmentation predictions
 - **3-1-ResizePredictions.ipynb:** Resizes UNet prediction results to original format.
 - **3-2-CombineData.ipynb:** Combines output data into a single csv file.
 - **3-3-Visualisation.ipynb:** Visualizes semantic segmentation results.

### 4: Prepare input data for age and weight predictions
 - **4-ApplyImageScale.ipynb:** Applies the image specific mm to pixel scale for each live bee image.
 - **4-OnlyFWL.ipynb:** Removes everything but the forewing lobe using wing segmentations.
 - **4-Recenter.ipynb:** Recenters wings around the segmented wing cells, removing variation in amount of visible wing base.
 - **4-RemovedFWL.ipynb:** Removes the forewing lobe using wing segmentations.
 - **4-StrictFilter.ipynb:** Sorts out bad quality live bee photographs with bad lighting or in which only parts of the wings were visible.

### 5: Visualization
 - **5-AgeHistogram.ipynb:** Creates an age histogram.
 - **5-AreaLossFWL.ipynb:** Shows the relationship between bee age and forewing lobe area with a linear trend line.
 - **5-ModelEvaluation.ipynb:** Visualizes model performance.

