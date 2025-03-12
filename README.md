# BeeWingAnalysis

This repository is used for segmenting wing structures and predicting wing segments and age from photographs and scans of bee wing images. The preprocessing utilizes Segment-Anything, followed by training and prediction using UNet++ models.

## Scripts

### 0: Helper scripts
 - **0-ImageSorter.ipynb:** Tool for manually sorting images into two categories using keyboard input
 - **0-MeasureDistance.ipynb:** Tool for manual distance measurement
 
### 1: Preprocessing wing scans
 - **1-1-ScansWingCropping.ipynb:** Loops through all Wing Scans, segments them into the individual wings and usees an OCR to extract the label number
 - **1-2-ScansManualImprovements.ipynb:** Renames wings with incorrect OCR results, adjust left/right wing identification, and deformed wings marked with an 'x'
 - **1-3-ScansWingAlignment.ipynb:** Identifies the background using segment-anything and determines the wing orientation and height
 - **1-4-ScansWingFlipper.ipynb:** Flips top to bottom or left to right by detecting the black color at the top and base of the wing
 
### 2: Preprocessing live bee wings
 - **2-1-LiveBeeLabelCropper.ipynb:** Crops the white background label using segment-anything
 - **2-2-LiveBeeContourFinder.ipynb:** Identifies wing and marker contours on the label. The marker length is measured and the wing is cropped and saved
 - **2-3-LiveBeeManualImprovements.ipynb:** Introduces manual corrections to wing cropping and marker length
 - **2-4-LiveBeeBackgroundRemoval.ipynb:** Removes the background using segment-anything
 - **2-5-LiveBeeSegmentation.ipynb:** Identifies wing segments using segment-anything and returns masks for UNet training and validation
 - **2-6-LiveBeeTrainValSplit.ipynb:** Turns successfully segmented wings into training and validation sets

### 3: Process segmentation predictions
 - **3-1-ResizePredictions.ipynb** Resizes UNet prediction results to original format
 - **3-2-ProcessPredictions.ipynb** Measures predicted segment areas
 - **3-3-Visualisation.ipynb** Visualizes predicted areas on top of the input images

## Workflow

![workflow](https://github.com/user-attachments/assets/c960d627-5cad-4617-8bff-7f6bcd965509)
