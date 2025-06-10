# BeeWingAnalysis

Repository for my Master's thesis in bioinformatics. The semi-automated wing segmentation tool developed for this project can be found [here](https://github.com/Jakob-Materna/SegmentAnywing). This repository is used for segmenting wing cells, age and weight from photographs and scans of bee wing images. The preprocessing ueses Segment-Anything, followed by training and prediction using CNN models.

## Scripts

### 0: Helper scripts
 - **0-GetCoordinates.ipynb:** Tool for getting image coordinates with a mouse click. 
 - **0-ImageSorter.ipynb:** Tool for manually sorting images into two categories using keyboard input.
 - **0-MeasureDistance.ipynb:** Tool for manual distance measurement.
 - **0-TrainValTestSplit.ipynb:** Splits the data set into training, validation and test set.
 
### 1: Preprocessing wing scans
 - **1-1-ScansWingCropping.ipynb:** Loops through all Wing Scans, segments them into the individual wings and usees an OCR to extract the label number.
 - **1-2-ScansManualImprovements.ipynb:** Renames wings with incorrect OCR results, adjust left/right wing identification, and deformed wings marked with an 'x'.
 - **1-3-ScansWingAlignment.ipynb:** Identifies the background using segment-anything and determines the wing orientation and height.
 - **1-4-ScansWingOrientation.ipynb:** Correctly orientates the images by detecting the black color at the top and base of the wing.

### 2: Preprocessing live bee wings
 - **2-1-LiveBeeLabelCropper.ipynb:**  Identifies the labels on the scaned sheets and saves them.
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
 - **4-Recenter.ipynb:** Recenters wings around the segmentated wing cells, removing variation in amount of visible wing base.
 - **4-RemovedFWL.ipynb:** Removes the forewing lobe using wing segmentations.
 - **4-StrictFilter.ipynb:** Sorts out bad quality live bee photographs with bad lighting or in which only parts of the wings were visible.
 - 
### 5: 
 - **5-AgeHistogram.ipynb:** Creates an age histogram.
 - **5-AreaLossFWL.ipynb:** Shows the relationship between bee age and forewing lobe area with a linear trend line.
 - **5-ModelEvaluation.ipynb:** Visualizes model performance.

## Workflow

![workflow](https://github.com/user-attachments/assets/c960d627-5cad-4617-8bff-7f6bcd965509)
