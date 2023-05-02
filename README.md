# 16664_perception_proj
Code for completing the perception project for 16-664-A Self-Driving Cars: Perception & Control.  The code can either be run locally or using Google Colab (preferred).  Using Colab makes setting up the environment and installing all packages easy and provides enough RAM to do larger batch sizes.

## Running on Colab (Recommended)
To run on Colab, open proj_script.ipynb in Colab and save a copy of the file to a folder called 16664_percep_proj on your own Google Drive.  The variables DRIVE_PATH and SYM_PATH in the second code block may need to be updated to match your directory structure.  There are also some cd commands throughout the notebook that may need to be updated to reflect your directory structure.<br /><br />
Ensure that you have the unzipped test and trainval folders uploaded to Google Drive as subfolders within the 16664_percep_proj folder.<br /><br />
Connecting to a runtime and running all of the code blocks in order should install all necessary libraries, train the model, and gather the model's predictions on all the test images.  The trained model will be saved in the 16664_percep_proj folder as trained_model.pth and the formatted predictions will be saved as submission.csv

## Running Locally
### Install packages
Make sure all the libraries necessary for the code to function are installed<br />
```pip install pandas```<br />
```pip install matplotlib```<br />
```pip install numpy```<br />
```pip install csv```<br />
```pip install torch```<br />
```pip install torchvision```<br />
### Training and testing
Before running any code, ensure that the 4 python files (CustomDataset.py, Network.py, TrainingScript.py, and TestScript.py) and the two data folders (trainval and test) are all in the same folder at the same level.<br /><br />
To train the model, run TrainingScript.py.  Doing so will produce a trained_model.pth file that contains the trained model.  This file is saved every 100 batches and at the end of training. From folder where all the scripts and data are stored, run in the terminal<br />
```python3 TrainingScript.py```<br /><br />
To the trained model's predictions for the test images, run TestScript.py.  This will produce a submission.csv file of all the predictions formatted in a way such that it can be directly uploaded to Kaggle for evaluation.  From folder where all the scripts and data are stored, run in the terminal<br />
```python3 TestScript.py```
