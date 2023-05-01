from torchvision.transforms import transforms
from glob import glob
import csv
import matplotlib.pyplot as plt
from Network import Network
import os
import torch

# Loading the model trained from TrainingScript.py
model = Network()
path = os.path.join(os.getcwd(), "trained_model.pth")
model.load_state_dict(torch.load(path))
transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Get all the test image file locations
sub_csv_file = os.path.join(os.getcwd(), "submission.csv")
root_dir = os.path.join(os.getcwd(),"test")
test_files = glob(os.path.join(root_dir, "*/*_image.jpg"))

# Trim the file names to just the folder and image number to match expected csv format
trimmed_files = [s.replace(os.getcwd(),'') for s in test_files]
trimmed_files = [s.replace('/test/','') for s in trimmed_files]
trimmed_files = [s.replace('_image.jpg','') for s in trimmed_files]

# Pass each test file through the trained network and record the predicted label
predictions = []
i = 0
for test_file in test_files:
    img = plt.imread(test_file)
    input = transformations(img)
    input = input[None, :]
    output = model(input)
    _, predicted = torch.max(output, 1)
    predictions.append(predicted.item())
    if i % 50 == 0:
        print(i)
    i += 1

# Create the csv file, including the expected header
header = ['guid/image', 'label']
data = [[file, pred] for (file, pred) in zip(trimmed_files, predictions)]
with open(sub_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)