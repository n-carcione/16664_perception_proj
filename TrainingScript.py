import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from Network import Network
from CustomDataset import CustomDataset

batch_size = 10
learning_rate = 0.001
weight_decay = 0.0001

csv_file = os.path.join(os.getcwd(), "trainval/labels.csv")
root_dir = os.path.join(os.getcwd(), "trainval")
transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
training_data = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transformations)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

model = Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def saveModel():
    path = "./trained_model.pth"
    torch.save(model.state_dict(), path)

def train(num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model willbe running on ", device, " device")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader, 0):
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict the classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backprop the loss
            loss.backward()
            # adjust the parameters based on the calculated gradients
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/100))
                running_loss = 0.0
                saveModel()

if __name__ == '__main__':
    train(5)
    print("Finished training")
    saveModel()