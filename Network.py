import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

# Example architecture
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Define the network layer structure here
        # Example from the website shown here
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*520*951, 3)

    def forward(self, input):
        # Define how a forward pass through the network would look here
        # This function contains all the nonlinear functions between layers
        # The example from the website is continued here
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = output.view(-1, 24*520*951)
        output = self.fc1(output)

        return output

# # Architecture from the slides
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()

#         # Define the network layer structure here
#         # Example from the website shown here
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(12)
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(12)
#         self.pool1 = nn.MaxPool2d(2,2)
#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(24)
#         self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(24)
#         self.pool2 = nn.MaxPool2d(2,2)
#         self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(12)
#         self.conv6 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(24)
#         self.pool3 = nn.MaxPool2d(2,2)
#         self.fc1 = nn.Linear(24*10*10, 3)

#     def forward(self, input):
#         # Define how a forward pass through the network would look here
#         # This function contains all the nonlinear functions between layers
#         # The example from the website is continued here
#         output = F.relu(self.bn1(self.conv1(input)))
#         output = F.relu(self.bn2(self.conv2(output)))
#         output = self.pool1(output)
#         output = F.relu(self.bn3(self.conv3(output)))
#         output = F.relu(self.bn4(self.conv4(output)))
#         output = self.pool2(output)
#         output = F.relu(self.bn5(self.conv5(output)))
#         output = F.relu(self.bn6(self.conv6(output)))
#         output = self.pool3(output)
#         output = output.view(-1, 24*10*10)
#         output = self.fc1(output)

#         return output