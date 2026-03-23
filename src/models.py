
# src/models.py - Deep Learning Models for Computer Vision

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512) # Assuming input image size 32x32 after two pools
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(ResNetBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNetBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNetBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNetBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4) # Assuming input image size 32x32
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    # Example usage
    dummy_input = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image
    
    cnn_model = SimpleCNN(num_classes=10)
    cnn_output = cnn_model(dummy_input)
    print(f"SimpleCNN output shape: {cnn_output.shape}")

    resnet_model = ResNet18(num_classes=10)
    resnet_output = resnet_model(dummy_input)
    print(f"ResNet18 output shape: {resnet_output.shape}")

    print("Deep Learning Models for Computer Vision module initialized.")

# This file contains implementations of common deep learning architectures for computer vision.
# It includes a basic Convolutional Neural Network (CNN) and a simplified ResNet-18 model.
# The models are built using PyTorch, a popular deep learning framework.
# Each model is defined as a class inheriting from nn.Module, allowing for easy integration.
# The forward method defines the pass through the network.
# ResNet blocks incorporate skip connections to mitigate the vanishing gradient problem.
# This code serves as a foundation for various computer vision tasks like image classification.
# Further extensions could include more advanced architectures, pre-trained weights, and data augmentation.
# The example usage demonstrates how to instantiate and run a forward pass through the models.
# This module is designed for educational purposes and as a starting point for research projects.
# It emphasizes clear, readable code and modular design principles.
# The use of BatchNorm layers helps stabilize training and accelerate convergence.
# MaxPool2d layers are used for downsampling feature maps.
# ReLU activation functions introduce non-linearity into the network.
# Linear layers perform the final classification.
# This is a core component for any deep learning computer vision project.
# It showcases the power and flexibility of PyTorch for building complex models.
# The code is well-commented to explain the different parts of the network.
# It's a valuable resource for anyone learning or working with computer vision.
# The models can be easily adapted for different datasets and tasks.
# Enjoy exploring the world of deep learning for computer vision!
