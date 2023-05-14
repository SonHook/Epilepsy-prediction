import os
import signal
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from mne.datasets import sample
from scipy.signal import stft
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.signal import stft
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# # Load the datasets
# with open('/home/sunhuaike/AI/23/fir_dataset_1.pkl', 'rb') as f:
#     X_test, y_test = pickle.load(f)
#
# with open('/home/sunhuaike/AI/23/fir_dataset_2.pkl', 'rb') as f:
#     X_1, y_1 = pickle.load(f)
#
# with open('/home/sunhuaike/AI/23/fir_dataset_3.pkl', 'rb') as f:
#     X_2, y_2 = pickle.load(f)
#
# with open('/home/sunhuaike/AI/23/fir_dataset_4.pkl', 'rb') as f:
#     X_3, y_3 = pickle.load(f)
#
# with open('/home/sunhuaike/AI/23/fir_dataset_5.pkl', 'rb') as f:
#     X_val, y_val = pickle.load(f)
#
# # Concatenate the remaining datasets for training and validation
# X_train = np.concatenate((X_1, X_2, X_3), axis=0)
# y_train = np.concatenate((y_1, y_2, y_3), axis=0)

# Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Define the paths to the dataset files
dataset_paths = ['/home/sunhuaike/AI/23/fir_dataset_1.pkl',
                 '/home/sunhuaike/AI/23/fir_dataset_2.pkl',
                 '/home/sunhuaike/AI/23/fir_dataset_3.pkl',
                 '/home/sunhuaike/AI/23/fir_dataset_4.pkl',
                 '/home/sunhuaike/AI/23/fir_dataset_5.pkl']

# Initialize the accuracy counter
avg_accuracy = 0

# Loop over the datasets, using each one as test set once
for i in range(5):
    # Load the test set
    with open(dataset_paths[i], 'rb') as f:
        X_test, y_test = pickle.load(f)

    # Load the validation set
    val_index = (i+1) % 5
    with open(dataset_paths[val_index], 'rb') as f:
        X_val, y_val = pickle.load(f)

    # Load the training set
    train_indices = [j for j in range(5) if j != i and j != val_index]
    X_train_list = []
    y_train_list = []
    for j in train_indices:
        with open(dataset_paths[j], 'rb') as f:
            X_train_j, y_train_j = pickle.load(f)
            X_train_list.append(X_train_j)
            y_train_list.append(y_train_j)
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    # Define dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Define CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 64, 128)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 64 * 64)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    # Initialize model and optimizer
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    # Define loss function and number of epochs
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100

    # Train model on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    '''xiugai------------------------'''
    best_loss = float('inf') # 初始化最佳验证损失为正无穷大
    early_stop = False # 初始化早期停止标志为 False
    '''xiugai------------------------'''
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)

            outputs = outputs.view(labels.shape[0], -1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[Epoch %d, Batch %d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        '''xiugai---------------------------------'''
        # Evaluate model on validation set and print accuracy
        with torch.no_grad():
            val_loss = 0.0
            for data in val_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            print('Validation loss: %.3f' % val_loss)

        # Check if the validation loss is decreasing
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_params = model.state_dict() # 记录最佳模型参数
        else:
            early_stop = True

        # Early stopping criterion
        if early_stop:
            print('Early stopping at epoch %d' % (epoch + 1))
            break

    # Load best model parameters
    model.load_state_dict(best_model_params)

    # Evaluate model on test set and print accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取每个输入图像的预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on test set ', i, ': %.2f %%' % accuracy)

    # Add this test set's accuracy to the average accuracy counter
    avg_accuracy += accuracy

# Compute and print the average accuracy over all 5 test sets
avg_accuracy /= 5
print('Average accuracy on test sets: %.2f %%' % avg_accuracy)
'''----------------------end'''