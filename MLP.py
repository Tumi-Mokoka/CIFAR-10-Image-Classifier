
# MKKBOI004 - Tumi Mokoka
# Assignment 1 - ANN that classifies the CIFAR10 Dataset
# note: classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import sys
import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
#import matplotlib.pyplot as plt
#import numpy as np

import torch.nn as nn #for layers in the MLP
import torch.nn.functional as F #the activation functions between layers

import torch.optim as optim #optimizers
from torchvision.transforms.autoaugment import AutoAugmentPolicy


#defining the MLP architecture
class MLP (nn.Module):
    def __init__ (self):
        num_HL1 = 512
        num_HL2 = 256
        num_dropout = 0.1

        super(MLP,self).__init__()
        self.flatten = nn.Flatten() # flattening 2D image for input layer
        self.fc1 = nn.Linear(3*32*32, num_HL1) #input is 3-channel 32x32 image
        self.fc2 = nn.Linear (num_HL1, num_HL2) # First HL
        self.fc3 = nn.Linear (num_HL2, 10) # Second HL
        self.output = nn.LogSoftmax (dim =1) # output layer
        self.dropout = nn.Dropout(p = num_dropout)

        # batch normalization implemeted
        self.BN1 = nn.BatchNorm1d(num_HL1)
        self.BN2 = nn.BatchNorm1d(num_HL2)

    def forward(self, x):
        # x = Batch 
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.BN1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.BN2(x)
        x = self.fc3(x) #output layer
        x = self.output(x)
        return x


#Training function
def train(net, train_loader, criterion, optimizer, device):
    net.train() #puts model in training mode
    running_loss = 0.0 #calc average loss across batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the network i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # back-propogation
        optimizer.step()  # weight updates
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)

#Testing function [using unseen data]
def test (net, test_loader, device):
    net.eval() #evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def main():

    load = False
    save =  False

    #check how many arguments in command line
    if len(sys.argv) > 1:
        if sys.argv[1] == "-load":
            load = True
            print ("Loading model...")
        elif sys.argv[1] == "-save":
            save = True
            print ("Saving model...")

    # Create the transform sequence
    transformTrain = transforms.Compose  (
        [
        transforms.RandomHorizontalFlip(p = 0.05), #random data augmentation for vertical flips
        transforms.ToTensor(),  # Convert to Tensor
        # Normalizes image to have mean = 0.5 and standard deviation = 0.5 for each RGB channel
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ]
                                    )
    
    transformTest = transforms.Compose  (
        [
        transforms.ToTensor(),  # Convert to Tensor
        # Normalizes image to have mean = 0.5 and standard deviation = 0.5 for each RGB channel
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ]
                                    )
    

    

    # Training dataset
    trainset = torchvision.datasets.CIFAR10(root= './data', train=True, download = True, transform=transformTrain)
    # Testing dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest)

    # using dataloaders --> pulls instances of the dataset in batches to feed into the training loop
    BATCH_SIZE = 32
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    

    # Identifying device
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
                )
    print(f"Using {device} device")


    # creating the netork
    mlp = MLP().to(device)
    
    #loading a pre-trained model
    if load:
        mlp.load_state_dict(torch.load("mlp_model.pt"))
        print("Model loaded")
        test_acc = test(mlp, test_loader, device)
        print(f"Test accuracy = {test_acc:.4f}")
    
    #training a new model
    else:
        LEARNING_RATE = 0.005
        MOMENTUM = 0.85

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(mlp.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

        for epoch in range(15):
            # for the last 5 training cycles momentum is halved each time --> 
            if epoch > 10: 
                MOMENTUM *= 0.25 # decreasing momentum after the 10th training cycle
                optimizer = optim.SGD (mlp.parameters(), lr = LEARNING_RATE, momentum= MOMENTUM)
            train_loss = train(mlp, train_loader, criterion, optimizer, device)
            test_acc = test(mlp, test_loader, device)
            print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
    

    # Save model
    if save:
        torch.save(mlp.state_dict(), "mlp_model.pt")
        print("MLP Model saved")
    

if __name__ == "__main__":
    main()

