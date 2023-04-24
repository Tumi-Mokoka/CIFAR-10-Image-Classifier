# MKKBOI004 - Tumi Mokoka
# Assignment 1 - ANN that classifies the CIFAR10 Dataset

import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
#import matplotlib.pyplot as plt
#import numpy as np

import torch.nn as nn #for layers in the MLP
import torch.nn.functional as F #the activation functions between layers

import torch.optim as optim #optimizers


#defining the MLP architecture
class MLP (nn.Module):
    def __init__ (self):
        super(MLP,self).__init__()
        self.flatten = nn.Flatten() # flattening 2D image for input layer
        self.fc1 = nn.Linear(3*32*32, 2048) #input is 3-channel 32x32 image
        self.fc2 = nn.Linear (2048, 1024) # First HL
        self.fc3 = nn.Linear (1024, 128) # Second HL
        self.fc4 = nn.Linear(128, 10) # third HL
        self.output = nn.LogSoftmax (dim =1) # output layer
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
        # x = Batch 
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x= self.dropout(x)
        x = F.relu(self.fc2(x))
        x= self.dropout(x)
        x = F.relu(self.fc3(x))
        x= self.dropout(x)
        x = self.fc4(x) #output layer
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
    # Create the transform sequence
    transform = transforms.Compose  (
        [
        transforms.ToTensor(),  # Convert to Tensor
        # Normalizes image to have mean = 0.5 and standard deviation = 0.5 for each RGB channel
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ]
                                    )

    # Training dataset
    trainset = torchvision.datasets.CIFAR10(root= './data', train=True, download = True, transform=transform)
    # Testing dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # using dataloaders --> pulls instances of the dataset in batches to feed into the training loop
    BATCH_SIZE = 32
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # note: classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

    # Identifying device
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
                )
    print(f"Using {device} device")

    

    mlp = MLP().to(device)

    LEARNING_RATE = 0.005
    MOMENTUM = 0.9

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(mlp.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

    for epoch in range(15):
        train_loss = train(mlp, train_loader, criterion, optimizer, device)
        test_acc = test(mlp, test_loader, device)
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
    


    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)

    """  
    with torch.no_grad():
        mlp.eval()
        varx = example_data.to(device)
        outputs = mlp.forward(varx)
        print(torch.exp(outputs[0])) 
        print (torch.exp()example_targets) """


# TODO add load and save functionality that accepts -load or -save flags
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

if __name__ == "__main__":
    main()