# MKKBOI005 - Tumi Mokoka
# Convolutional Neural network 

import sys
import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as Fun

#defining the MLP architecture
#TODO use convnet calculator to double check number of inputs and outputs for convolutions
# TODO implement data augmentation and dropout

class CNN (nn.Module):
    def __init__ (self):
        super().__init__()
        # nn.conv2d --> in_channels, out_channels, kernel_size (single int), stride
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d (6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120) #input is 3-channel 32x32 image
        self.fc2 = nn.Linear (120, 84) # First HL
        self.fc3 = nn.Linear (84 ,10) # Second HL
        self.output = nn.LogSoftmax (dim =1) # output 
        
        self.BN1 = nn.BatchNorm1d(120)
        self.BN2 = nn.BatchNorm1d(84)
        self.dropout = nn.Dropout(p = 0.1)
        

    def forward(self, x):
        # x = Batch 
        x = Fun.relu(self.conv1(x))
        x = self.pool(x)
        x = Fun.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,1) #flatten output tensor from convolution layers
        x = Fun.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.BN1(x)
        x = Fun.relu(self.fc2(x))
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
        transforms.RandomRotation(degrees = 5), #random data augmentation for rotations
        #transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.1),
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

    #creating the convolutional network
    cnn = CNN().to(device)

    #loading a pre-trained model
    if load:
        cnn.load_state_dict(torch.load("cnn_model.pt"))
        print("Model loaded")
        test_acc = test(cnn, test_loader, device)
        print(f"Test accuracy = {test_acc:.4f}")
    
    #training a new model
    else:   
        LEARNING_RATE = 0.005
        MOMENTUM = 0.9

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(cnn.parameters(), lr= LEARNING_RATE, momentum = MOMENTUM, weight_decay= 1e-6)

        # train for 15 epochs
        for epoch in range(15):
            train_loss = train(cnn, train_loader, criterion, optimizer, device)
            test_acc = test(cnn, test_loader, device)
            print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")


    # Save model
    if save:
        torch.save(cnn.state_dict(), "cnn_model.pt")
        print("CNN Model saved")

if __name__ == "__main__":
    main()