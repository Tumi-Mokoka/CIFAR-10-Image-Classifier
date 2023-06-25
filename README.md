# CIFAR-10-Image-Classifier
___
[a set of neural networks designed to classify images from the CIFAR-10 dataset which are accessible here (https://www.cs.toronto.edu/~kriz/cifar.html)
___

## Execution 

the respective files are run using the standard python3 execution ("python3 MLP.py" in the command line) 
- when the code in the python files is run the respective neural network is built and then tested

if you would like to save the model that is trained then an alternate executable format us used
- "python MODEL_NAME.py [-load | -save]" where MODEL_NAME is the name of the model (MLP, CNN or RESNET)
- using "-load" will load the existing model into the program and test its performance on the test data
- using "-save" will train a new model and save it's parameters in a new file which can be used in the future

### 1. MLP
this is an implementation of a standard feed-forward Multi-Layer Perceptron [MLP]


### 2. CNN
this is an impementation o the LeNet5 convolutional network [LeCun et al.] 

this includes the following 
- a convolutional layer with a 5x5 filter and 6 output channels
- a MaxPool subsampling layer with a 2x2 filter and a stride of 2
- another convolutional layer with filter size 5x5, 6 input channels and 16 output channels
- a MaxPool subsampling layer with a 2x2 filter and a stride of 2
- a fully connected layer (120,84)
- a final fully connected layer (84,10)


