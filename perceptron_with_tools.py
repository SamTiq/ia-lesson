import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# HYPER-PARAMETERS
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NB_EPOCHS = 100


# DATASET/DATALOADER
# Create both train and test sets
train_set = torchvision.datasets.FashionMNIST('.', train=True, transform=T.ToTensor(), download=True)
test_set = torchvision.datasets.FashionMNIST('.', train=False, transform=T.ToTensor(), download=True)

# Create both train and test loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)


# MODEL (Perceptron)
# Define the weight of the model
weight = torch.zeros(28*28, 10)
# Define the bias of the model
bias = torch.zeros(1, 10)

model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(28*28, 32), torch.nn.ReLU(), torch.nn.Linear(32,10))

# ITERATIONS
for epoch in range(NB_EPOCHS):
    # TRAIN
    for x, t in train_loader:
        # PREPROCESS
        # Flatten the images from 28x28 to 784
        x = torch.nn.Flatten()(x)
        # Create one-hot vectors from the targets
        t = F.one_hot(t, num_classes=10)

        # FORWARD
        # Compute the output of the model for the input x
        y = model(x)

        # BACKWARD
        # Compute the gradient (minus the derivative of the error) on the output layer
        # Note: Here the error is (t - y)^2
        grad = 2*(t-y)
        # Compute delta weight and update the weight 
        weight += (x.T@grad) * LEARNING_RATE
        # Compute delta bias and update the bias 
        bias += grad.sum(0) * LEARNING_RATE


    # TEST
    test_acc = 0.0
    for x, t in test_loader:
        # PREPROCESS
        # Flatten the images from 28x28 to 784
        x = torch.nn.Flatten()(x)
        
        # FORWARD
        # Compute the output of the model for the input x
        y = x@weight + bias
        
        # COMPUTE ACCURACY
        test_acc += (y.argmax(1) == t).sum()/len(test_set)


    # DISPLAY
    print('Epoch: {}  |  Acc: {:.3f}%'.format(epoch, test_acc*100))