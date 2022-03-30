import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# HYPER-PARAMETERS
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
NB_EPOCHS = 100


# DATASET/DATALOADER
# Create both train and test sets
train_set = torchvision.datasets.FashionMNIST('.', train=True, transform=T.ToTensor(), download=True)
test_set = torchvision.datasets.FashionMNIST('.', train=False, transform=T.ToTensor(), download=True)

# Create both train and test loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

# create the model, define its loss function, and an optimizaiton algorithm
model = torch.nn.Sequential(torch.nn.Flatten(),
                            torch.nn.Linear(28*28, 32),
                            torch.nn.ReLU(),
                            torch.nn.Linear(32, 10))

criterion = torch.nn.MSELoss()  # fonction de coût
# optimiseur, ce qui va dire comment les poids se mettent à jour
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# lr correspond à la taille des pas


# passe 100 fois sur la database (1 epoch = un parcours de toute la base)
for epoch in range(100):
    train_loss = 0.0
    for x, t in train_loader:  # pour chaque élement de la database

        y = model(x)  # donne l'ordonnée avec l'abcisse si j'ai compris

        # calcule l'erreur avec la prédiction et la target
        loss = criterion(y, t)
        loss.backward()  # calcule les deltas de chaque paramètre (w nous donne deltaw)
        optimizer.step()  # applique chaque delta à son paramètre (w + deltaw)
        optimizer.zero_grad()  # nettoie le deltaw, le remet à 0

        train_loss += loss/len(train_set)

    test_acc = 0.0
    test_loss = 0.0

    for x, t in test_loader:
        y=model(x)

        test_acc += (y.argmax(1) == t).sum()/len(test_set)
        test_loss += criterion(y, t)/len(test_set)
    
    print('Epoch: {}  |  Acc: {:.3f}%'.format(epoch, test_acc*100))