import logging
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from model import *

def load_data(data_dir, batch_size):
    from torchvision import transforms,datasets
    from torch.utils.data import DataLoader

    transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor()])

    # download and create datasets
    train_dataset = datasets.MNIST(root=data_dir,
                                   train=True,
                                   transform=transforms,
                                   download=True)

    valid_dataset = datasets.MNIST(root=data_dir,
                                   train=False,
                                   transform=transforms)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    return train_loader, valid_loader


def train(train_loader, model, criterion, optimizer, epoch):
    '''
    Function for the training step of the training loop
    '''

    logging.info("Training: Epoch {}".format(epoch))
    model.train()
    running_loss = 0

    for step, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        prediction = model(x)
        loss = criterion(prediction, target)
        running_loss += loss.item() * x.size(0)
        # Backward pass
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            logging.info("Step = {0}, Training loss = {1}".format(step, loss))

    epoch_loss = running_loss / len(train_loader.dataset)
    logging.info("Epoch = {0}, Training Epoch loss = {1}".format(epoch, epoch_loss))
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, epoch):
    '''
    Function for the validation step of the training loop
    '''

    logging.info("Validation: Epoch {}".format(epoch))
    model.eval()
    running_loss = 0

    for x, target in valid_loader:
        # Forward pass
        prediction = model(x)
        loss = criterion(prediction, target)
        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    logging.info("Epoch = {0}, Validation Epoch loss = {1}".format(epoch, epoch_loss))

    return model, epoch_loss


def get_accuracy(model, data_loader):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''

    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for x, target in data_loader:
            probs = model(x)
            _,predicted_labels = torch.max(probs, 1)

            n += target.size(0)
            correct_pred += (predicted_labels == target).sum()

    return 1.0 * correct_pred / n


def training_loop(epochs, batch_size):
    '''
    Function defining the entire training loop
    '''

    train_loader,valid_loader  = load_data("dataset", batch_size)

    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        train_losses.append(train_loss)

        # Predict over training set
        train_accuracy = get_accuracy(model, train_loader)
        logging.info("Epoch {0}, Training Accuracy = {1}".format(epoch, train_accuracy))

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, epoch)
            valid_losses.append(valid_loss)

        # Predict over validation set
        validation_accuracy = get_accuracy(model, valid_loader)
        logging.info("Epoch {0}, Validation Accuracy = {1}".format(epoch, validation_accuracy))

    plot_losses(train_losses, valid_losses)


    return model, optimizer, (train_losses, valid_losses)


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.show()

    # change the plot style to default
    plt.savefig("accuracy_plot")

