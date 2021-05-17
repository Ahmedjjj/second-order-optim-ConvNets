import logging
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from model import *
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def load_data(data_dir, batch_size=100, dataset=None):

    transformer = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor()])

    if dataset == "CIFAR":
        dataset_loader = datasets.CIFAR10
        transformer = transforms.Compose([transforms.Grayscale(),
                                         transforms.Resize((32, 32)),
                                         transforms.ToTensor()])

    elif dataset == "Fashion_MNIST":
        dataset_loader = datasets.FashionMNIST
    else:
        dataset_loader = datasets.MNIST

    # download and create datasets
    train_dataset = dataset_loader(root=data_dir,
                                   train=True,
                                   transform=transformer,
                                   download=True)

    valid_dataset = dataset_loader(root=data_dir,
                                   train=False,
                                   transform=transformer,
                                   download=False)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    return train_loader, valid_loader


def train(train_loader, eval_loader=None, num_epochs=10, model=None, criterion=None, optimizer=None):
    '''
    Training function. If the evaluation dataset is provided, the function will compute the evaluation loss and accuracy
    '''

    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []



    if model is None:
        model = LeNet5()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        num_train_correct_class = 0

        for step, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            prediction = model(x)
            # compute loss
            loss = criterion(prediction, target)
            running_loss += loss.item() * x.size(0)
            # Backward pass
            loss.backward()
            optimizer.step()
            num_train_correct_class += get_num_correct_class(prediction, target)
            if step % 100 == 99:
                logging.info("Training: Epoch = {0}, Step = {1}, loss = {2}".format(epoch, step+1, loss))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 1.0 * num_train_correct_class / len(train_loader)
        logging.info("Training: Epoch = {0}, loss = {1}, Accuracy {2}"
                     .format(epoch, epoch_loss, epoch_accuracy))

        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)

        if eval_loader is not None:
            model.eval()
            running_loss = 0
            num_validation_correct_class = 0

            for x, target in eval_loader:
                # Forward pass
                prediction = model(x)
                loss = criterion(prediction, target)
                running_loss += loss.item() * x.size(0)
                num_validation_correct_class += get_num_correct_class(prediction, target)

            epoch_loss = running_loss / len(eval_loader.dataset)
            epoch_accuracy = 1.0 * num_validation_correct_class / len(eval_loader)
            logging.info("Validation: Epoch = {0}, loss = {1}, Accuracy {2}"
                         .format(epoch, epoch_loss, epoch_accuracy))
            validation_losses.append(epoch_loss)
            validation_accuracies.append(epoch_accuracy)

    plot_losses(training_losses, validation_losses, training_accuracies, validation_accuracies)
    return training_losses, validation_losses, training_accuracies, validation_accuracies


def get_num_correct_class(pred, targets):
    _, labels = torch.max(pred, 1)
    return sum([label == target for (label, target) in zip(labels, targets)])


def plot_losses(train_losses, valid_losses, training_accuracies, validation_accuracies):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn

    fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))

    ax[0].plot(train_losses, color='blue', label='Training loss')
    ax[0].plot(valid_losses, color='red', label='Validation loss')
    ax[0].set(title="Loss over epochs", xlabel='Epoch', ylabel='Loss')
    ax[0].legend()

    ax[1].plot(training_accuracies, color='blue', label='Training accuracy')
    ax[1].plot(validation_accuracies, color='red', label='Validation accuracy')
    ax[1].set(title="Accuracy over epochs", xlabel='Epoch', ylabel='Accuracy')
    ax[1].legend()

    fig.show()

    # change the plot style to default
    plt.savefig("accuracy_plot")
