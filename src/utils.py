import logging
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.model import LeNet5


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

    test_dataset = dataset_loader(root=data_dir,
                                  train=False,
                                  transform=transformer,
                                  download=True)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader


def train(train_loader, test_loader=None, num_epochs=10, model=LeNet5, criterion=nn.CrossEntropyLoss,
          optimizer=torch.optim.Adam, use_gpu=True, **kwargs):
    """
    Training function. If the evaluation dataset is provided, the function will compute the evaluation loss and accuracy

    Args:
        train_loader:
        test_loader:
        num_epochs:
        model:
        criterion:
        optimizer:
        use_gpu:
    """

    training_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []

    # Use gpu if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    model = model().to(device)
    criterion = criterion()
    optimizer = optimizer(model.parameters(), **kwargs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        num_train_correct_class = 0

        for step, (x, target) in enumerate(train_loader):
            x, target = x.to(device), target.to(device)

            def closure():
                optimizer.zero_grad()
                # Forward pass
                prediction = model(x)
                # compute loss
                loss = criterion(prediction, target)
                # Backward pass
                loss.backward()
                return loss

            loss = closure().item()
            running_loss += loss * x.size(0)
            optimizer.step(closure)
            num_train_correct_class += get_num_correct_class(model(x), target)
            if step % 100 == 99:
                logging.info("Training: Epoch = {0}, Step = {1}, loss = {2}".format(epoch, step + 1, loss))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 1.0 * num_train_correct_class / len(train_loader.dataset)
        logging.info("Training: Epoch = {0}, loss = {1}, Accuracy {2}"
                     .format(epoch, epoch_loss, epoch_accuracy))

        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)

        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                model.eval()
                running_loss = 0
                num_test_correct_class = 0

                for x, target in test_loader:
                    x, target = x.to(device), target.to(device)
                    # Forward pass
                    prediction = model(x)
                    loss = criterion(prediction, target)
                    running_loss += loss.item() * x.size(0)
                    num_test_correct_class += get_num_correct_class(prediction, target)

                epoch_loss = running_loss / len(test_loader.dataset)
                epoch_accuracy = 1.0 * num_test_correct_class / len(test_loader.dataset)
                logging.info("Validation: Epoch = {0}, loss = {1}, Accuracy {2}"
                             .format(epoch, epoch_loss, epoch_accuracy))
                test_losses.append(epoch_loss)
                test_accuracies.append(epoch_accuracy)

    plot_losses(training_losses, test_losses, training_accuracies, test_accuracies)
    return training_losses, test_losses, training_accuracies, test_accuracies


def get_num_correct_class(pred, targets):
    probs = F.softmax(pred, dim=1)
    labels = torch.argmax(probs, dim=1)
    return sum([label == target for (label, target) in zip(labels, targets)])


def plot_losses(train_losses, valid_losses, training_accuracies, validation_accuracies):
    """
    Function for plotting training and validation losses
    """

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
