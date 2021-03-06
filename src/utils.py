import logging
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.model import LeNet5


def load_data(data_dir, batch_size=100, dataset=None):
    """
    Load a dataset, and download it if necessary
    Args:
        data_dir: directory where to store the data
        batch_size: batch size to use
        dataset: dataset name , one of ['MNIST, 'Fashion_MNIST, 'CIFAR']
    Returns:
        two torch.data.DataLoaders, train and test in that order
    """
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
          optimizer=torch.optim.Adam, use_gpu=True, create_graph=False, **kwargs):
    """
    Training function. If the test dataset is provided, the function will compute the test loss and accuracy

    Args:
        train_loader: training set
        test_loader: test set
        num_epochs: num of epochs to train on
        model: torch.nn.model
        criterion: loss criterion used
        optimizer: torch.optim.Optimizer, optimizer to train with
        use_gpu: boolean, whether to use the gpu or not
        create_graph: boolean, used by some optimizers in the backward function

    Returns:
        training losses, test losses, training accuracies, test accuracies, final trained model

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
                loss.backward(create_graph=create_graph)
                return loss

            loss = closure().item()
            running_loss += loss * x.size(0)
            optimizer.step(closure=closure)
            num_train_correct_class += get_num_correct_class(model(x), target).item()

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
                    num_test_correct_class += get_num_correct_class(prediction, target).item()

                epoch_loss = running_loss / len(test_loader.dataset)
                epoch_accuracy = 1.0 * num_test_correct_class / len(test_loader.dataset)
                logging.info("Validation: Epoch = {0}, loss = {1}, Accuracy {2}"
                             .format(epoch, epoch_loss, epoch_accuracy))
                test_losses.append(epoch_loss)
                test_accuracies.append(epoch_accuracy)
    torch.cuda.empty_cache()

    return training_losses, test_losses, training_accuracies, test_accuracies, model


def get_num_correct_class(pred, targets):
    """
    Compute the accuracy of predictions
    Args:
        pred: predicted labels
        targets: true labels

    Returns: Accuracy

    """
    probs = F.softmax(pred, dim=1)
    labels = torch.argmax(probs, dim=1)
    return sum([label == target for (label, target) in zip(labels, targets)])
