import logging
import torch
from apollo import Apollo
from optim_adahessian import Adahessian

import src.utils as utils


def run_experiment(dataset, optimizer, optimizer_name, batch_size=100, num_epochs=10, data_dir='/home/app/datasets',
                   seed=12345,
                   use_gpu=True, create_graph=False, **kwargs):
    """
    Runs a single training experiment, i.e one dataset and one optimizer
    Args:
        dataset: dataset name, one of ['MNIST', 'Fashion_MNIST', 'CIFAR']
        optimizer: torch.optim.optimizer
        optimizer_name: name of the optimizer
        batch_size: batch size for training
        num_epochs: num epochs to train
        data_dir: directory containing the dataset, or where to store a downloaded dataset
        seed: seed to set for reproducibility
        use_gpu: whether to try using the gpu or not
        create_graph: used by the training function, when computing the gradient
        **kwargs: kwargs to pass to the utils.train function
    Returns:
        Dictionary of results
    """
    torch.manual_seed(seed)

    logging.info("Dataset: {0}, Optimizer: {1}".format(dataset, optimizer_name))

    train_loader, test_loader = utils.load_data(data_dir, batch_size=batch_size, dataset=dataset)

    training_losses, test_losses, training_accuracies, test_accuracies, model = \
        utils.train(train_loader, test_loader, use_gpu=use_gpu, create_graph=create_graph,
                    num_epochs=num_epochs, optimizer=optimizer, **kwargs)

    return {'training_losses': training_losses, 'test_losses': test_losses,
            'training_accuracies': training_accuracies, 'test_accuracies': test_accuracies,
            'model': model}


def run_experiment_all_optimizers(datasets, optimizers, num_epochs=100, save_dir='/home/app/results'):
    """
    Runs training for a set of datasets on a set of optimizers, uses hard-coded parameters for each optimizer
    Args:
        datasets: list of dataset names
        optimizers: list of optimizers
        num_epochs: number of epochs to train on
        save_dir: where to find the dataset or download it
    Returns:
    list of dicts, representing the results of each experiment
    """
    # Arguments specific to each optimizer
    optimizer_args = {
        torch.optim.LBFGS: {'name': 'LBFGS', 'kwargs': {'history_size': 5, 'max_iter': 4,
                                                        'line_search_fn': 'strong_wolfe'}},
        Apollo: {'name': 'Apollo', 'kwargs': {'lr': 0.5}},
        Adahessian: {'name': 'AdaHessian', 'kwargs': {'create_graph': True}},
        torch.optim.Adam: {'name': 'Adam', 'kwargs': {}}
    }

    batch_size = 100
    # Use a large batch size to reduce Hessian estimation noise for L-BFGS
    if torch.optim.LBFGS in optimizers:
        batch_size = 1000

    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for optimizer in optimizers:
            results[dataset][optimizer_args[optimizer]['name']] = \
                run_experiment(dataset, optimizer, optimizer_args[optimizer]['name'],
                               batch_size=batch_size, num_epochs=num_epochs, **optimizer_args[optimizer]['kwargs'])

    return results
