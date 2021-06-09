import torch
from apollo import Apollo
from optim_adahessian import Adahessian

import src.utils as utils


def run_experiment(dataset, optimizer, batch_size=100, num_epochs=10, data_dir='/home/app/datasets', seed=12345,
                   use_gpu=True, create_graph=False, **kwargs):
    torch.manual_seed(seed)
    train_loader, test_loader = utils.load_data(data_dir, batch_size=batch_size, dataset=dataset)

    training_losses, test_losses, training_accuracies, test_accuracies = utils.train(train_loader, test_loader,
                                                                                     use_gpu=use_gpu,
                                                                                     create_graph=create_graph,
                                                                                     num_epochs=num_epochs,
                                                                                     optimizer=optimizer, **kwargs)
    return {'dataset': dataset, 'optimizer': optimizer, 'training_losses': training_losses,
            'test_losses': test_losses, 'training_accuracies': training_accuracies,
            'test_accuracies': test_accuracies}


def run_experiment_all_optimizers(dataset, optimizers, num_epochs=100, save_dir='/home/app/results'):
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

    results = []
    for optimizer in optimizers:
        results.append(run_experiment(dataset, optimizer, batch_size=batch_size, num_epochs=num_epochs,
                                      **optimizer_args[optimizer]['kwargs']))

    return results
