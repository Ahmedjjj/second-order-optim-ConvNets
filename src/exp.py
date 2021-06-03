import torch

import src.utils as utils


def run_experiments(datasets, optimizers, batch_size=100, data_dir='/home/app/datasets', seed=12345):
    torch.manual_seed(seed)
    results = []
    for dataset in datasets:
        train_loader, test_loader = utils.load_data(data_dir, batch_size=batch_size, dataset=dataset)
        for optimizer in optimizers:
            training_losses, test_losses, training_accuracies, test_accuracies = utils.train(train_loader, test_loader,
                                                                                             optimizer=optimizer)
            results.append({'dataset': dataset, 'optimizer': optimizer, 'training_losses': training_losses,
                            'test_losses': test_losses, 'training_accuracies': training_accuracies,
                            'test_accuracies': test_accuracies})

    return results

