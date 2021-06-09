import torch

import src.utils as utils


def run_experiment(dataset, optimizer, batch_size=100, data_dir='/home/app/datasets', seed=12345, **kwargs):
    torch.manual_seed(seed)
    results = []
    train_loader, test_loader = utils.load_data(data_dir, batch_size=batch_size, dataset=dataset)

    training_losses, test_losses, training_accuracies, test_accuracies = utils.train(train_loader, test_loader,
                                                                                             optimizer=optimizer, **kwargs)
    results.append({'dataset': dataset, 'optimizer': optimizer, 'training_losses': training_losses,
                            'test_losses': test_losses, 'training_accuracies': training_accuracies,
                            'test_accuracies': test_accuracies})

    return results

