import logging
import pickle
import sys
import torch
from apollo import Apollo
from optim_adahessian import Adahessian
from torch.optim import LBFGS

from src.exp import run_experiment_all_optimizers

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s")

optimizers = [torch.optim.Adam, LBFGS, Adahessian, Apollo]

datasets = ['MNIST', 'Fashion_MNIST', 'CIFAR']

dataset, num_epochs = sys.argv[1], int(sys.argv[2])
results = run_experiment_all_optimizers([dataset], optimizers, num_epochs=num_epochs)
save_path = '/home/app/results/' + dataset + "_results.dic"
with open(save_path, "wb") as file:
    pickle.dump(results, file)
