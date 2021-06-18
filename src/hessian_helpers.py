import copy

import numpy as np
import torch
from pyhessian import hessian

import src.utils as utils


def perturb_model_2d(model, model_per, direction_1, direction_2, alpha_1, alpha_2):
    """
    This function will return a model with parameters perturbed in the direction
    defined by alpha_1 * direction_1 + alpha_2 * direction_2
    Args:
        model: initial model
        model_per: copy of model, passed as parameter to save
        direction_1: top eigenvector of hessian
        direction_2: second top eigenvector of hessian
        alpha_1: amplitude of perturbation in direction_1
        alpha_2: amplitude of perturbation in direction_2

    Returns:
        model_per: perturbed model in two directions

    """
    for param_p, param_o, d_1, d_2 in zip(
            model_per.parameters(), model.parameters(), direction_1, direction_2
    ):
        with torch.no_grad():
            param_p.data = param_o.data + d_1 * alpha_1 + d_2 * alpha_2

    return model_per


def perturb_model(model, model_per, direction, alpha):
    """
    This function will return a model with parameters perturbed in a given direction
    with an magnitude equal to alpha
    Args:
        model: initial model
        model_per: copy of model, passed as parameter to save
        direction: top eigenvector
        alpha: magnitude of the perturbation
        
    Returns:
        model_per: perturbed model in one directioon
    """
    for param_p, param_o, d in zip(
            model_per.parameters(), model.parameters(), direction
    ):
        with torch.no_grad():
            param_p.data = param_o.data + d * alpha

    return model_per


def get_hessian(model, dataset_name, batch_size):
    """
    This function creates a Pyhessian object given which can be used to compute top_n 
    eigenvalues, density or trace of the hessian of the model
    Args:
        model: nn.Module
        dataset_name: name of the dataset used to train the model
        batch_size: size of the batch used to create the Pyhessian object
        
    Returns:
        model_hessian: Pyhessian object
        inputs: inputs used to create model_hesssian
        targets: targets used to create model_hesssian
        criterion: loss criterion
    
    """
    # Work with only one (big) batch
    train_loader, _ = utils.load_data(
        "/home/app/datasets", batch_size=batch_size, dataset=dataset_name
    )
    inputs, targets = next(iter(train_loader))
    criterion = torch.nn.CrossEntropyLoss()
    model_hessian = hessian(model, criterion, data=(inputs, targets), cuda=False)

    return model_hessian, inputs, targets, criterion


def compute_minimum_shape(
        model, dataset_name, max_amp_pert=0.5, num_per=25, batch_size=2000, use_3d=False
):
    """
    This function compute the 2D or 3D shape of the local minimum reached by the model.
    This is done by perturbing the model's weights along the direction of the top eigenvectors
    of its hessian.
    Args:
        model: nn.Module
        dataset_name: dataset name used to train the model
        max_amp_pert: maximum amplitude of the perturbation
        num_per: number of perturbations
        batch_size: batch_size used to create a Pyhessian object. The latter is used
                    to approximate the top_n eigenvalues
        use_3d: if set to true, the model is perturbed along the direction of the top 2 eigenvectors
                of its hessian
        
    Returns:
        alphas: array containing the amplitudes of perturbations
        losses: array of losses for each perturbation
    """
    model_hessian, inputs, targets, criterion = get_hessian(model, dataset_name, batch_size)
    top_n = 2 if use_3d else 1
    top_eigenvalues, top_eigenvectors = model_hessian.eigenvalues(top_n=top_n)

    alphas = np.linspace(-max_amp_pert, max_amp_pert, num_per).astype(np.float32)
    losses = []
    model_per = copy.deepcopy(model)

    for alpha in alphas:
        if use_3d:
            for alpha_2 in alphas:
                model_per = perturb_model_2d(
                    model,
                    model_per,
                    top_eigenvectors[0],
                    top_eigenvectors[1],
                    alpha,
                    alpha_2,
                )
                losses.append(criterion(model_per(inputs), targets).item())
        else:
            model_per = perturb_model(model, model_per, top_eigenvectors[0], alpha)
            losses.append(criterion(model_per(inputs), targets).item())

    return alphas, losses


def compute_hessian_trace(model, dataset_name, batch_size=2000):
    """
    This function return the trace of the hessian of a model.
    Args:
        model: nn.module
        dataset_name: dataset name used to train the model
        batch_size:  batch_size used to create a Pyhessian object
    Return:
        output: return the diagonal of the hessian matrix
    """
    model_hessian, _, _, _ = get_hessian(model, dataset_name, batch_size)
    return model_hessian.trace()


def compute_esd(model, dataset_name, batch_size=2000):
    """
    This function return the eigenvalue spectral density of the hessian of the model.
    Args:
        model: nn.module
        dataset_name: dataset name used to train the model
        batch_size:  batch_size used to create a Pyhessian object
    Return:
        output: return an array containing the range of eigenvalues and another containing the
                density of each eigenvalue.
    """
    model_hessian, _, _, _ = get_hessian(model, dataset_name, batch_size)
    return model_hessian.density()
