# Second-order-optim-ConvNets

##### By Ahmed J., Rami A.

This code is our work to EPFL's CS-439 Optimization for Machine Learning course Project.

## Dataset
We use the CIFAR10, MNIST and Fashion-MNIST datasets. We download these automatically
 if we don't have access to the code.

## Dependencies
We recommend using `docker` to install all the dependencies.  
We include a Dockerfile as well as a Makefile with a target `image` in order to build the image:  
You can run `make image` from this directory.

## Reproducibility
All results presented in the report can be reproduced my running `make run-experiments`.  
This will train the network on all datasets for all optimizers and save the results as pickled dictionaries
in the results folder. We use a seed at each iteration to control for randomness.
We include pickle files from all our last run since the experiments take a while to finish.  
The plots can be visualized in the two notebooks:
- `notebooks/loss_visualization.ipynb` which will show the evolution of the loss/accuracy across epochs.
- `notebooks/curvature.ipynb` which contains visualization of the local minima found.
Whenever possible the notebooks will be pre-run as some of the processing takes time.  
Please run the notebooks by running `make notebook` and accessing `localhost:8888` 

## 3D interactive plots
We include 3D interactive plots of the loss landscape for the three datasets and four optimizers.  
These can be found in `notebooks/curvature.ipynb`. Please enable WebGL in your browser in order to view them  
properly.

## Folders
- `src/`: contains Python .py files.
- `notebooks/`: contains Jupyter notebooks for out visualizations.
- `results/`: contains pickle files of our results.

## Files
- `Makefile Dockerfile .gitignore .dockerignore`: usual project setup files.
- `src/exp.py` : functions to execute experiments.
- `src/hessian_helpers.py`: functions computing second order (Hessian) information.
- `src/model.py`: LeNet5 model definition. 
- `src/utils.py`: training utils.
- `src/main.py`: main entry-point for running experiments.
- `notebooks/loss_visualization.ipynb`: visualizations of the loss across epochs.
- `notebooks/curvature.ipynb`: visualizations of the shape of the local minima.

## Contact
In case any help is needed:
- Ahmed Jellouli : ahmed.jellouli@epfl.ch
- Rami Azouz : rami.azouz@epfl.ch
