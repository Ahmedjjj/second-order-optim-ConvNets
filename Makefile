image:
	docker build -t optim:gpu . --build-arg user=$(shell id -u) \
								--build-arg usern=$(shell id -un)

notebook:
	docker run --rm --name optim_notebook -p 8888:8888 \
	-v  "$(shell pwd)":/home/app -w /home/app optim:gpu \
    jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token=''

notebook-gpu:
	docker run --gpus all --rm --name optim_notebook -p 8888:8888 \
	-v  "$(shell pwd)":/home/app -w /home/app optim:gpu \
    jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token=''

run-experiments:
	docker run --rm --name experiments -v "$(shell pwd)":/home/app -w /home/app optim:gpu \
	/bin/bash -c "python3 src/main.py MNIST 10 ; python3 src/main.py Fashion_MNIST 10; python3 src/main.py CIFAR 50"