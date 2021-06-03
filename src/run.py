from Utils import *

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s")

def main():
    # Available datasets "MNIST", "Fashion_MNIST", "CIFAR"
    train_loader, evaluation_loader = load_data("datasets", batch_size=100, dataset="Fashion_MNIST")
    _ = train(train_loader, evaluation_loader)


if __name__ == "__main__":
    main()
