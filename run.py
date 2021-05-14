from Utils import *

logging.getLogger(__name__).setLevel(logging.INFO)

def main():
    _ = training_loop(10, 100)


if __name__ == "__main__":
    main()
