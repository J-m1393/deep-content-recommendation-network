import argparse
from .train import main as train_main
from .infer import main as infer_main

# Simple dispatcher: python -m recommender train|infer ...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["train", "infer"])
    args, _ = parser.parse_known_args()

    if args.cmd == "train":
        train_main()
    else:
        infer_main()

if __name__ == "__main__":
    main()
