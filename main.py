from parser import parse_args

import torch

from train2d import train2d

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def main():
    args = parse_args()
    train2d(args)


if __name__ == "__main__":
    main()
