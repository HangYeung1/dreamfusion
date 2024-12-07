import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

from guidance import GuideDict, GuideList


@dataclass
class Config:
    """Diffusion Configuration Data Class.

    Attributes:
        mode (str): SDS domain.
        prompt (str): Guiding prompt.
        negative_prompt (str): Negative prompt.
        guide (GuideType): Guidance model class.
        iterations (int): Number of training iterations.
        batch_size (int): Number of images to train.
        lr (float): Adam learning rate.
        weight_decay (float): Adam weight decay rate.
        t_range (float): Diffusion sampling interval.
        guidance_scale (float): Classifier-free guidance weight.
        output_path (Path): File output path.
        device (torch.device): Device of training.
        dtype (torch.dtype): Precision of training.
    """

    mode: str
    prompt: str
    negative_prompt: str
    guide: type
    iterations: int
    batch_size: int
    lr: float
    weight_decay: float
    t_range: float
    guidance_scale: float
    output_path: Path
    device: torch.device
    dtype: torch.dtype


def parse_t_range(range_str: str) -> Tuple[float, float]:
    """Parse Tuple[float, float] t_range from "{float},{float}"."""

    try:
        t_range = range_str.split(",")
        t_range = [float(t) for t in t_range]
        if not len(t_range) == 2:
            raise TypeError
        return t_range
    except TypeError:
        raise argparse.ArgumentTypeError(f'{range_str} isn\'t a range "float,float".')


def yaml_to_args(yaml_path: str) -> List[str]:
    """Convert YAML file to argparse argument list."""

    try:
        with open(yaml_path) as file:
            yaml_dict = yaml.safe_load(file)
            yaml_list = []
            for key, value in yaml_dict.items():
                yaml_list.append(str(f"--{key}"))
                yaml_list.append(str(value))
            return yaml_list
    except TypeError:
        raise argparse.ArgumentTypeError("Invalid YAML formatting.")


def parse_args(arg_list: None | List[str] = None) -> Config:
    """Parse SDS configuration from argparse argument list.

    Arguments:
        arg_list (List[str], optional): Argument list to parse. Defaults to
            options passed from command line.

    Returns:
        Converted Config data class.
    """

    # Define parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["2D", "3D"], default="2D")
    parser.add_argument("--prompt", type=str, default="A dog")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guide", type=str, choices=GuideList, default="StableGuide")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--t_range", type=parse_t_range, default=(0.02, 0.98))
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--output_path", type=str, default="output")

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "float64"],
        default="float16" if torch.cuda.is_available() else "float32",
    )

    parser.add_argument("--yaml", type=str)

    # Parse arguments and return
    args = parser.parse_args(args=arg_list)
    if args.yaml:
        logging.info("Overriding CLI args with YAML")
        yaml_arg_list = yaml_to_args(args.yaml)
        args = parser.parse_args(args=yaml_arg_list)

    return Config(
        mode=args.mode,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guide=GuideDict[args.guide],
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        t_range=args.t_range,
        guidance_scale=args.guidance_scale,
        output_path=Path(args.output_path),
        device=torch.device(args.device),
        dtype=getattr(torch, args.dtype),
    )


# Run parser.py to test the parser
if __name__ == "__main__":
    print(parse_args())
