import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

from guidance.GuideDict import GuideDict, GuideList


@dataclass
class Config:
    """Diffusion Configuration Data Class.

    Attributes:
        mode (str): Type of SDS.
        prompt (str): Guiding prompt.
        negative_prompt (str): Negative prompt.
        guide (GuideType): Guidance model class
        iterations (int): Number of training iterations.
        lr (float): SDS learning rate.
        t_range (float): Diffusion sampling interval in [0, 1].
        guidance_scale (float): Classifier-free guidance weight.
        output_path (Path): Checkpoint and file output path
        device (torch.device): Device where training occurs.
        dtype (torch.dtype): Precision of training.
    """

    mode: str
    prompt: str
    negative_prompt: str
    guide: type
    iterations: int
    lr: float
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
        assert len(t_range) == 2
        return t_range
    except (TypeError, AssertionError):
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

    Args:
        arg_list (List[str], optional): Argument list to parse. Defaults to
            options passed from command line.

    Returns:
        Converted Config data class.
    """

    # Define parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="2d")
    parser.add_argument("--prompt", type=str, default="A dog")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guide", type=str, choices=GuideList, default="StableGuide")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--t_range", type=parse_t_range, default=(0.02, 0.98))
    parser.add_argument("--guidance_scale", type=float, default=25)
    parser.add_argument("--output_path", type=str, default="/output")

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
    if hasattr(args, "yaml"):
        logging.info("Overriding CLI args with YAML")
        yaml_arg_list = yaml_to_args(args.yaml)
        args = parser.parse_args(args=yaml_arg_list)

    return Config(
        mode=args.mode,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guide=GuideDict[args.guide],
        iterations=args.iterations,
        lr=args.lr,
        t_range=args.t_range,
        guidance_scale=args.guidance_scale,
        output_path=Path(args.output_path),
        device=torch.device(args.device),
        dtype=getattr(torch, args.dtype),
    )


# Run parser.py to test the parser
if __name__ == "__main__":
    print(parse_args())
