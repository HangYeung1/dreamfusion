import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
import yaml

from guidance.GuideDict import GuideDict


@dataclass
class SDSConfig:
    """Diffusion Configuration Data Class.

    Attributes:
        prompt (str): Guiding prompt.
        guide (GuideType): Guidance model class
        iterations (int): Number of training iterations.
        lr (float): SDS learning rate.
        t_range (float): Diffusion sampling interval in [0, 1].
        guidance_scale (float): Classifier-free guidance weight.
        device (torch.device): Device where training occurs.
        dtype (torch.dtype): Precision of training.
    """

    prompt: str
    guide: type
    iterations: int
    lr: float
    t_range: float
    guidance_scale: float
    device: torch.device
    dtype: torch.dtype


def parse_positive_int(val_str: str) -> int:
    """Parse positive integer from string."""
    try:
        val = int(val_str)
        assert val > 0
        return val
    except (AssertionError, TypeError):
        raise argparse.ArgumentTypeError(
            f"{val_str} isn't a positive integer."
        )


def parse_positive_float(val_str) -> int:
    """Parse positive float from string."""
    try:
        val = float(val_str)
        assert val > 0
        return val
    except (AssertionError, TypeError):
        raise argparse.ArgumentTypeError(f"{val_str} isn't a positive float.")


def parse_guide(guide_str: str) -> type:
    """Parse GuideType from guide class name."""

    try:
        guide = GuideDict.get(guide_str)
        assert guide
        return guide
    except AssertionError:
        raise argparse.ArgumentTypeError(
            f"{guide_str} isn't the name of a guidance class."
        )


def parse_t_range(range_str: str) -> Tuple[float, float]:
    """Parse Tuple[float, float] t_range from "{float},{float}"."""

    try:
        t_range = range_str.split(",")
        assert len(t_range) == 2
        t_range = [float(t) for t in t_range]
        assert t_range[0] < t_range[1]
        return t_range
    except (TypeError, AssertionError):
        raise argparse.ArgumentTypeError(
            f'{range_str} isn\'t a non-empty range "float,float".'
        )


def parse_device(device_str: str) -> torch.device:
    """Parse torch device from device name."""

    try:
        return torch.device(device_str)
    except RuntimeError:
        raise argparse.ArgumentTypeError(
            f"{device_str} isn't a valid torch device."
        )


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse torch dtype from dtype name."""

    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise argparse.ArgumentTypeError(
            f"{dtype_str} isn't a valid torch data type."
        )


def yaml_to_args(yaml_path: str) -> List[str]:
    """Convert YAML file to argparse argument list."""

    try:
        with open(yaml_path) as file:
            yaml_dict = yaml.safe_load(file)
            yaml_list = ["cli"]
            for key, value in yaml_dict.items():
                yaml_list.append(str(f"--{key}"))
                yaml_list.append(str(value))
            return yaml_list
    except TypeError:
        raise argparse.ArgumentTypeError("Invalid YAML formatting.")


def parse_args(arg_list: None | List[str] = None) -> SDSConfig:
    """Parse SDS configuration from argparse argument list.

    The parser has two modes: cli and yaml. In cli mode, all SDS configuration
    options are passed as flags from the command line. In yaml mode, all options
    are passed through a pre-existing YAML file. For more information, run the
    parser with help flags.

    Args:
        arg_list (List[str], optional): Argument list to parse. Defaults to
            options passed from command line.

    Returns:
        Converted SDSConfig data class.
    """

    # Define main parser args
    parser = argparse.ArgumentParser(description="Configuration for SDS.")
    subparsers = parser.add_subparsers(
        description="""There are two methods to configure SDS: pass arguments 
        through the command line (cli) or pass arguments from a pre-made YAML 
        file (yaml).""",
        help="select cli or yaml to pass SDS arguments.",
        required=True,
    )

    # Define cli subparser args
    cli_parser = subparsers.add_parser(
        "cli",
        description="Configure SDS by passing each flag in the command line.",
        help="configuration via CLI.",
    )
    cli_parser.add_argument(
        "--prompt", type=str, help="guiding prompt.", required=True
    )
    cli_parser.add_argument(
        "--guide",
        type=parse_guide,
        help="name of diffusion guidance class.",
        required=True,
    )
    cli_parser.add_argument(
        "--iterations",
        type=parse_positive_int,
        help="number of training iterations.",
        required=True,
    )
    cli_parser.add_argument(
        "--lr",
        type=parse_positive_float,
        help="training learning rate.",
        required=True,
    )
    cli_parser.add_argument(
        "--t_range",
        type=parse_t_range,
        help='"float,float" of diffusion time range.',
        required=True,
    )
    cli_parser.add_argument(
        "--guidance_scale",
        type=parse_positive_float,
        help="float of classifier free guidance scale.",
        required=True,
    )
    cli_parser.add_argument(
        "--device",
        type=parse_device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="torch device of training.",
    )
    cli_parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default=torch.float16 if torch.cuda.is_available() else torch.float32,
        help="torch data type of training.",
    )

    # Define yaml subparser args
    yaml_parser = subparsers.add_parser(
        "yaml",
        description="Configure SDS by passing preset flags from a YAML file.",
        help="configuration via YAML",
    )
    yaml_parser.add_argument(
        "yaml_path",
        type=str,
        help="path to YAML config file.",
    )

    # Parse the arguments
    args = parser.parse_args(args=arg_list)
    if hasattr(args, "yaml_path"):
        yaml_args = yaml_to_args(args.yaml_path)
        args = parser.parse_args(args=yaml_args)

    return SDSConfig(
        prompt=args.prompt,
        guide=args.guide,
        iterations=args.iterations,
        lr=args.lr,
        t_range=args.t_range,
        guidance_scale=args.guidance_scale,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    print(parse_args())
