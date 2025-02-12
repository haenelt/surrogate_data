# -*- coding: utf-8 -*-
"""Python package for the generation of null distributions with preserved spatial
autocorrealation."""

from argparse import SUPPRESS, ArgumentParser


def _get_parser() -> ArgumentParser:
    """Parse commandline arguments."""
    # Disable default help
    parser = ArgumentParser(add_help=False)
    optional = parser.add_argument_group("optional arguments")

    # Add required arguments
    parser.add_argument(
        "--subj_id", dest="subj_id", type=int, help="Subject ID (1,...,8)."
    )

    # Add back help
    optional.add_argument(
        "--help",
        action="help",
        default=SUPPRESS,
        help="Show this help message and exit.",
    )
    optional.add_argument(
        "--arch",
        dest="arch",
        type=str,
        help=("Use linear, punet  or unet model architecture (default: %(default)s)."),
        default="unet",
    )
    optional.add_argument("--params", nargs="*", help="Arbitrary key=value pairs")

    return parser


def make_surrogate():
    """Workflow for surrogate map generation."""
    pass


def _main(argv: tuple | None = None):
    """Perform surrogate map generation."""
    args = _get_parser().parse_args(argv)
    params = dict(param.split("=") for param in args.params) if args.params else {}
    print(args.subj_id)
    print(params)
    # make_surrogate(**kwargs)


if __name__ == "__main__":
    _main()
