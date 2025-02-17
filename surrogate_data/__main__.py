# -*- coding: utf-8 -*-
"""Python package for the generation of null distributions with preserved spatial
autocorrealation."""

from argparse import SUPPRESS, ArgumentParser
from pathlib import Path

import numpy as np

from surrogate_data.data import load_image
from surrogate_data.surrogate import SurrogateMap


def _get_parser() -> ArgumentParser:
    """Parse commandline arguments."""
    # Disable default help
    parser = ArgumentParser(add_help=False)
    optional = parser.add_argument_group("optional arguments")

    # Add required arguments
    parser.add_argument(
        "--in",
        dest="file_in",
        type=str,
        help="File name of input image.",
    )
    parser.add_argument(
        "--out",
        dest="file_out",
        type=str,
        help="File name (*.npy) of generated surrogate maps.",
    )
    parser.add_argument(
        "--n", dest="n_surr", type=int, help="Number of generated surrogate maps."
    )

    # Add back help
    optional.add_argument(
        "--help",
        action="help",
        default=SUPPRESS,
        help="Show this help message and exit.",
    )
    optional.add_argument(
        "--bw",
        dest="bw",
        type=float,
        help="Bandwidth for smoothed variogram computation (default: %(default)s).",
        default=1.0,
    )
    optional.add_argument(
        "--kernel",
        dest="kernel",
        type=str,
        choices=["gaussian", "exp", "uniform"],
        help="Distance-depending smoothing kernel (default: %(default)s).",
        default="gaussian",
    )
    optional.add_argument(
        "--lags",
        dest="lags",
        type=list,
        help=(
            "List of lags for variogram computation. The list is expected to have "
            "three entries [start, stop, num] (default: %(default)s)."
        ),
        default=[],
    )
    optional.add_argument(
        "--ksmooth",
        dest="k_smooth",
        type=list,
        help=(
            "List of smoothing kernels, for which variogram fitting is computed "
            "(default: %(default)s)."
        ),
        default=[],
    )
    optional.add_argument(
        "--num_cores",
        dest="num_cores",
        type=int,
        help=("Number of cores (default: %(default)s)."),
        default=1,
    )

    return parser


def make_surrogate(**kwargs) -> None:
    """Workflow for surrogate map generation."""

    # read data
    data = load_image(kwargs["file_in"])

    # parameters
    if kwargs.get("k_smooth"):
        start, stop, num_points = kwargs["k_smooth"]
        _k_smooth = np.linspace(start, stop, num_points)
    else:
        _k_smooth = SurrogateMap.DEFAULT_K_SMOOTH

    if kwargs.get("lags"):
        start, stop, num_points = kwargs["lags"]
        _lags = np.linspace(start, stop, num_points)
    else:
        _lags = SurrogateMap.DEFAULT_LAGS

    kernel = kwargs.get("kernel", SurrogateMap.DEFAULT_KERNEL)
    bw = kwargs.get("bw", SurrogateMap.DEFAULT_BW)
    num_cores = kwargs.get("num_cores", SurrogateMap.DEFAULT_NUM_CORES)

    surr = SurrogateMap(
        data, k_smooth=_k_smooth, lags=_lags, kernel=kernel, bw=bw, num_cores=num_cores
    )

    # fit variogram
    _ = surr.fit()

    # generate new surrogate maxp
    surr_data = surr.generate(kwargs["n_surr"])

    # save to disk
    Path(kwargs["file_out"]).parent.mkdir(exist_ok=True, parents=True)
    np.save(kwargs["file_out"], surr_data)

    print("Done.")


def _main(argv: tuple | None = None) -> None:
    """Perform surrogate map generation."""
    args = _get_parser().parse_args(argv)
    make_surrogate(**vars(args))


if __name__ == "__main__":
    _main()
