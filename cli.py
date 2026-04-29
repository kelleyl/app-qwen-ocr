#!/usr/bin/env python3
"""
DO NOT CHANGE the name of this file.
"""

import argparse
import sys
from contextlib import redirect_stdout

import app
import clams.app
from clams import AppMetadata


def metadata_to_argparser(app_metadata: AppMetadata) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{app_metadata.name}: {app_metadata.description} (visit {app_metadata.url} for more info)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    for parameter in app_metadata.parameters:
        if parameter.multivalued:
            a = parser.add_argument(
                f"--{parameter.name}",
                help=parameter.description,
                nargs='+', action='extend', type=str,
            )
        else:
            a = parser.add_argument(
                f"--{parameter.name}",
                help=parameter.description,
                nargs=1, action='store', type=str,
            )
        if parameter.choices is not None:
            a.choices = parameter.choices
        if parameter.default is not None:
            a.help += f" (default: {parameter.default}"
            if parameter.type == "boolean":
                a.help += (f", any value except for {[v for v in clams.app.falsy_values if isinstance(v, str)]} "
                           f"will be interpreted as True")
            a.help += ')'

    parser.add_argument(
        'IN_MMIF_FILE', nargs='?', type=argparse.FileType('r'),
        help='Input MMIF file path or `-` for STDIN.',
        default=None if sys.stdin.isatty() else sys.stdin,
    )
    parser.add_argument(
        'OUT_MMIF_FILE', nargs='?', type=argparse.FileType('w'),
        help='Output MMIF file path or `-` for STDOUT.',
        default=sys.stdout,
    )
    return parser


if __name__ == "__main__":
    clamsapp = app.get_app()
    arg_parser = metadata_to_argparser(app_metadata=clamsapp.metadata)
    args = arg_parser.parse_args()
    if args.IN_MMIF_FILE:
        in_data = args.IN_MMIF_FILE.read()
        params = {}
        for pname, pvalue in vars(args).items():
            if pvalue is None or pname in ['IN_MMIF_FILE', 'OUT_MMIF_FILE']:
                continue
            elif isinstance(pvalue, list):
                params[pname] = pvalue
            else:
                params[pname] = [pvalue]
        if args.OUT_MMIF_FILE.name == '<stdout>':
            with redirect_stdout(sys.stderr):
                out_mmif = clamsapp.annotate(in_data, **params)
        else:
            out_mmif = clamsapp.annotate(in_data, **params)
        args.OUT_MMIF_FILE.write(out_mmif)
    else:
        arg_parser.print_help()
        sys.exit(1)
