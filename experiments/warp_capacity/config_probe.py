#!/usr/bin/env python3
"""Resolve and compare the hardware knobs for the bounded warp-capacity sweep."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pc_pipeline'))
from run import FLAGS, SOURCE


def resolve(flags):
    output = subprocess.check_output(['python3', str(SOURCE / 'ci/gen_config.py'),
        '--config', str(SOURCE / 'VX_config.toml'), '--cflags', flags + ' -DVX_CFG_XLEN=32'], text=True)
    return {name: value for token in shlex.split(output)
            for name, sep, value in [token.removeprefix('-D').partition('=')] if sep}


def config(warps, controlled=True):
    flags = FLAGS.replace('-DVX_CFG_NUM_WARPS=8', f'-DVX_CFG_NUM_WARPS={warps}')
    if controlled:
        flags += ' -DVX_CFG_LMEM_LOG_SIZE=16 -DVX_CFG_LSU_PENDING_SIZE=16 -DVX_CFG_AMO_RS_SIZE=8'
    return flags, resolve(flags)


def differences(first, second):
    return {key: [first.get(key), second.get(key)] for key in sorted(first.keys() | second.keys())
            if first.get(key) != second.get(key)}


def main():
    natural = {w: config(w, False)[1] for w in (8, 16)}
    controlled = {w: config(w) for w in (8, 16)}
    result = dict(natural_differences=differences(natural[8], natural[16]),
                  controlled_differences=differences(controlled[8][1], controlled[16][1]),
                  configurations={w: dict(flags=x[0], resolved=x[1]) for w, x in controlled.items()},
                  baseline_changes=differences(natural[8], controlled[8][1]))
    (HERE / 'configurations.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'configurations'}, indent=2))


if __name__ == '__main__':
    main()
