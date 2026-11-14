#!/usr/bin/env python3
"""Run bounded, source-frozen activation-partition experiments in timing SimX."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import TOOLS, result_path, run_log_dir

from prepare_sources import APP, HERE, prepare
sys.path.insert(0, str(HERE.parent / 'warp_capacity'))
from config_probe import SOURCE, config
from run_sweep import COMMIT, parse_case



def selection(tiles=5, k=64, mode=4, aw=1, unroll=1, producer=0, depth=1, seed=1):
    return dict(tiles=tiles, k=k, mode=mode, activation_warps=aw if mode == 4 else 0,
                unroll=unroll, producer_relu=producer, c_depth=depth, seed=seed)


def selections(phase):
    variants = [(2, 0, 1), (3, 0, 1)] + [(4, a, u) for a in (1, 2) for u in (1, 4)]
    if phase in ('matrix', 'smoke'):
        shapes = [(t, k) for t in (1, 2, 5) for k in (16, 64)] if phase == 'matrix' else [(1, 16)]
        return [selection(t, k, m, a, u) for t, k in shapes for m, a, u in variants]
    if phase == 'controls':
        return ([selection(aw=a, unroll=u, producer=1) for a in (1, 2) for u in (1, 4)]
            + [selection(mode=m, aw=a, unroll=u, seed=2026) for m, a, u in variants]
            + [selection(k=k, aw=a, unroll=4, depth=2) for k in (16, 64) for a in (1, 2)])
    if phase in ('capacity', 'repeat'):
        return [selection(mode=m, aw=a, unroll=u) for _ in range(2 if phase == 'repeat' else 1) for m, a, u in variants]
    return [selection(aw=a, unroll=u) for a in (1, 2) for u in (1, 4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['smoke', 'matrix', 'controls', 'capacity', 'repeat', 'trace'])
    parser.add_argument('--warps', type=int, choices=[8, 16], default=16)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() != COMMIT:
        raise RuntimeError('Pinned source required')
    if subprocess.check_output(['git', 'diff', '--name-only'], cwd=SOURCE, text=True):
        raise RuntimeError('Unexpected tracked changes')
    flags, resolved = config(args.warps)
    expected = json.loads(result_path('persistent_w16_matrix_results.json').read_text())['effective_config']
    if resolved != dict(expected, VX_CFG_NUM_WARPS=str(args.warps)):
        raise RuntimeError('Unexpected effective configuration')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = run_log_dir(f'activation_w{args.warps}_{args.phase}_{stamp}')
    logs.mkdir(parents=True, exist_ok=False)
    hashes = prepare()
    target = SOURCE / 'tests/regression/activation_balance'
    if target.exists() and not (target / '.project20-owned').exists():
        raise RuntimeError('Refusing to overwrite an existing application')
    target.mkdir(exist_ok=True)
    for name in hashes:
        shutil.copyfile(APP / name, target / name)
        shutil.copyfile(APP / name, logs / name)
    (target / '.project20-owned').write_text(COMMIT + '\n')
    build = SOURCE / f'build32-warp-capacity-{args.warps}'
    build.mkdir(exist_ok=True)
    result = dict(status='running', phase=args.phase, warps=args.warps, source_commit=COMMIT,
        source_hashes=hashes, effective_config=resolved, configs=flags, build=str(build), logs=str(logs),
        ordinary_prefetch='cross-output-tile', protocol='ready/free; input participants=5; output participants=4+activation_warps',
        steps=[], cases=[])
    env = os.environ.copy()
    for name in ('CONFIGS', 'DEBUG', 'PERF', 'SCOPE', 'SIMX_FUNCTIONAL', 'VORTEX_DRIVER', 'VORTEX_PROFILING',
                 'VORTEX_HOME', 'VORTEX_PATH', 'VORTEX_BUILD_DIR', 'VORTEX_RT_PATH', 'VORTEX_KN_PATH',
                 'TOOLDIR', 'TOOLCHAIN_REV', 'DESTDIR', 'MAKEFLAGS', 'MFLAGS'):
        env.pop(name, None)
    env.update(CONFIGS=flags, CCACHE_DISABLE='1', MAKEFLAGS='-j4')

    def save():
        content = json.dumps(result, indent=2) + '\n'
        (logs / 'results.json').write_text(content)
        result_path(f'activation_w{args.warps}_{args.phase}_results.json').write_text(content)

    def run(name, command, timeout=300):
        print(name + ': ' + shlex.join(command), flush=True)
        path = logs / (name + '.log')
        with path.open('w') as out:
            out.write('command=' + shlex.join(command) + '\ncwd=' + str(build) + '\nCONFIGS=' + flags + '\n')
            out.flush()
            proc = subprocess.Popen(command, cwd=build, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        result['steps'].append(dict(name=name, log=str(path), command=command, exit_code=code))
        save()
        text = path.read_text(errors='replace')
        if code:
            raise RuntimeError(name + ': exit=' + str(code) + '\n' + text[-3000:])
        return text, path

    try:
        save()
        run('configure', ['../configure', '--xlen=32', f'--tooldir={TOOLS}'])
        run('software', ['make', '-j4', 'software'], timeout=600)
        run('build_app', ['make', '-j4', '-C', 'tests/regression/activation_balance'])
        for name in ('kernel.vxbin', 'kernel.dump'):
            shutil.copyfile(build / 'tests/regression/activation_balance' / name, logs / name)
        result['binary_sha256'] = hashlib.sha256((logs / 'kernel.vxbin').read_bytes()).hexdigest()
        for index, case in enumerate(selections(args.phase)):
            t, k, mode, aw, u, pr, depth, seed = (case[n] for n in ('tiles', 'k', 'mode', 'activation_warps', 'unroll', 'producer_relu', 'c_depth', 'seed'))
            name = f't{t}_k{k}_v{mode}_a{aw}_u{u}_p{pr}_q{depth}_s{seed}_r{index}'
            command = ['./ci/blackbox.sh', '--driver=simx', '--app=activation_balance',
                f'--args=-m64 -n{32*t} -k{k} -v{mode} -w{aw or 1} -u{u} -e{pr} -q{depth} -g1 -s{seed}', '--perf=7']
            raw = None
            if args.phase == 'trace':
                raw = build / f'activation_a{aw}_u{u}.trace.log'
                command += ['--debug=3', '--log=' + str(raw)]
            text, path = run(name, command)
            if raw:
                with raw.open('rb') as src, gzip.open(logs / (name + '.trace.log.gz'), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                text = raw.read_text(errors='replace')
                case['trace'] = str(logs / (name + '.trace.log.gz'))
            case.update(name=name, log=str(path), driver='simx', **parse_case(text))
            meta = re.search(r'PERSISTENT output_tiles=(\d+) ctas=(\d+) c_depth=(\d+) smem=(\d+)', text)
            if meta:
                case['smem'] = int(meta[4])
            result['cases'].append(case)
            save()
            print(json.dumps({n: case.get(n) for n in ('name', 'passed', 'chain')}), flush=True)
            wanted_warps = 5 + aw if mode == 4 else mode + 2
            if (not case['passed'] or case['cta_warps'] != wanted_warps
                    or case['hardware_warps'] != args.warps or not meta or int(meta[2]) != 1):
                raise RuntimeError('Configuration or correctness failure: ' + name)
        result['status'] = 'complete'
        save()
    except Exception as error:
        result.update(status='failed', error=str(error))
        save()
        raise
    print(json.dumps(dict(status='complete', cases=len(result['cases']), logs=str(logs))), flush=True)


if __name__ == '__main__':
    main()
