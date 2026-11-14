#!/usr/bin/env python3
"""Explore persistent CTA counts using the unchanged activation-balance binary."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import TOOLS

HERE = Path(__file__).resolve().parent
COURSE = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'warp_capacity'))
from config_probe import SOURCE, config
from run_sweep import COMMIT, parse_case

VARIANTS = [(2, 0, 1), (3, 0, 1), (4, 1, 4), (4, 2, 4)]


def cases(phase):
    if phase == 'smoke':
        shapes = [(128, 64, 16, 2)]
    elif phase == 'matrix':
        shapes = [(256, 128, k, g) for k in (64, 256) for g in (1, 2, 4, 8)]
    elif phase == 'validation':
        shapes = [(192, 96, 16, 4), (192, 160, 64, 8), (256, 128, 64, 8)]
    elif phase == 'repeat':
        shapes = [(256, 128, 64, g) for _ in range(2) for g in (4, 8)]
    elif phase == 'besttrace':
        shapes = [(256, 128, 64, 2)]
    else:
        shapes = [(256, 128, 64, 8)]
    variants = VARIANTS if phase != 'trace' else [VARIANTS[i] for i in (0, 1, 3)]
    if phase == 'besttrace':
        variants = VARIANTS[2:]
    return [dict(m=m, n=n, k=k, ctas=g, mode=v, activation_warps=a, unroll=u,
                 producer_relu=0, c_depth=1, seed=2026 if phase == 'validation' else 1)
            for m, n, k, g in shapes for v, a, u in variants]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=['smoke', 'matrix', 'validation', 'repeat', 'trace', 'finish', 'best', 'besttrace'])
    p.add_argument('--warps', type=int, choices=[8, 16], required=True)
    args = p.parse_args()
    if args.phase == 'finish':
        for phase in ('validation', 'repeat', 'trace'):
            subprocess.run([sys.executable, __file__, '--phase', phase, '--warps', str(args.warps)], check=True)
        return
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() != COMMIT:
        raise RuntimeError('Pinned source required')
    if subprocess.check_output(['git', 'diff', '--name-only'], cwd=SOURCE, text=True):
        raise RuntimeError('Unexpected tracked changes')
    reference = json.loads((COURSE / 'activation_w16_matrix_results.json').read_text())
    app = SOURCE / 'tests/regression/activation_balance'
    hashes = {name: hashlib.sha256((app / name).read_bytes()).hexdigest() for name in reference['source_hashes']}
    mirror = HERE.parent / 'activation_balance/activation_balance'
    if hashes != reference['source_hashes'] or any(hashlib.sha256((mirror / n).read_bytes()).hexdigest() != h for n, h in hashes.items()):
        raise RuntimeError('Validated application changed')
    flags, resolved = config(args.warps)
    if resolved != dict(reference['effective_config'], VX_CFG_NUM_WARPS=str(args.warps)):
        raise RuntimeError('Unexpected hardware configuration')
    build = SOURCE / f'build32-warp-capacity-{args.warps}'
    build.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = COURSE / 'logs' / f'multicta_w{args.warps}_{args.phase}_{stamp}'
    logs.mkdir(parents=True, exist_ok=False)
    for name in hashes:
        shutil.copyfile(app / name, logs / name)
    result = dict(status='running', phase=args.phase, hardware_warps=args.warps,
        source_commit=COMMIT, source_hashes=hashes, configs=flags, effective_config=resolved,
        ordinary_prefetch='cross-output-tile', build=str(build), logs=str(logs), steps=[], cases=[])
    env = os.environ.copy()
    for name in ('CONFIGS', 'DEBUG', 'PERF', 'SCOPE', 'SIMX_FUNCTIONAL', 'VORTEX_DRIVER', 'VORTEX_PROFILING',
                 'VORTEX_HOME', 'VORTEX_PATH', 'VORTEX_BUILD_DIR', 'VORTEX_RT_PATH', 'VORTEX_KN_PATH',
                 'TOOLDIR', 'TOOLCHAIN_REV', 'DESTDIR', 'MAKEFLAGS', 'MFLAGS'):
        env.pop(name, None)
    env.update(CONFIGS=flags, CCACHE_DISABLE='1', MAKEFLAGS='-j4')

    def save():
        text = json.dumps(result, indent=2) + '\n'
        (logs / 'results.json').write_text(text)
        (COURSE / f'multicta_w{args.warps}_{args.phase}_results.json').write_text(text)

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
        result['steps'].append(dict(name=name, command=command, log=str(path), exit_code=code))
        save()
        text = path.read_text(errors='replace')
        if code:
            raise RuntimeError(name + ': exit=' + str(code) + '\n' + text[-2600:])
        return text, path

    try:
        save()
        run('configure', ['../configure', '--xlen=32', f'--tooldir={TOOLS}'])
        run('software', ['make', '-j4', 'software'], timeout=600)
        run('build_app', ['make', '-j4', '-C', 'tests/regression/activation_balance'])
        for name in ('kernel.vxbin', 'kernel.dump'):
            shutil.copyfile(build / 'tests/regression/activation_balance' / name, logs / name)
        result['binary_sha256'] = hashlib.sha256((logs / 'kernel.vxbin').read_bytes()).hexdigest()
        if result['binary_sha256'] != reference['binary_sha256']:
            raise RuntimeError('Device binary no longer reproduces the previous experiment')
        selected = cases(args.phase)
        if args.phase == 'best':
            matrix = json.loads((COURSE / f'multicta_w{args.warps}_matrix_results.json').read_text())
            if matrix['status'] != 'complete':
                raise RuntimeError('The full matrix must precede best-CTA validation')
            selected = []
            for k in (64, 256):
                for v, a, u in VARIANTS:
                    best = min((c for c in matrix['cases'] if (c['k'], c['mode'], c['activation_warps']) == (k, v, a)),
                               key=lambda c: c['chain']['cycles'])
                    selected.append(dict(m=256, n=128, k=k, ctas=best['ctas'], mode=v, activation_warps=a,
                        unroll=u, producer_relu=0, c_depth=1, seed=2026, selected_from=matrix['logs']))
        for index, case in enumerate(selected):
            m, n, k, g, mode, a, u, seed = (case[key] for key in ('m', 'n', 'k', 'ctas', 'mode', 'activation_warps', 'unroll', 'seed'))
            tiles = (m//64)*(n//32)
            counts = [1+(tiles-1-i)//g for i in range(g)]
            name = f'm{m}_n{n}_k{k}_g{g}_v{mode}_a{a}_u{u}_s{seed}_r{index}'
            command = ['./ci/blackbox.sh', '--driver=simx', '--app=activation_balance',
                f'--args=-m{m} -n{n} -k{k} -g{g} -v{mode} -w{a or 1} -u{u} -e0 -q1 -s{seed}', '--perf=7']
            raw = None
            if args.phase in ('trace', 'besttrace'):
                raw = build / f'multicta_v{mode}_a{a}.trace.log'
                command += ['--debug=3', '--log=' + str(raw)]
            text, path = run(name, command)
            if raw:
                with raw.open('rb') as src, gzip.open(logs / (name + '.trace.log.gz'), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                text = raw.read_text(errors='replace')
                case['trace'] = str(logs / (name + '.trace.log.gz'))
            case.update(name=name, log=str(path), output_tiles=tiles, tiles_per_cta=counts, driver='simx', **parse_case(text))
            meta = re.search(r'PERSISTENT output_tiles=(\d+) ctas=(\d+) c_depth=(\d+) smem=(\d+)', text)
            if meta:
                case['smem'] = int(meta[4])
            result['cases'].append(case)
            save()
            print(json.dumps({key: case.get(key) for key in ('name', 'passed', 'chain')}), flush=True)
            if (not case['passed'] or min(case['coverage'].values()) <= 0
                    or case['cta_warps'] != (5+a if mode == 4 else mode+2)
                    or case['hardware_warps'] != args.warps or not meta
                    or list(map(int, meta.groups()[:3])) != [tiles, g, 1]):
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
