#!/usr/bin/env python3
"""Controlled 8/16-warp comparisons with unchanged GEMM/ReLU application sources."""

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

from config_probe import HERE, SOURCE, config, differences

COURSE = HERE.parents[1]
COMMIT = '7fe19369587c087f902c3e2bc2add612aa910a95'


def parse_case(text):
    chain = re.search(r'CHAIN cycles=(\d+) instructions=(\d+) reads=(\d+) writes=(\d+) launches=(\d+)', text)
    coverage = re.search(r'COVERAGE positive=(\d+) negative=(\d+) zero=(\d+)', text)
    verify = re.search(r'VERIFY errors=(\d+) output_hash=(\d+)', text)
    hw = re.search(r'hardware_warps=(\d+) compute_warps=(\d+) active_CTA_warps=(\d+)', text)
    passed = bool(chain and coverage and verify and hw and verify[1] == '0'
                  and re.search(r'^PASSED!$', text, re.MULTILINE))
    result = dict(passed=passed)
    if passed:
        result.update(chain=dict(zip(('cycles', 'instructions', 'reads', 'writes', 'launches'), map(int, chain.groups()))),
                      coverage=dict(zip(('positive', 'negative', 'zero'), map(int, coverage.groups()))),
                      hardware_warps=int(hw[1]), compute_warps=int(hw[2]), cta_warps=int(hw[3]),
                      output_hash=verify[2])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['matrix', 'trace'], required=True)
    parser.add_argument('--warps', type=int, choices=[8, 16], required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() != COMMIT:
        raise RuntimeError('Pinned commit required')
    if subprocess.check_output(['git', 'diff', '--name-only'], cwd=SOURCE, text=True):
        raise RuntimeError('Unexpected tracked changes')
    expected = json.loads((HERE.parent / 'gemm_relu/analysis.json').read_text())['source_hashes']
    app = SOURCE / 'tests/regression/gemm_relu'
    hashes = {name: hashlib.sha256((app / name).read_bytes()).hexdigest() for name in expected}
    if hashes != expected:
        raise RuntimeError('GEMM/ReLU application changed since the validated comparison')
    flags, resolved = config(args.warps)
    _, baseline = config(8)
    diff = differences(baseline, resolved)
    if set(diff) - {'VX_CFG_NUM_WARPS'}:
        raise RuntimeError('Uncontrolled configuration difference: ' + str(diff))
    build = SOURCE / f'build32-warp-capacity-{args.warps}'
    build.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = COURSE / 'logs' / f'warps{args.warps}_{args.phase}_{stamp}'
    logs.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', phase=args.phase, warps=args.warps, source_commit=COMMIT,
                  source_hashes=hashes, build=str(build), logs=str(logs), configs=flags,
                  effective_config=resolved, differences_from_8=diff, steps=[], cases=[])
    env = os.environ.copy()
    for name in ('CONFIGS', 'DEBUG', 'PERF', 'SCOPE', 'SIMX_FUNCTIONAL', 'VORTEX_DRIVER', 'VORTEX_PROFILING',
                 'VORTEX_HOME', 'VORTEX_PATH', 'VORTEX_BUILD_DIR', 'VORTEX_RT_PATH',
                 'VORTEX_KN_PATH', 'TOOLDIR', 'TOOLCHAIN_REV', 'DESTDIR', 'MAKEFLAGS', 'MFLAGS'):
        env.pop(name, None)
    env.update(CONFIGS=flags, CCACHE_DISABLE='1', MAKEFLAGS='-j4')

    def save():
        text = json.dumps(result, indent=2) + '\n'
        (logs / 'results.json').write_text(text)
        (COURSE / f'warp{args.warps}_{args.phase}_results.json').write_text(text)

    def run(name, command, timeout=300):
        print(name + ': ' + shlex.join(command), flush=True)
        path = logs / (name + '.log')
        with path.open('w') as out:
            out.write('command=' + shlex.join(command) + '\ncwd=' + str(build) + '\nCONFIGS=' + flags + '\n')
            out.flush()
            p = subprocess.Popen(command, cwd=build, env=env, stdout=out,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                code = 124
        result['steps'].append(dict(name=name, command=command, log=str(path), exit_code=code))
        save()
        if code:
            raise RuntimeError(name + ': exit=' + str(code) + '\n' + path.read_text(errors='replace')[-2500:])
        return path.read_text(errors='replace'), path

    try:
        for name in hashes:
            shutil.copyfile(app / name, logs / name)
        save()
        run('configure', ['../configure', '--xlen=32', f'--tooldir={TOOLS}'])
        run('software', ['make', '-j4', 'software'], timeout=600)
        run('build_app', ['make', '-j4', '-C', 'tests/regression/gemm_relu'])
        for name in ('kernel.vxbin', 'kernel.dump'):
            shutil.copyfile(build / 'tests/regression/gemm_relu' / name, logs / name)
        result['binary_sha256'] = hashlib.sha256((logs / 'kernel.vxbin').read_bytes()).hexdigest()
        shapes = ([(64, 32, k, 1) for k in (64, 256)] +
                  [(m, 128, k, 1) for m in (128, 256) for k in (64, 256)] +
                  [(64, 32, 256, 2026), (128, 128, 64, 2026)])
        if args.phase == 'trace':
            shapes = [(128, 128, 64, 1)]
        for m, n, k, seed in shapes:
            for mode in (2, 3):
                name = f'm{m}_n{n}_k{k}_s{seed}_v{mode}'
                command = ['./ci/blackbox.sh', '--driver=simx', '--app=gemm_relu',
                           f'--args=-m{m} -n{n} -k{k} -s{seed} -v{mode}', '--perf=7']
                raw = None
                if args.phase == 'trace':
                    raw = build / f'warp_capacity_v{mode}.trace.log'
                    command += ['--debug=3', '--log=' + str(raw)]
                text, path = run(name, command)
                if raw is not None:
                    with gzip.open(logs / (name + '.trace.log.gz'), 'wb') as dst, raw.open('rb') as src:
                        shutil.copyfileobj(src, dst)
                    text = raw.read_text(errors='replace')
                case = dict(name=name, m=m, n=n, k=k, seed=seed, mode=mode, warps=args.warps,
                            log=str(path), **parse_case(text))
                if raw is not None:
                    case['trace'] = str(logs / (name + '.trace.log.gz'))
                result['cases'].append(case)
                save()
                print(json.dumps({key: case.get(key) for key in ('name', 'passed', 'chain')}), flush=True)
                if not case['passed'] or case['hardware_warps'] != args.warps or min(case['coverage'].values()) <= 0:
                    raise RuntimeError('Correctness/configuration failure: ' + name)
        result['status'] = 'complete'
        save()
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        save()
        raise
    print(json.dumps(dict(status='complete', logs=str(logs))), flush=True)


if __name__ == '__main__':
    main()
