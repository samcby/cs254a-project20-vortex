#!/usr/bin/env python3
"""Bounded persistent fused/two-stage/three-stage experiments on Vortex v3.0."""

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


def prepare(logs):
    old_app = SOURCE / 'tests/regression/gemm_relu'
    reference = json.loads((HERE.parent / 'gemm_relu/analysis.json').read_text())['source_hashes']
    if any(hashlib.sha256((old_app / name).read_bytes()).hexdigest() != value for name, value in reference.items()):
        raise RuntimeError('Validated source application changed')
    app = SOURCE / 'tests/regression/persistent_pipeline'
    if app.exists() and not (app / '.project20-owned').exists():
        raise RuntimeError('Refusing to replace an existing app')
    app.mkdir(exist_ok=True)
    mirror = HERE / 'persistent_pipeline'
    host = (old_app / 'main.cpp').read_text()

    def replace(before, after):
        nonlocal host
        if host.count(before) != 1:
            raise RuntimeError('Host source mismatch: ' + before)
        host = host.replace(before, after)

    replace('arg.M = 64; arg.N = 32; arg.K = 64; arg.depth = 2;',
            'arg.M = 64; arg.N = 32; arg.K = 64; arg.depth = 2; arg.c_depth = 1;')
    replace('uint32_t mode = 2, seed = 1, activation_stride = 0;',
            'uint32_t mode = 2, seed = 1, activation_stride = 0, persistent_ctas = 1;')
    replace('"m:n:k:v:s:a:"', '"m:n:k:v:s:a:q:g:"')
    replace("case 'a':", "case 'q': arg.c_depth = std::strtoul(optarg, nullptr, 0); break;\n"
            "    case 'g': persistent_ctas = std::strtoul(optarg, nullptr, 0); break;\n    case 'a':")
    replace('mode > 3', '(mode < 2 || mode > 4 || !persistent_ctas || (arg.c_depth != 1 && arg.c_depth != 2))')
    replace('const uint32_t active_warps = width + (mode == 3 ? 1 : 0);',
            'const uint32_t active_warps = width + mode - 2;')
    replace('  vx_queue_info_t qi =',
            '  if (persistent_ctas > (arg.M / cta_m) * (arg.N / cfg::xtileN)) { cleanup(); return 1; }\n'
            '  vx_queue_info_t qi =')
    replace('mode == 3 ? "gemm_pc" : "gemm_base"',
            'mode == 4 ? "gemm_three" : (mode == 3 ? "gemm_pc" : "gemm_base")')
    replace('li.ndim = 2;', 'li.ndim = 1;')
    replace('li.grid_dim[0] = arg.N / cfg::xtileN; li.grid_dim[1] = arg.M / cta_m;',
            'li.grid_dim[0] = persistent_ctas; li.grid_dim[1] = 1;')
    replace('li.lmem_size = 2 * (cta_m * cfg::tileK + cfg::tileK * cfg::xtileN) * 2;',
            'li.lmem_size = 2 * (cta_m * cfg::tileK + cfg::tileK * cfg::xtileN) * 2\n'
            '      + (mode == 4 ? arg.c_depth * cta_m * cfg::xtileN * 4 : 0);\n'
            '  std::cout << "PERSISTENT output_tiles=" << (arg.M / cta_m) * (arg.N / cfg::xtileN)\n'
            '            << " ctas=" << persistent_ctas << " c_depth=" << arg.c_depth\n'
            '            << " smem=" << li.lmem_size << std::endl;')
    (mirror / 'main.cpp').write_text(host)
    (mirror / 'common.h').write_text((old_app / 'common.h').read_text().replace(
        'uint32_t fuse;', 'uint32_t fuse;\n  uint32_t c_depth;'))
    (mirror / 'Makefile').write_text((old_app / 'Makefile').read_text().replace(
        'PROJECT := gemm_relu', 'PROJECT := persistent_pipeline').replace(
        '$(SRC_DIR)/base.cpp $(SRC_DIR)/pc.cpp $(SRC_DIR)/relu.cpp',
        '$(SRC_DIR)/kernel.cpp $(SRC_DIR)/relu.cpp'))
    shutil.copyfile(old_app / 'relu.cpp', mirror / 'relu.cpp')
    for path in mirror.iterdir():
        if path.is_file():
            shutil.copyfile(path, app / path.name)
            shutil.copyfile(path, logs / path.name)
    (app / '.project20-owned').write_text(COMMIT + '\n')
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in mirror.iterdir() if p.is_file()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['matrix', 'depth', 'trace', 'repeat', 'rtl'], required=True)
    p.add_argument('--warps', type=int, choices=[8, 16], default=16)
    p.add_argument('--rtl-build-timeout', type=int, default=1800,
                   help='Seconds for the first RTL model build, separate from each 600-second smoke run')
    args = p.parse_args()
    if args.rtl_build_timeout <= 0:
        p.error('--rtl-build-timeout must be positive')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() != COMMIT:
        raise RuntimeError('Pinned source required')
    if subprocess.check_output(['git', 'diff', '--name-only'], cwd=SOURCE, text=True):
        raise RuntimeError('Unexpected tracked edits')
    flags, resolved = config(args.warps)
    build = SOURCE / f'build32-warp-capacity-{args.warps}'
    build.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = COURSE / 'logs' / f'persistent_w{args.warps}_{args.phase}_{stamp}'
    logs.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', phase=args.phase, warps=args.warps, source_commit=COMMIT,
                  configs=flags, effective_config=resolved, ordinary_prefetch='cross-output-tile',
                  build=str(build), logs=str(logs), steps=[], cases=[])
    env = os.environ.copy()
    for name in ('CONFIGS', 'DEBUG', 'PERF', 'SCOPE', 'SIMX_FUNCTIONAL', 'VORTEX_DRIVER', 'VORTEX_PROFILING',
                 'VORTEX_HOME', 'VORTEX_PATH', 'VORTEX_BUILD_DIR', 'VORTEX_RT_PATH',
                 'VORTEX_KN_PATH', 'TOOLDIR', 'TOOLCHAIN_REV', 'DESTDIR', 'MAKEFLAGS', 'MFLAGS'):
        env.pop(name, None)
    env.update(CONFIGS=flags, CCACHE_DISABLE='1', MAKEFLAGS='-j4')
    if args.phase == 'rtl':
        env['THREADS'] = '4'

    def save():
        text = json.dumps(result, indent=2) + '\n'
        (logs / 'results.json').write_text(text)
        (COURSE / f'persistent_w{args.warps}_{args.phase}_results.json').write_text(text)

    def run(name, command, timeout=300):
        print(name + ': ' + shlex.join(command), flush=True)
        path = logs / (name + '.log')
        with path.open('w') as out:
            out.write('command=' + shlex.join(command) + '\ncwd=' + str(build) + '\nCONFIGS=' + flags + '\n')
            out.flush()
            proc = subprocess.Popen(command, cwd=build, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        result['steps'].append(dict(name=name, command=command, log=str(path), exit_code=code))
        save()
        content = path.read_text(errors='replace')
        if code:
            raise RuntimeError(name + ': exit=' + str(code) + '\n' + content[-2400:])
        return content, path

    try:
        result['source_hashes'] = prepare(logs)
        run('configure', ['../configure', '--xlen=32', f'--tooldir={TOOLS}'])
        run('software', ['make', '-j4', 'software'], timeout=600)
        run('build_app', ['make', '-j4', '-C', 'tests/regression/persistent_pipeline'])
        if args.phase == 'rtl':
            run('build_rtlsim', ['make', '-j4', '-C', 'tests/regression/persistent_pipeline', 'runtime-rtlsim'],
                timeout=args.rtl_build_timeout)
        if args.phase == 'matrix':
            selections = [(t, k, mode, 1, 1, 1) for t in (1, 2, 5) for k in (16, 64) for mode in (2, 3, 4)]
        elif args.phase == 'depth':
            selections = [(5, k, 4, 2, 1, seed) for k, seed in ((16, 1), (64, 1), (64, 2026))]
            selections += [(5, 64, mode, 1, 1, 2026) for mode in (2, 3, 4)]
        elif args.phase == 'trace':
            selections = [(5, 64, mode, 1, 1, 1) for mode in (3, 4)]
        elif args.phase == 'repeat':
            selections = [(5, 64, mode, depth, 1, 1) for _ in range(3)
                          for mode, depth in ((2, 1), (3, 1), (4, 1), (4, 2))]
        else:
            selections = [(1, 16, mode, 1, 1, 1) for mode in (2, 4)] + [(2, 16, 4, 1, 1, 1)]
        for index, (tiles, k, mode, cdepth, ctas, seed) in enumerate(selections):
            name = f't{tiles}_k{k}_v{mode}_q{cdepth}_g{ctas}_s{seed}'
            if args.phase == 'repeat':
                name += f'_r{index}'
            driver = 'rtlsim' if args.phase == 'rtl' else 'simx'
            command = ['./ci/blackbox.sh', '--driver=' + driver, '--app=persistent_pipeline',
                f'--args=-m64 -n{32 * tiles} -k{k} -v{mode} -q{cdepth} -g{ctas} -s{seed}', '--perf=7']
            raw = None
            if args.phase == 'trace':
                raw = build / f'persistent_v{mode}.trace.log'
                command += ['--debug=3', '--log=' + str(raw)]
            text, path = run(name, command, timeout=600 if driver == 'rtlsim' else 300)
            if raw:
                with raw.open('rb') as src, gzip.open(logs / (name + '.trace.log.gz'), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                text = raw.read_text(errors='replace')
            case = dict(name=name, tiles=tiles, k=k, mode=mode, c_depth=cdepth, ctas=ctas,
                        seed=seed, driver=driver, log=str(path), **parse_case(text))
            meta = re.search(r'PERSISTENT output_tiles=(\d+) ctas=(\d+) c_depth=(\d+) smem=(\d+)', text)
            if meta:
                case['smem'] = int(meta[4])
            if raw:
                case['trace'] = str(logs / (name + '.trace.log.gz'))
            result['cases'].append(case)
            save()
            print(json.dumps({x: case.get(x) for x in ('name', 'passed', 'chain', 'smem')}), flush=True)
            if not case['passed'] or case['cta_warps'] != mode + 2 or case['hardware_warps'] != args.warps:
                raise RuntimeError('Correctness or configuration failure: ' + name)
        result['status'] = 'complete'
        save()
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        save()
        raise
    print(json.dumps(dict(status='complete', logs=str(logs))), flush=True)


if __name__ == '__main__':
    main()
