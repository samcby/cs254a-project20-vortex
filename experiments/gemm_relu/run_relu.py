#!/usr/bin/env python3
"""Run exact signed-input GEMM/ReLU controls on the existing timing SimX build."""

import argparse
from datetime import datetime, timezone
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pc_pipeline'))
from run import BUILD, COMMIT, FLAGS, SOURCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'matrix', 'activation', 'validation'], required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() != COMMIT:
        raise RuntimeError('Pinned Vortex commit required')
    if subprocess.check_output(['git', 'diff', '--name-only'], cwd=SOURCE, text=True):
        raise RuntimeError('Unexpected tracked source edits')
    prior = json.loads(result_path('pc_indexing_results.json').read_text())
    pc_source = SOURCE / 'tests/regression/sgemm_tcu_wg_pc_mask/kernel.cpp'
    if hashlib.sha256(pc_source.read_bytes()).hexdigest() != prior['source_hashes']['kernel.cpp']:
        raise RuntimeError('Prior tested PC source changed')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = run_log_dir('relu_' + args.phase + '_' + stamp)
    logs.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', phase=args.phase, source_commit=COMMIT, configs=FLAGS,
                  effective_config=prior['effective_config'], logs=str(logs),
                  metric='sum of per-kernel cycle deltas; cumulative core counters and per-launch uncore counters',
                  counter_semantics='core-delta_uncore-per-launch_v2',
                  steps=[], cases=[])
    env = os.environ.copy()
    for name in ('CONFIGS', 'DEBUG', 'PERF', 'SCOPE', 'SIMX_FUNCTIONAL', 'VORTEX_DRIVER', 'VORTEX_PROFILING',
                 'VORTEX_HOME', 'VORTEX_PATH', 'VORTEX_BUILD_DIR', 'VORTEX_RT_PATH',
                 'VORTEX_KN_PATH', 'TOOLDIR', 'TOOLCHAIN_REV', 'DESTDIR', 'MAKEFLAGS', 'MFLAGS'):
        env.pop(name, None)
    env.update(CONFIGS=FLAGS, CCACHE_DISABLE='1', MAKEFLAGS='-j4')

    def save():
        text = json.dumps(result, indent=2) + '\n'
        (logs / 'results.json').write_text(text)
        result_path('relu_' + args.phase + '_results.json').write_text(text)

    def run(name, command, timeout=300):
        print(name + ': ' + shlex.join(command), flush=True)
        path = logs / (name + '.log')
        with path.open('w') as out:
            out.write('command=' + shlex.join(command) + '\ncwd=' + str(BUILD) + '\nCONFIGS=' + FLAGS + '\n')
            out.flush()
            p = subprocess.Popen(command, cwd=BUILD, env=env, stdout=out,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                code = 124
        result['steps'].append(dict(name=name, command=command, log=str(path), exit_code=code))
        save()
        content = path.read_text(errors='replace')
        if code:
            raise RuntimeError(name + ': ' + content[-2500:])
        return content, path

    try:
        dest = SOURCE / 'tests/regression/gemm_relu'
        if dest.exists() and not (dest / '.project20-owned').exists():
            raise RuntimeError('Refusing to replace an existing app')
        dest.mkdir(exist_ok=True)
        base = (SOURCE / 'tests/regression/sgemm_tcu_wg_dxa/kernel.cpp').read_text()
        base = base.replace('__kernel void kernel_main(', '__kernel void gemm_base(')
        base = base.replace('  // Store the computed C tile to global memory.',
            '  if (arg->fuse) {\n#pragma unroll\n'
            '    for (uint32_t i = 0; i < ctx::fragment_acc::NR; ++i) {\n'
            '      fragC.data[i] = __builtin_fmaxf(fragC.data[i], 0.0f);\n'
            '    }\n  }\n\n  // Store the computed C tile to global memory.')
        pc = pc_source.read_text().replace('__kernel void kernel_main(', '__kernel void gemm_pc(')
        pc = pc.replace('    auto output =', '    if (arg->fuse) {\n#pragma unroll\n'
            '      for (uint32_t i = 0; i < ctx::fragment_acc::NR; ++i) {\n'
            '        frag.data[i] = __builtin_fmaxf(frag.data[i], 0.0f);\n'
            '      }\n    }\n    auto output =')
        (HERE / 'gemm_relu/base.cpp').write_text(base)
        (HERE / 'gemm_relu/pc.cpp').write_text(pc)
        for path in (HERE / 'gemm_relu').iterdir():
            if path.is_file():
                shutil.copyfile(path, dest / path.name)
                shutil.copyfile(path, logs / path.name)
        (dest / '.project20-owned').write_text(COMMIT + '\n')
        result['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (HERE / 'gemm_relu').iterdir() if p.is_file()}
        save()
        run('configure', ['../configure', '--xlen=32', f'--tooldir={TOOLS}'])
        run('software', ['make', '-j4', 'software'], timeout=600)
        run('build', ['make', '-j4', '-C', 'tests/regression/gemm_relu'])
        shapes = [(64, 32, 16, 1)] if args.phase == 'smoke' else [
            (64, 32, k, 1) for k in (16, 48, 64, 256)] + [
            (128, 128, k, 1) for k in (64, 256)] + [
            (64, 32, 48, 2026), (128, 128, 64, 2026)]
        if args.phase == 'validation':
            shapes = [(64, 32, 256, 2026)]
        for m, n, k, seed in shapes:
            variants = [(1, 1)] if args.phase == 'activation' else [(mode, 0) for mode in (0, 1, 2, 3)]
            if args.phase == 'validation':
                variants.append((1, 1))
            for mode, activation_stride in variants:
                name = f'm{m}_n{n}_k{k}_s{seed}_v{mode}_a{activation_stride}'
                options = f'-m{m} -n{n} -k{k} -s{seed} -v{mode} -a{activation_stride}'
                text, path = run(name, ['./ci/blackbox.sh', '--driver=simx', '--app=gemm_relu',
                                         '--args=' + options, '--perf=7'])
                stages = [dict(name=a, cycles=int(b), instructions=int(c), reads=int(d), writes=int(e))
                          for a, b, c, d, e in re.findall(
                              r'STAGE (\w+) cycles=(\d+) instructions=(\d+) reads=(\d+) writes=(\d+)', text)]
                chain = re.search(r'CHAIN cycles=(\d+) instructions=(\d+) reads=(\d+) writes=(\d+) launches=(\d+)', text)
                coverage = re.search(r'COVERAGE positive=(\d+) negative=(\d+) zero=(\d+)', text)
                verify = re.search(r'VERIFY errors=(\d+) output_hash=(\d+)', text)
                passed = bool(chain and coverage and verify and verify[1] == '0'
                              and re.search(r'^PASSED!$', text, re.MULTILINE))
                record = dict(name=name, m=m, n=n, k=k, seed=seed, mode=mode, activation_stride=activation_stride,
                              passed=passed, stages=stages, log=str(path))
                record['snapshots'] = [dict(name=a, cumulative_cycles=int(b), cumulative_instructions=int(c))
                    for a, b, c in re.findall(r'SNAPSHOT (\w+) cumulative_cycles=(\d+) cumulative_instructions=(\d+)', text)]
                if chain:
                    record['chain'] = dict(zip(('cycles', 'instructions', 'reads', 'writes', 'launches'),
                                               map(int, chain.groups())))
                if coverage:
                    record['coverage'] = dict(zip(('positive', 'negative', 'zero'), map(int, coverage.groups())))
                if verify:
                    record['output_hash'] = verify[2]
                result['cases'].append(record)
                save()
                print(json.dumps(dict(name=name, passed=passed, chain=record.get('chain'))), flush=True)
                if not passed or len(stages) != (2 if mode == 1 else 1):
                    raise RuntimeError('Correctness or stage capture failed: ' + name)
                if record['chain']['cycles'] != record['snapshots'][-1]['cumulative_cycles']:
                    raise RuntimeError('Chain cycle aggregation double-counted a launch')
        result['status'] = 'complete'
        save()
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        save()
        raise
    print(json.dumps(dict(status=result['status'], logs=str(logs))), flush=True)


if __name__ == '__main__':
    main()
