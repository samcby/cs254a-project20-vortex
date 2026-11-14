#!/usr/bin/env python3
"""Install and run the bounded Project 20 producer/consumer experiments in WSL."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import SOURCE, TOOLS, result_path, run_log_dir
import re
import shlex
import shutil
import signal
import subprocess

HERE = Path(__file__).resolve().parent
BUILD = SOURCE / 'build32-pipeline'
COMMIT = '7fe19369587c087f902c3e2bc2add612aa910a95'
FLAGS = ('-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_ISSUE_WIDTH=4 '
         '-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE '
         '-DWGMMA_NRC=16 -DVX_CFG_NUM_ALU_BLOCKS=1 -DVX_CFG_NUM_LSU_BLOCKS=1 '
         '-DVX_CFG_NUM_FPU_BLOCKS=1 -DITYPE=fp16 -DOTYPE=fp32 -DPERF_ENABLE')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['micro', 'gemm', 'diagnostic', 'trace', 'indexing'], required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if actual != COMMIT or subprocess.check_output(['git', 'diff', '--name-only'], cwd=SOURCE, text=True):
        raise RuntimeError('Expected pinned source without tracked modifications')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = run_log_dir('pc_' + args.phase + '_' + stamp)
    logs.mkdir(parents=True, exist_ok=False)
    result = dict(source_commit=actual, phase=args.phase, build=str(BUILD),
                  configs=FLAGS, logs=str(logs), protocol='ready/free with final drain',
                  status='running', steps=[], cases=[])
    resolved = subprocess.check_output(['python3', str(SOURCE / 'ci/gen_config.py'),
        '--config', str(SOURCE / 'VX_config.toml'), '--cflags', FLAGS + ' -DVX_CFG_XLEN=32'], text=True)
    selected = {'VX_CFG_NUM_CORES', 'VX_CFG_NUM_WARPS', 'VX_CFG_NUM_THREADS',
                'VX_CFG_ISSUE_WIDTH', 'VX_CFG_NUM_TCU_BLOCKS', 'VX_CFG_LMEM_LOG_SIZE',
                'VX_CFG_L2_ENABLED', 'VX_CFG_L3_ENABLED', 'VX_CFG_NUM_BARRIERS'}
    result['effective_config'] = {name: value for token in shlex.split(resolved)
        for name, sep, value in [token.removeprefix('-D').partition('=')] if sep and name in selected}
    env = os.environ.copy()
    for name in ('CONFIGS', 'DEBUG', 'PERF', 'SCOPE', 'SIMX_FUNCTIONAL', 'VORTEX_DRIVER', 'VORTEX_PROFILING',
                 'VORTEX_HOME', 'VORTEX_PATH', 'VORTEX_BUILD_DIR', 'VORTEX_RT_PATH',
                 'VORTEX_KN_PATH', 'TOOLDIR', 'TOOLCHAIN_REV', 'DESTDIR', 'MAKEFLAGS', 'MFLAGS'):
        env.pop(name, None)
    env.update(CCACHE_DISABLE='1', MAKEFLAGS='-j4', CONFIGS=FLAGS)

    def save():
        (logs / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        result_path('pc_' + args.phase + '_results.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(name, command, case_env=None, timeout=300, required=True):
        print(name + ': ' + shlex.join(command), flush=True)
        path = logs / (name + '.log')
        step = dict(name=name, command=command, configs=(case_env or env)['CONFIGS'], log=str(path))
        with path.open('w') as out:
            out.write('command=' + shlex.join(command) + '\ncwd=' + str(BUILD)
                      + '\nCONFIGS=' + step['configs'] + '\n')
            out.flush()
            process = subprocess.Popen(command, cwd=BUILD, env=case_env or env,
                                       stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
        step['exit_code'] = code
        result['steps'].append(step)
        save()
        content = path.read_text(errors='replace')
        if code and required:
            raise RuntimeError(name + ': exit=' + str(code) + '\n' + content[-2200:])
        return code, content, path

    def case(name, app, options, case_env=None, profile=7):
        code, content, path = run(name, ['./ci/blackbox.sh', '--driver=simx',
            '--app=' + app, '--args=' + options, '--perf=' + str(profile)], case_env, required=False)
        summary = re.findall(r'PERF:\s+instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)', content)
        record = dict(name=name, app=app, options=options, profile=profile,
                      passed=code == 0 and bool(re.search(r'^PASSED!?$', content, re.MULTILINE)),
                      exit_code=code, summary=summary[-1] if summary else None,
                      perf=[line for line in content.splitlines() if line.startswith('PERF:')],
                      verification=[line for line in content.splitlines()
                                    if line.startswith(('reference drift:', 'smem:', 'CTA tile:', 'tiles='))],
                      tiles=[dict(tile=int(m[0]), issue=int(m[1]), begin=int(m[2]), end=int(m[3]))
                             for m in re.findall(r'TILE (\d+) issue=(\d+) begin=(\d+) end=(\d+)', content)],
                      log=str(path))
        result['cases'].append(record)
        save()
        print(json.dumps({key: record[key] for key in ['name', 'passed', 'summary']}), flush=True)
        if not record['passed']:
            raise RuntimeError(name + ': failed correctness or completion; see ' + str(path))

    def install_gemm():
        original = SOURCE / 'tests/regression/sgemm_tcu_wg_dxa'
        dest = SOURCE / 'tests/regression/sgemm_tcu_wg_pc'
        if dest.exists() and not (dest / '.project20-owned').exists():
            raise RuntimeError('Refusing to replace existing app ' + str(dest))
        dest.mkdir(exist_ok=True)
        host = (original / 'main.cpp').read_text()

        def replace_once(before, after):
            nonlocal host
            if host.count(before) != 1:
                raise RuntimeError('Host source no longer matches: ' + before)
            host = host.replace(before, after)

        replace_once('uint32_t warps = 0;', 'uint32_t warps = 0;\nuint32_t buffer_depth = 2;')
        comment_start = host.index('// `warps` (= warps per CTA')
        comment_end = host.index('uint32_t warps = 0;', comment_start)
        host = host[:comment_start] + '// Full compute warpgroup; a separate fifth warp issues DXA.\n' + host[comment_end:]
        comment_start = host.index('  // WGMMA group size = ISSUE_WIDTH.')
        comment_end = host.index('  uint64_t issue_width;', comment_start)
        host = host[:comment_start] + '  // Preserve one complete WGMMA group and reserve one loader warp.\n' + host[comment_end:]
        replace_once('"m:n:k:h"', '"m:n:k:b:h"')
        replace_once("case 'm':", "case 'b': buffer_depth = atoi(optarg); break;\n    case 'm':")
        replace_once('parse_args(argc, argv);',
                     'parse_args(argc, argv);\n  if (buffer_depth != 1 && buffer_depth != 2) { return 1; }')
        replace_once('if (warps > num_warps)', 'if (warps + 1 > num_warps)')
        replace_once('uint32_t block_dim[2] = {warps * (uint32_t)NT, 1};',
                     'uint32_t block_dim[2] = {(warps + 1) * (uint32_t)NT, 1};')
        start = host.index('  // SMEM:')
        end = host.index('\n  std::cout << "input type:', start)
        host = host[:start] + ('  uint32_t smem_size = buffer_depth * (cta_M * cfg::tileK + '
                              'cfg::tileK * cfg::xtileN) * sizeof(itype_t);\n') + host[end:]
        replace_once('kernel_arg.K = K;', 'kernel_arg.K = K;\n  kernel_arg.depth = buffer_depth;')
        replace_once('std::cout << "smem: "',
                     'std::cout << "active CTA warps: " << (warps + 1) << ", depth=" << buffer_depth << std::endl;\n  std::cout << "smem: "')
        (dest / 'main.cpp').write_text(host)
        common = (original / 'common.h').read_text().replace('uint32_t M, N, K;',
                                                            'uint32_t M, N, K;\n  uint32_t depth;')
        (dest / 'common.h').write_text(common)
        (dest / 'Makefile').write_text((original / 'Makefile').read_text().replace(
            'PROJECT := sgemm_tcu_wg_dxa', 'PROJECT := sgemm_tcu_wg_pc'))
        shutil.copyfile(HERE / 'gemm_kernel.cpp', dest / 'kernel.cpp')
        (dest / '.project20-owned').write_text(COMMIT + '\n')
        mirror = HERE / 'sgemm_tcu_wg_pc'
        mirror.mkdir(exist_ok=True)
        for name in ('main.cpp', 'common.h', 'Makefile', 'kernel.cpp'):
            shutil.copyfile(dest / name, mirror / name)
            shutil.copyfile(dest / name, logs / ('gemm_' + name))
        result['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in mirror.iterdir() if p.is_file()}

    def install_control():
        original = SOURCE / 'tests/regression/sgemm_tcu_wg_dxa'
        pc = SOURCE / 'tests/regression/sgemm_tcu_wg_pc'
        dest = SOURCE / 'tests/regression/sgemm_tcu_wg_control5'
        if dest.exists() and not (dest / '.project20-owned').exists():
            raise RuntimeError('Refusing to replace existing app ' + str(dest))
        dest.mkdir(exist_ok=True)
        host = (pc / 'main.cpp').read_text().replace(
            '// Full compute warpgroup; a separate fifth warp issues DXA.',
            '// The fifth warp only participates in barriers; the original compute group issues DXA.').replace(
            '// Preserve one complete WGMMA group and reserve one loader warp.',
            '// Match the producer/consumer CTA footprint with a synchronization-only fifth warp.')
        (dest / 'main.cpp').write_text(host)
        shutil.copyfile(pc / 'common.h', dest / 'common.h')
        (dest / 'Makefile').write_text((original / 'Makefile').read_text().replace(
            'PROJECT := sgemm_tcu_wg_dxa', 'PROJECT := sgemm_tcu_wg_control5'))
        code = (original / 'kernel.cpp').read_text()
        code = code.replace('uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;',
                            'uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS - 1;')
        code = code.replace('    auto A_warp =', '    if (warp_rank >= num_warps) { return; }\n    auto A_warp =')
        code = code.replace('  // Store the computed C tile to global memory.',
                            '  if (warp_rank >= num_warps) { return; }\n  // Store the computed C tile to global memory.')
        (dest / 'kernel.cpp').write_text(code)
        (dest / '.project20-owned').write_text(COMMIT + '\n')
        mirror = HERE / 'sgemm_tcu_wg_control5'
        mirror.mkdir(exist_ok=True)
        for name in ('main.cpp', 'common.h', 'Makefile', 'kernel.cpp'):
            shutil.copyfile(dest / name, mirror / name)
            shutil.copyfile(dest / name, logs / ('control_' + name))

    def install_indexing():
        original = SOURCE / 'tests/regression/sgemm_tcu_wg_pc'
        dest = SOURCE / 'tests/regression/sgemm_tcu_wg_pc_mask'
        if dest.exists() and not (dest / '.project20-owned').exists():
            raise RuntimeError('Refusing to replace existing app ' + str(dest))
        dest.mkdir(exist_ok=True)
        for name in ('main.cpp', 'common.h', 'Makefile', 'kernel.cpp'):
            content = (original / name).read_text()
            if name == 'Makefile':
                content = content.replace('PROJECT := sgemm_tcu_wg_pc', 'PROJECT := sgemm_tcu_wg_pc_mask')
            elif name == 'kernel.cpp':
                if content.count('i % depth') != 3:
                    raise RuntimeError('Expected exactly three buffer-index expressions')
                content = content.replace('i % depth', 'i & (depth - 1)')
            (dest / name).write_text(content)
            (logs / ('mask_' + name)).write_text(content)
        (dest / '.project20-owned').write_text(COMMIT + '\n')
        mirror = HERE / 'sgemm_tcu_wg_pc_mask'
        mirror.mkdir(exist_ok=True)
        for name in ('main.cpp', 'common.h', 'Makefile', 'kernel.cpp'):
            shutil.copyfile(dest / name, mirror / name)
        result['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in mirror.iterdir() if p.is_file()}

    try:
        if args.phase == 'micro':
            app = 'dxa_pc'
            dest = SOURCE / 'tests/regression' / app
            if dest.exists() and not (dest / '.project20-owned').exists():
                raise RuntimeError('Refusing to replace existing app ' + str(dest))
            dest.mkdir(exist_ok=True)
            for path in (HERE / app).iterdir():
                if path.is_file():
                    shutil.copyfile(path, dest / path.name)
                    shutil.copyfile(path, logs / ('micro_' + path.name))
            (dest / '.project20-owned').write_text(COMMIT + '\n')
            result['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (HERE / app).iterdir() if p.is_file()}
        elif args.phase == 'gemm':
            install_gemm()
        elif args.phase == 'diagnostic':
            install_control()
        elif args.phase == 'indexing':
            install_indexing()
        BUILD.mkdir(exist_ok=True)
        run('configure', ['../configure', '--xlen=32', f'--tooldir={TOOLS}'])
        run('software', ['make', '-j4', 'software'], timeout=600)
        if args.phase == 'micro':
            run('build_micro', ['make', '-j4', '-C', 'tests/regression/dxa_pc'])
            for depth in (1, 2):
                for tiles in (1, 2, 3, 9):
                    case('pc_d' + str(depth) + '_n' + str(tiles), 'dxa_pc',
                         f'-p1 -d{depth} -n{tiles} -w0')
            for work in (0, 32):
                for pipeline in (0, 1):
                    case(f'perf_p{pipeline}_w{work}', 'dxa_pc', f'-p{pipeline} -d2 -n64 -w{work}')
            for pipeline in (0, 1):
                case('trace_p' + str(pipeline), 'dxa_pc', f'-p{pipeline} -d2 -n9 -w32 -t')
        elif args.phase == 'gemm':
            run('build_pc_gemm', ['make', '-j4', '-C', 'tests/regression/sgemm_tcu_wg_pc'])
            for depth in (1, 2):
                for k in (16, 32, 48, 64, 256):
                    case(f'pc_d{depth}_small_k{k}', 'sgemm_tcu_wg_pc', f'-m64 -n32 -k{k} -b{depth}')
                for k in (32, 64, 256):
                    case(f'pc_d{depth}_large_k{k}', 'sgemm_tcu_wg_pc', f'-m128 -n128 -k{k} -b{depth}')
            for variant in ('single', 'double'):
                case_env = {**env, 'CONFIGS': FLAGS + (' -DDXA_DOUBLE_BUFFER' if variant == 'double' else '')}
                run('build_baseline_' + variant,
                    ['make', '-j4', '-C', 'tests/regression/sgemm_tcu_wg_dxa'], case_env)
                for k in (16, 32, 48, 64, 256):
                    case(f'{variant}_small_k{k}', 'sgemm_tcu_wg_dxa', f'-m64 -n32 -k{k}', case_env)
                case(f'{variant}_large_k64', 'sgemm_tcu_wg_dxa', '-m128 -n128 -k64', case_env)
        elif args.phase == 'indexing':
            run('build_pc_mask', ['make', '-j4', '-C', 'tests/regression/sgemm_tcu_wg_pc_mask'])
            for depth, m, n, k in [(1, 64, 32, 16), (1, 64, 32, 48)] + [
                    (2, 64, 32, k) for k in (16, 32, 48, 64, 256)] + [
                    (2, 128, 128, k) for k in (64, 256)]:
                case(f'mask_d{depth}_m{m}_n{n}_k{k}', 'sgemm_tcu_wg_pc_mask',
                     f'-m{m} -n{n} -k{k} -b{depth}')
            dump = (BUILD / 'tests/regression/sgemm_tcu_wg_pc_mask/kernel.dump').read_text()
            result['kernel_text_has_remu'] = bool(re.search(r'\bremu\b', dump.split('Disassembly of section .comment:')[0]))
            shutil.copyfile(BUILD / 'tests/regression/sgemm_tcu_wg_pc_mask/kernel.dump', logs / 'mask_kernel.dump')
            shutil.copyfile(BUILD / 'tests/regression/sgemm_tcu_wg_pc/kernel.dump', logs / 'original_pc_kernel.dump')
        else:
            if args.phase == 'diagnostic':
                run('build_micro', ['make', '-j4', '-C', 'tests/regression/dxa_pc'])
                for work in (0, 32):
                    case(f'control_pc_d1_w{work}', 'dxa_pc', f'-p1 -d1 -n64 -w{work}')
                control_env = {**env, 'CONFIGS': FLAGS + ' -DDXA_DOUBLE_BUFFER'}
                run('build_control5', ['make', '-j4', '-C', 'tests/regression/sgemm_tcu_wg_control5'], control_env)
                for m, n in ((64, 32), (128, 128)):
                    for k in (64, 256):
                        case(f'control5_m{m}_n{n}_k{k}', 'sgemm_tcu_wg_control5', f'-m{m} -n{n} -k{k} -b2', control_env)
                result['controls_status'] = 'complete'
            run('build_pc_trace', ['make', '-j4', '-C', 'tests/regression/sgemm_tcu_wg_pc'])
            for depth in (1, 2):
                raw = logs / f'gemm_depth{depth}.trace.log'
                # blackbox's shell redirection requires a space-free log path.
                linux_raw = BUILD / f'pc_gemm_depth{depth}.trace.log'
                code, _, _ = run(f'trace_depth{depth}', ['./ci/blackbox.sh', '--driver=simx',
                    '--app=sgemm_tcu_wg_pc', f'--args=-m64 -n32 -k64 -b{depth}',
                    '--debug=3', '--perf=16', '--log=' + str(linux_raw)])
                shutil.copyfile(linux_raw, raw)
                content = raw.read_text(errors='replace')
                if code or not re.search(r'^PASSED!?$', content, re.MULTILINE):
                    raise RuntimeError('Debug trace did not complete correctly: ' + str(raw))
                result.setdefault('debug_traces', []).append(dict(depth=depth, log=str(raw),
                    bytes=raw.stat().st_size, passed=True))
                save()
        result['status'] = 'complete'
        result['git_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=SOURCE, text=True)
        save()
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        save()
        raise
    print(json.dumps(dict(status=result['status'], logs=str(logs))), flush=True)


if __name__ == '__main__':
    main()
