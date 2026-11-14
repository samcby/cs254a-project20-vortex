#!/usr/bin/env python3
"""Measure a frozen configuration selector on held-out shapes and a diagnostic grid."""

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

from freeze_policy import HERE,CONFIG_FIELDS,predict,sha
sys.path.insert(0,str(HERE.parent/'warp_capacity'))
from config_probe import SOURCE,config
from run_sweep import COMMIT,parse_case


def selections(phase,w,policy,manifest):
    if phase=='policy':
        return [dict(m=s['m'],n=s['n'],k=s['k'],seed=1,**predict(policy,s['m'],s['n'],s['k'],w)) for s in manifest['shapes']]
    if phase=='oracle':
        return [dict(m=s['m'],n=s['n'],k=s['k'],seed=1,**c) for s in manifest['shapes']
            for c in manifest['oracle_candidates'] if c['ctas']<=s['tiles']]
    oracle=json.loads(result_path(f'policy_w{w}_oracle_results.json').read_text())
    assert oracle['status']=='complete'
    selected=[]
    for s in manifest['shapes']:
        choices=[c for c in oracle['cases'] if (c['m'],c['n'],c['k'])==(s['m'],s['n'],s['k'])]
        frozen=predict(policy,s['m'],s['n'],s['k'],w)
        configs=[frozen,min(choices,key=lambda c:c['chain']['cycles']),
                 min((c for c in choices if c['mode']==2),key=lambda c:c['chain']['cycles'])]
        unique={tuple(c[f] for f in CONFIG_FIELDS):c for c in configs}
        selected += [dict(m=s['m'],n=s['n'],k=s['k'],seed=2026,**{f:c[f] for f in CONFIG_FIELDS}) for c in unique.values()]
    return selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['policy','oracle','validate'],required=True)
    p.add_argument('--warps',choices=[8,16],type=int,required=True)
    args=p.parse_args()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()!=COMMIT:
        raise RuntimeError('Pinned source required')
    if subprocess.check_output(['git','diff','--name-only'],cwd=SOURCE,text=True):
        raise RuntimeError('Unexpected tracked changes')
    policy=json.loads((HERE/'policy.json').read_text(encoding='utf-8'))
    manifest=json.loads((HERE/'holdout_manifest.json').read_text(encoding='utf-8'))
    assert sha(HERE/'policy.json')==manifest['policy_sha256']
    assert all(sha(result_path(n))==h for n,h in policy['training_sources'].items()), 'Training record changed'
    app=SOURCE/'tests/regression/activation_balance'
    if any(sha(app/n)!=h for n,h in policy['source_hashes'].items()):
        raise RuntimeError('Validated application changed')
    flags,resolved=config(args.warps)
    assert resolved==dict(policy['effective_config_16'],VX_CFG_NUM_WARPS=str(args.warps))
    if args.phase!='policy':
        first=json.loads(result_path(f'policy_w{args.warps}_policy_results.json').read_text())
        assert first['status']=='complete' and first['policy_sha256']==manifest['policy_sha256']
    build=SOURCE/f'build32-warp-capacity-{args.warps}'
    build.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs=run_log_dir(f'policy_w{args.warps}_{args.phase}_{stamp}')
    logs.mkdir(parents=True,exist_ok=False)
    for name in policy['source_hashes']:
        shutil.copyfile(app/name,logs/name)
    for name in ('policy.json','holdout_manifest.json'):
        shutil.copyfile(HERE/name,logs/name)
    result=dict(status='running',phase=args.phase,hardware_warps=args.warps,source_commit=COMMIT,
        source_hashes=policy['source_hashes'],policy_sha256=manifest['policy_sha256'],
        manifest_sha256=sha(HERE/'holdout_manifest.json'),effective_config=resolved,configs=flags,
        build=str(build),logs=str(logs),steps=[],cases=[])
    env=os.environ.copy()
    for name in ('CONFIGS','DEBUG','PERF','SCOPE', 'SIMX_FUNCTIONAL','VORTEX_DRIVER','VORTEX_PROFILING','VORTEX_HOME','VORTEX_PATH',
                 'VORTEX_BUILD_DIR','VORTEX_RT_PATH','VORTEX_KN_PATH','TOOLDIR','TOOLCHAIN_REV','DESTDIR','MAKEFLAGS','MFLAGS'):
        env.pop(name,None)
    env.update(CONFIGS=flags,CCACHE_DISABLE='1',MAKEFLAGS='-j4')

    def save():
        text=json.dumps(result,indent=2)+'\n'
        (logs/'results.json').write_text(text)
        result_path(f'policy_w{args.warps}_{args.phase}_results.json').write_text(text)

    def run(name,command,timeout=300):
        print(name+': '+shlex.join(command),flush=True)
        path=logs/(name+'.log')
        with path.open('w') as out:
            out.write('command='+shlex.join(command)+'\ncwd='+str(build)+'\nCONFIGS='+flags+'\n');out.flush()
            proc=subprocess.Popen(command,cwd=build,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
        result['steps'].append(dict(name=name,command=command,log=str(path),exit_code=code));save()
        text=path.read_text(errors='replace')
        if code:
            raise RuntimeError(name+': exit='+str(code)+'\n'+text[-2400:])
        return text,path

    try:
        save()
        run('configure',['../configure','--xlen=32',f'--tooldir={TOOLS}'])
        run('software',['make','-j4','software'],600)
        run('build_app',['make','-j4','-C','tests/regression/activation_balance'])
        for name in ('kernel.vxbin','kernel.dump'):
            shutil.copyfile(build/'tests/regression/activation_balance'/name,logs/name)
        result['binary_sha256']=sha(logs/'kernel.vxbin')
        assert result['binary_sha256']==policy['binary_sha256']
        for i,c in enumerate(selections(args.phase,args.warps,policy,manifest)):
            m,n,k,g,v,a,u,seed=(c[f] for f in ('m','n','k','ctas','mode','activation_warps','unroll','seed'))
            name=f'm{m}_n{n}_k{k}_g{g}_v{v}_a{a}_u{u}_s{seed}_r{i}'
            command=['./ci/blackbox.sh','--driver=simx','--app=activation_balance',
                f'--args=-m{m} -n{n} -k{k} -g{g} -v{v} -w{a or 1} -u{u} -e0 -q1 -s{seed}','--perf=7']
            text,path=run(name,command)
            c.update(name=name,log=str(path),**parse_case(text))
            meta=re.search(r'PERSISTENT output_tiles=(\d+) ctas=(\d+) c_depth=(\d+) smem=(\d+)',text)
            if meta:
                c['smem']=int(meta[4])
            result['cases'].append(c);save()
            print(json.dumps({f:c.get(f) for f in ('name','passed','chain')}),flush=True)
            assert c['passed'] and min(c['coverage'].values())>0 and c['hardware_warps']==args.warps
            assert c['cta_warps']==(5+a if v==4 else v+2)
            assert meta and list(map(int,meta.groups()[:3]))==[(m//64)*(n//32),g,1]
        result['status']='complete';save()
    except Exception as error:
        result.update(status='failed',error=str(error));save();raise
    print(json.dumps(dict(status='complete',cases=len(result['cases']),logs=str(logs))),flush=True)


if __name__=='__main__':
    main()
