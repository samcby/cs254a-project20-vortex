#!/usr/bin/env python3
"""Measure supplemental training and frozen v1/v2 selections on new shapes."""

import argparse
from datetime import datetime,timezone
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import TOOLS, result_path, run_log_dir

from policy_model import HERE,OLD,FIELDS,sha
sys.path.insert(0,str(HERE.parent/'warp_capacity'))
from config_probe import SOURCE,config
from run_sweep import COMMIT,parse_case


def identity(c):
    return tuple(c[f] for f in FIELDS)


def grid(shapes,p):
    return [dict(**s,**v,ctas=g,seed=1) for s in shapes for v in p['variants'] for g in p['cta_candidates']
            if g<=(s['m']//64)*(s['n']//32)]


def choices(phase,w,p,manifest):
    if phase=='train':
        return grid(p['train_shapes'],p)
    if phase=='oracle':
        return grid(p['holdout_shapes'],p)
    selected=[]
    oracle=None
    if phase=='validate':
        oracle=json.loads(result_path(f'policy2_w{w}_oracle_results.json').read_text())
        assert oracle['status']=='complete'
    for shape in p['holdout_shapes']:
        m,n,k=(shape[f] for f in ('m','n','k'))
        predictions=[c for c in manifest['predictions'] if (c['hardware_warps'],c['m'],c['n'],c['k'])==(w,m,n,k)]
        unique={}
        for c in predictions:
            key=identity(c)
            unique.setdefault(key,dict(**shape,**{f:c[f] for f in FIELDS},selectors=[]))['selectors'].append(c['selector'])
        if oracle:
            cs=[c for c in oracle['cases'] if (c['m'],c['n'],c['k'])==(m,n,k)]
            for c in (min(cs,key=lambda c:c['chain']['cycles']),min((c for c in cs if c['mode']==2),key=lambda c:c['chain']['cycles'])):
                unique.setdefault(identity(c),dict(**shape,**{f:c[f] for f in FIELDS},selectors=[]))
        selected += [dict(**c,seed=2026 if oracle else 1) for c in unique.values()]
    return selected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['train','policy','oracle','validate'])
    parser.add_argument('--warps',type=int,choices=[8,16],required=True);args=parser.parse_args()
    p=json.loads((HERE/'experiment_plan.json').read_text(encoding='utf-8'))
    assert sha(OLD/'policy.json')==p['v1_policy_sha256']
    assert all(sha(result_path(n))==h for n,h in p['base_training_sources'].items())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=SOURCE,text=True)
    app=SOURCE/'tests/regression/activation_balance'
    assert all(sha(app/n)==h for n,h in p['source_hashes'].items())
    flags,resolved=config(args.warps)
    assert resolved==dict(p['effective_config_16'],VX_CFG_NUM_WARPS=str(args.warps))
    manifest=None;policy=None
    if args.phase!='train':
        policy=json.loads((HERE/'policy_v2.json').read_text(encoding='utf-8'))
        manifest=json.loads((HERE/'holdout_manifest.json').read_text(encoding='utf-8'))
        assert manifest['policy_sha256']==sha(HERE/'policy_v2.json')
        assert policy['plan_sha256']==manifest['plan_sha256']==sha(HERE/'experiment_plan.json')
        assert all(sha(result_path(n))==h for n,h in policy['training_sources'].items())
    if args.phase in ('oracle','validate'):
        for w in (8,16):
            run=json.loads(result_path(f'policy2_w{w}_policy_results.json').read_text())
            assert run['status']=='complete' and run['policy_sha256']==manifest['policy_sha256']
    build=SOURCE/f'build32-warp-capacity-{args.warps}'
    build.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs=run_log_dir(f'policy2_w{args.warps}_{args.phase}_{stamp}');logs.mkdir(parents=True,exist_ok=False)
    for n in p['source_hashes']:
        shutil.copyfile(app/n,logs/n)
    artifacts=['experiment_plan.json']+(['policy_v2.json','holdout_manifest.json'] if policy else [])
    for n in artifacts:
        shutil.copyfile(HERE/n,logs/n)
    result=dict(status='running',phase=args.phase,hardware_warps=args.warps,source_commit=COMMIT,
        source_hashes=p['source_hashes'],plan_sha256=sha(HERE/'experiment_plan.json'),
        policy_sha256=sha(HERE/'policy_v2.json') if policy else None,
        effective_config=resolved,configs=flags,build=str(build),logs=str(logs),steps=[],cases=[])
    env=os.environ.copy()
    for n in ('CONFIGS','DEBUG','PERF','SCOPE', 'SIMX_FUNCTIONAL','VORTEX_DRIVER','VORTEX_PROFILING','VORTEX_HOME','VORTEX_PATH',
              'VORTEX_BUILD_DIR','VORTEX_RT_PATH','VORTEX_KN_PATH','TOOLDIR','TOOLCHAIN_REV','DESTDIR','MAKEFLAGS','MFLAGS'):
        env.pop(n,None)
    env.update(CONFIGS=flags,CCACHE_DISABLE='1',MAKEFLAGS='-j4')
    def save():
        text=json.dumps(result,indent=2)+'\n'
        (logs/'results.json').write_text(text);result_path(f'policy2_w{args.warps}_{args.phase}_results.json').write_text(text)
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
        save();run('configure',['../configure','--xlen=32',f'--tooldir={TOOLS}'])
        run('software',['make','-j4','software'],600);run('build_app',['make','-j4','-C','tests/regression/activation_balance'])
        for n in ('kernel.vxbin','kernel.dump'):
            shutil.copyfile(build/'tests/regression/activation_balance'/n,logs/n)
        result['binary_sha256']=sha(logs/'kernel.vxbin');assert result['binary_sha256']==p['binary_sha256']
        for i,c in enumerate(choices(args.phase,args.warps,p,manifest)):
            m,n,k,g,v,a,u,seed=(c[f] for f in ('m','n','k','ctas','mode','activation_warps','unroll','seed'))
            name=f'm{m}_n{n}_k{k}_g{g}_v{v}_a{a}_u{u}_s{seed}_r{i}'
            text,path=run(name,['./ci/blackbox.sh','--driver=simx','--app=activation_balance',
                f'--args=-m{m} -n{n} -k{k} -g{g} -v{v} -w{a or 1} -u{u} -e0 -q1 -s{seed}','--perf=7'])
            c.update(name=name,log=str(path),**parse_case(text))
            meta=re.search(r'PERSISTENT output_tiles=(\d+) ctas=(\d+) c_depth=(\d+) smem=(\d+)',text)
            if meta:c['smem']=int(meta[4])
            result['cases'].append(c);save()
            print(json.dumps({f:c.get(f) for f in ('name','passed','chain')}),flush=True)
            assert c['passed'] and min(c['coverage'].values())>0 and c['hardware_warps']==args.warps
            assert c['cta_warps']==(5+a if v==4 else v+2)
            assert meta and list(map(int,meta.groups()[:3]))==[(m//64)*(n//32),g,1]
        result['status']='complete';save()
    except Exception as error:
        result.update(status='failed',error=str(error));save();raise
    print(json.dumps(dict(status='complete',cases=len(result['cases']),logs=str(logs))),flush=True)


if __name__=='__main__':main()
