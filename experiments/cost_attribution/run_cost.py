#!/usr/bin/env python3
"""Run finite, resource-aware kernel ablations and isolated timing-neutral traces."""

from datetime import datetime,timezone
import argparse
import gzip
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

from prepare_cost import HERE,SOURCE,OVERLAY,APP,prepare_app,prepare_capacity,prepare_observer,once,sha
sys.path.insert(0,str(HERE.parent/'warp_capacity'))
from config_probe import config
from run_sweep import COMMIT,parse_case

METHODS=[(2,0,0),(3,0,0),(3,0,1),(9,0,1),(5,0,0),(5,0,1),(6,0,1),(7,0,1),(4,1,0),(8,1,0),(4,2,0),(8,2,0)]


def cases(phase,w,capacity=False):
    k,g,a=(64,2,1) if w==16 else (256,4,2)
    if capacity:
        if phase=='normal':selected=[(gg,v,1) for gg in ((g,1) if w==8 else (g,)) for v in (2,10,11)]
        elif phase=='validate':selected=[(g,v,s) for s in (1,2026) for v in (10,11)]
        elif phase=='trace':selected=[(g,v,1) for v in (10,11)]
        else:selected=[(1,v,1) for v in (10,11)]
        return [dict(m=256,n=128,k=k,ctas=gg,mode=v,aw=0,reserve=0,seed=s) for gg,v,s in selected]
    if phase=='smoke':
        return [dict(m=64,n=32,k=16,ctas=1,mode=v,aw=aa,reserve=r,seed=1) for v,aa,r in ((9,0,1),(8,2,0),(7,0,1))]
    if phase=='normal':
        selected=[(g,v,aa,r,1) for v,aa,r in METHODS]
        selected += [(1,v,aa,r,1) for v,aa,r in ((2,0,0),(3,0,0),(9,0,1),(4,a,0),(8,a,0))]
    elif phase=='validate':
        selected=[(g,v,aa,r,seed) for seed in (1,2026) for v,aa,r in ((3,0,0),(9,0,1),(4,a,0),(8,a,0))]
    else:
        selected=[(g,v,aa,r,1) for v,aa,r in ((2,0,0),(3,0,0),(9,0,1),(7,0,1),(4,a,0),(8,a,0))]
        selected += [(1,v,aa,r,1) for v,aa,r in ((2,0,0),(4,a,0))]
    return [dict(m=256,n=128,k=k,ctas=gg,mode=v,aw=aa,reserve=r,seed=seed) for gg,v,aa,r,seed in selected]


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','normal','validate','trace'],required=True)
    p.add_argument('--warps',type=int,choices=[8,16],required=True);p.add_argument('--capacity',action='store_true');args=p.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=SOURCE,text=True)
    hashes,mirror=prepare_capacity() if args.capacity else (prepare_app(),APP)
    app_name='cost_capacity' if args.capacity else 'cost_pipeline'
    prefix='cost_capacity' if args.capacity else 'cost'
    app=SOURCE/('tests/regression/'+app_name)
    if app.exists() and not (app/'.project20-owned').exists():raise RuntimeError('Unowned application')
    app.mkdir(exist_ok=True)
    for n in hashes:shutil.copyfile(mirror/n,app/n)
    (app/'.project20-owned').write_text(COMMIT+'\n')
    observer=prepare_observer() if args.phase=='trace' else None
    build=SOURCE/(f'build32-cost-observer-{args.warps}' if observer else f'build32-warp-capacity-{args.warps}')
    build.mkdir(exist_ok=True)
    flags,resolved=config(args.warps)
    reference=json.loads(result_path('activation_w16_matrix_results.json').read_text())
    assert resolved==dict(reference['effective_config'],VX_CFG_NUM_WARPS=str(args.warps))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs=run_log_dir(f'{prefix}_w{args.warps}_{args.phase}_{stamp}');logs.mkdir(parents=True,exist_ok=False)
    for n in hashes:shutil.copyfile(mirror/n,logs/n)
    if observer:shutil.copyfile(HERE/'observer_sources.json',logs/'observer_sources.json')
    result=dict(status='running',phase=args.phase,warps=args.warps,source_commit=COMMIT,source_hashes=hashes,
        configs=flags,effective_config=resolved,observer=observer,build=str(build),logs=str(logs),steps=[],cases=[])
    env=os.environ.copy()
    for n in ('CONFIGS','DEBUG','PERF','SCOPE', 'SIMX_FUNCTIONAL','VORTEX_DRIVER','VORTEX_PROFILING','VORTEX_HOME','VORTEX_PATH',
              'VORTEX_BUILD_DIR','VORTEX_RT_PATH','VORTEX_KN_PATH','TOOLDIR','TOOLCHAIN_REV','DESTDIR','MAKEFLAGS','MFLAGS'):
        env.pop(n,None)
    env.update(CONFIGS=flags,CCACHE_DISABLE='1',MAKEFLAGS='-j4')
    def save():
        t=json.dumps(result,indent=2)+'\n';(logs/'results.json').write_text(t);result_path(f'{prefix}_w{args.warps}_{args.phase}_results.json').write_text(t)
    def run(name,command,timeout=300):
        print(name+': '+shlex.join(command),flush=True);path=logs/(name+'.log')
        with path.open('w') as out:
            out.write('command='+shlex.join(command)+'\ncwd='+str(build)+'\nCONFIGS='+flags+'\n');out.flush()
            proc=subprocess.Popen(command,cwd=build,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
        result['steps'].append(dict(name=name,command=command,log=str(path),exit_code=code));save()
        text=path.read_text(errors='replace')
        if code:raise RuntimeError(name+': exit='+str(code)+'\n'+text[-2600:])
        return text,path
    try:
        save();run('configure',['../configure','--xlen=32',f'--tooldir={TOOLS}'])
        if observer:
            mf=build/'sim/simx/Makefile'
            original='SRC_DIR = $(VORTEX_HOME)/sim/simx';observed='SRC_DIR = '+str(OVERLAY)
            current=mf.read_text()
            if original in current:mf.write_text(once(current,original,observed))
            else:assert current.count(observed)==1
            shutil.copyfile(mf,logs/'observer_simx_Makefile')
        run('software',['make','-j4','software'],600);run('build_app',['make','-j4','-C','tests/regression/'+app_name])
        for n in ('kernel.vxbin','kernel.dump'):shutil.copyfile(build/('tests/regression/'+app_name)/n,logs/n)
        result['binary_sha256']=sha(logs/'kernel.vxbin')
        for index,c in enumerate(cases(args.phase,args.warps,args.capacity)):
            m,n,k,g,v,a,r,seed=(c[x] for x in ('m','n','k','ctas','mode','aw','reserve','seed'))
            name=f'm{m}_n{n}_k{k}_g{g}_v{v}_a{a}_r{r}_s{seed}_i{index}'
            cmd=['./ci/blackbox.sh','--driver=simx','--app='+app_name,
                f'--args=-m{m} -n{n} -k{k} -g{g} -v{v} -w{a or 1} -u4 -e0 -q1 -r{r} -s{seed}','--perf=7']
            raw=None
            if observer:
                raw=build/'cost.trace.log';cmd += ['--debug=3','--log='+str(raw)]
            text,path=run(name,cmd,timeout=600 if observer else 300)
            if raw:
                with raw.open('rb') as src,gzip.open(logs/(name+'.trace.log.gz'),'wb') as dst:shutil.copyfileobj(src,dst)
                text=raw.read_text(errors='replace');c['trace']=str(logs/(name+'.trace.log.gz'))
            c.update(log=str(path),**parse_case(text))
            c['core_counters']={int(x):int(y) for x,y in re.findall(r'COST_COUNTER csr=(\d+) value=(\d+)',text)}
            mm=re.search(r'COST_MAP A=(\d+) B=(\d+) Y=(\d+) A_bytes=(\d+) B_bytes=(\d+) Y_bytes=(\d+)',text)
            c['memory_map']=dict(zip(('A','B','Y','A_bytes','B_bytes','Y_bytes'),map(int,mm.groups()))) if mm else None
            meta=re.search(r'PERSISTENT output_tiles=(\d+) ctas=(\d+) c_depth=(\d+) smem=(\d+)',text)
            c['smem']=int(meta[4]) if meta else None
            result['cases'].append(c);save()
            print(json.dumps(dict(case=name,passed=c['passed'],chain=c.get('chain'))),flush=True)
            expected=(v-6 if v>=10 else 5+a if v in (4,8) else v if v in (5,6,7) else 5 if v==9 else v+2)
            assert c['passed'] and c['cta_warps']==expected and c['hardware_warps']==args.warps
            assert len(c['core_counters'])==18 and c['memory_map']
        result['status']='complete';save()
    except Exception as e:result.update(status='failed',error=str(e));save();raise
    print(json.dumps(dict(status='complete',cases=len(result['cases']),logs=str(logs))),flush=True)


if __name__=='__main__':main()
