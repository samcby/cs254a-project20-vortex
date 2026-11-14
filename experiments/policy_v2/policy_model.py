#!/usr/bin/env python3
"""Freeze the v2 data/feature ablation and an independent holdout plan."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
COURSE=HERE.parents[1]
OLD=HERE.parent/'policy_generalization'
FIELDS=('mode','activation_warps','unroll','ctas')
VARIANTS=[(2,0,1),(3,0,1),(4,1,4),(4,2,4)]
TRAIN=[(128,128,96),(128,128,128),(256,128,128)]
TEST=[(128,96,128),(192,64,128),(256,96,144),(192,128,112),(384,96,160),(256,192,224)]
spec=importlib.util.spec_from_file_location('project20_v1_frozen',OLD/'freeze_policy.py')
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path,value):
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if path.exists() and path.read_text(encoding='utf-8')!=text:
        raise RuntimeError('Refusing to overwrite a different frozen artifact: '+str(path))
    if not path.exists():
        path.write_text(text,encoding='utf-8',newline='\n')


def base_training():
    v1=json.loads((OLD/'policy.json').read_text(encoding='utf-8'))
    assert sha(OLD/'policy.json')==json.loads((OLD/'holdout_manifest.json').read_text(encoding='utf-8'))['policy_sha256']
    sources={n:h for n,h in v1['training_sources'].items()}
    rows=list(v1['training_rows'])
    for w in (8,16):
        name=f'policy_w{w}_oracle_results.json'
        run=json.loads((COURSE/name).read_text())
        assert run['status']=='complete' and run['source_hashes']==v1['source_hashes'] and run['binary_sha256']==v1['binary_sha256']
        sources[name]=sha(COURSE/name)
        for c in run['cases']:
            rows.append(row(c,w,name))
    return v1,sources,rows


def row(c,w,source):
    return dict(hardware_warps=w,m=c['m'],n=c['n'],k=c['k'],tiles=(c['m']//64)*(c['n']//32),
        **{f:c[f] for f in FIELDS},cycles=c['chain']['cycles'],cta_warps=c['cta_warps'],smem=c['smem'],source=source)


def plan():
    v1,sources,rows=base_training()
    old_shapes={(r['m'],r['n'],r['k']) for r in rows}
    assert not (set(TEST)&(old_shapes|set(TRAIN)))
    assert not (set(TRAIN)&old_shapes)
    value=dict(version='policy-v2-ablation-plan',v1_policy_sha256=sha(OLD/'policy.json'),
        source_commit=v1['source_commit'],source_hashes=v1['source_hashes'],binary_sha256=v1['binary_sha256'],
        effective_config_16=v1['effective_config_16'],base_training_sources=sources,
        train_shapes=[dict(m=m,n=n,k=k) for m,n,k in TRAIN],
        holdout_shapes=[dict(m=m,n=n,k=k) for m,n,k in TEST],
        variants=[dict(mode=v,activation_warps=a,unroll=u) for v,a,u in VARIANTS],cta_candidates=[1,2,4,8],
        near_fast_threshold=0.01,
        models=dict(v2_tk='abs(log2(T/Ta))+abs(log2(K/Ka))',
                    v2_mnk='abs(log2(M/Ma))+abs(log2(N/Na))+abs(log2(K/Ka))'),
        tie_break='distance, anchor tile count, anchor K, anchor M, anchor N',
        choice='Per exact shape, within 1% fastest prefer fewer CTA warps, then SMEM, cycles, CTA count.',
        evaluation='Freeze both v2 selectors before new holdout measurements. Compare v1/v2-tk/v2-mnk on the same new holdouts. '
                   'v1 holdouts are now v2 training, never independent v2 validation.')
    write_frozen(HERE/'experiment_plan.json',value)
    return value


def predict(policy,kind,m,n,k,w):
    if kind=='v1':
        old=json.loads((OLD/'policy.json').read_text(encoding='utf-8'))
        result=legacy.predict(old,m,n,k,w)
        return {f:result[f] for f in FIELDS}
    t=(m//64)*(n//32)
    def distance(a):
        if kind=='v2_tk':
            return abs(math.log2(t/a['tiles']))+abs(math.log2(k/a['k']))
        return abs(math.log2(m/a['m']))+abs(math.log2(n/a['n']))+abs(math.log2(k/a['k']))
    a=min((a for a in policy['anchors'] if a['hardware_warps']==w),
          key=lambda a:(distance(a),a['tiles'],a['k'],a['m'],a['n']))
    result={f:a['choice'][f] for f in FIELDS}
    result['ctas']=max(g for g in (1,2,4,8) if g<=min(t,result['ctas']))
    return result


def freeze():
    p=plan()
    v1,sources,rows=base_training()
    for w in (8,16):
        name=f'policy2_w{w}_train_results.json'
        run=json.loads((COURSE/name).read_text())
        assert run['status']=='complete' and run['plan_sha256']==sha(HERE/'experiment_plan.json')
        assert run['source_hashes']==v1['source_hashes'] and run['binary_sha256']==v1['binary_sha256']
        sources[name]=sha(COURSE/name)
        rows.extend(row(c,w,name) for c in run['cases'])
    grouped={}
    for r in rows:
        grouped.setdefault((r['hardware_warps'],r['m'],r['n'],r['k']),[]).append(r)
    anchors=[]
    for (w,m,n,k),cs in sorted(grouped.items()):
        assert len({tuple(c[f] for f in FIELDS) for c in cs})==len(cs)
        fastest=min(c['cycles'] for c in cs)
        choice=min((c for c in cs if c['cycles']<=1.01*fastest),
                   key=lambda c:(c['cta_warps'],c['smem'],c['cycles'],c['ctas']))
        anchors.append(dict(hardware_warps=w,m=m,n=n,k=k,tiles=(m//64)*(n//32),
                            candidates=len(cs),fastest_cycles=fastest,choice=choice))
    policy=dict(version='history-table-v2-ablation',plan_sha256=sha(HERE/'experiment_plan.json'),
        v1_policy_sha256=p['v1_policy_sha256'],source_hashes=p['source_hashes'],binary_sha256=p['binary_sha256'],
        training_sources=sources,training_rows=rows,anchors=anchors,models=p['models'],near_fast_threshold=0.01,
        note='Anchors are exact M/N/K shapes; never minimize absolute cycles across different geometries.')
    write_frozen(HERE/'policy_v2.json',policy)
    manifest=dict(plan_sha256=sha(HERE/'experiment_plan.json'),policy_sha256=sha(HERE/'policy_v2.json'),
        predictions=[dict(selector=kind,hardware_warps=w,m=s['m'],n=s['n'],k=s['k'],
            **predict(policy,kind,s['m'],s['n'],s['k'],w)) for w in (8,16) for s in p['holdout_shapes']
                     for kind in ('v1','v2_tk','v2_mnk')])
    write_frozen(HERE/'holdout_manifest.json',manifest)
    return policy,manifest


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');args=a.parse_args()
    if args.freeze:
        policy,manifest=freeze()
        print(json.dumps(dict(policy_sha256=manifest['policy_sha256'],rows=len(policy['training_rows']),anchors=len(policy['anchors']),predictions=manifest['predictions'])))
    else:
        p=plan();print(json.dumps(dict(plan_sha256=sha(HERE/'experiment_plan.json'),train=p['train_shapes'],test=p['holdout_shapes'])))
