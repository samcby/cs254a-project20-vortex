#!/usr/bin/env python3
"""Freeze a history-only configuration table before measuring holdout shapes."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
COURSE = HERE.parents[1]
VARIANTS = [(2, 0, 1), (3, 0, 1), (4, 1, 4), (4, 2, 4)]
SHAPES = [(64,128,64),(256,32,64),(256,64,128),(192,128,64),(384,128,256),(320,128,192)]
CONFIG_FIELDS = ('mode','activation_warps','unroll','ctas')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(policy, m, n, k, warps):
    tiles=(m//64)*(n//32)
    anchors=[r for r in policy['anchors'] if r['hardware_warps']==warps]
    anchor=min(anchors,key=lambda r:(abs(math.log2(tiles/r['tiles']))+abs(math.log2(k/r['k'])),r['tiles'],r['k']))
    result={f:anchor['choice'][f] for f in CONFIG_FIELDS}
    result['ctas']=max(g for g in (1,2,4,8) if g<=min(result['ctas'],tiles))
    result.update(anchor_tiles=anchor['tiles'],anchor_k=anchor['k'],
        distance=abs(math.log2(tiles/anchor['tiles']))+abs(math.log2(k/anchor['k'])))
    return result


def freeze():
    files=[('activation_w16_matrix_results.json',16),('activation_w8_capacity_results.json',8),
           ('multicta_w8_matrix_results.json',8),('multicta_w16_matrix_results.json',16)]
    sources={}
    rows=[]
    reference=json.loads((COURSE/files[0][0]).read_text())
    for name,w in files:
        path=COURSE/name
        run=json.loads(path.read_text())
        assert run['status']=='complete' and run['source_hashes']==reference['source_hashes']
        assert run['binary_sha256']==reference['binary_sha256']
        sources[name]=sha(path)
        for c in run['cases']:
            if ((c['mode'],c['activation_warps'],c['unroll']) not in VARIANTS
                    or c['seed']!=1 or c.get('producer_relu',0) or c['c_depth']!=1):
                continue
            m,n=(c['m'],c['n']) if 'm' in c else (64,32*c['tiles'])
            rows.append(dict(hardware_warps=w,m=m,n=n,k=c['k'],tiles=(m//64)*(n//32),
                mode=c['mode'],activation_warps=c['activation_warps'],unroll=c['unroll'],
                ctas=c.get('ctas',1),cycles=c['chain']['cycles'],cta_warps=c['cta_warps'],smem=c['smem'],source=name))
    groups={}
    for r in rows:
        groups.setdefault((r['hardware_warps'],r['tiles'],r['k']),[]).append(r)
    anchors=[]
    for (w,t,k),cs in sorted(groups.items()):
        assert {(c['mode'],c['activation_warps'],c['unroll']) for c in cs}==set(VARIANTS)
        fastest=min(c['cycles'] for c in cs)
        eligible=[c for c in cs if c['cycles']<=1.01*fastest]
        choice=min(eligible,key=lambda c:(c['cta_warps'],c['smem'],c['cycles'],c['ctas']))
        anchors.append(dict(hardware_warps=w,tiles=t,k=k,candidates=len(cs),fastest_cycles=fastest,choice=choice))
    policy=dict(version='history-table-v1',source_commit=reference['source_commit'],
        source_hashes=reference['source_hashes'],binary_sha256=reference['binary_sha256'],
        effective_config_16=reference['effective_config'],training_sources=sources,training_rows=rows,
        rule='Nearest anchor in abs(log2(T/Ta))+abs(log2(K/Ka)); ties prefer lower T then lower K. '
             'At each anchor, configurations within 1% of fastest prefer fewer CTA warps, then SMEM, then cycles, then CTA count. '
             'CTA count is capped to a member of {1,2,4,8} no larger than tile count.',
        near_fast_threshold=0.01,anchors=anchors)
    training_shapes={(r['m'],r['n'],r['k']) for r in rows}
    assert not (training_shapes & set(SHAPES)), 'Holdout shape leaked into table construction'
    encoded=json.dumps(policy,indent=2,ensure_ascii=False)+'\n'
    digest=hashlib.sha256(encoded.encode()).hexdigest()
    manifest=dict(policy_sha256=digest,shapes=[dict(m=m,n=n,k=k,tiles=(m//64)*(n//32)) for m,n,k in SHAPES],
        predictions=[dict(m=m,n=n,k=k,hardware_warps=w,**predict(policy,m,n,k,w)) for w in (8,16) for m,n,k in SHAPES],
        oracle_candidates=[dict(mode=v,activation_warps=a,unroll=u,ctas=g)
            for v,a,u in VARIANTS for g in (1,2,4,8)],
        candidate_filter='ctas <= output tile count',
        evaluation='Policy launches before oracle measurements. Oracle is a diagnostic grid, not an input to this policy version.')
    outputs={'policy.json':encoded,'holdout_manifest.json':json.dumps(manifest,indent=2,ensure_ascii=False)+'\n'}
    HERE.mkdir(exist_ok=True)
    for name,content in outputs.items():
        path=HERE/name
        if path.exists() and path.read_text(encoding='utf-8')!=content:
            raise RuntimeError('Refusing to rewrite the frozen policy/holdout manifest; use a new version')
        if not path.exists():
            path.write_text(content,encoding='utf-8',newline='\n')
    return policy,manifest


if __name__=='__main__':
    policy,manifest=freeze()
    print(json.dumps(dict(policy_sha256=manifest['policy_sha256'],training_rows=len(policy['training_rows']),
        anchors=len(policy['anchors']),holdouts=len(SHAPES),predictions=manifest['predictions'])))
