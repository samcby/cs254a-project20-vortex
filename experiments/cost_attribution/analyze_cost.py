#!/usr/bin/env python3
"""Attribute measured costs without treating overlapping waits as additive time."""
from bisect import bisect_right
from collections import Counter,defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import log_path, result_path
import re

HERE=Path(__file__).resolve().parent
FIELDS=('m','n','k','ctas','mode','aw','reserve','seed')
def local(p):
    return log_path(p)
def key(c):return tuple(c[x] for x in FIELDS)
def label(c):
    v,a,r=c['mode'],c['aw'],c['reserve']
    return {2:'普通融合',3:'两阶段'+('，预留14KiB' if r else ''),
        4:f'三阶段 A{a}/U4',5:f'两阶段+末尾join5，预留{14 if r else 6}KiB',
        6:'两阶段+1闲置warp+join6',7:'两阶段+2闲置warp+join7',
        8:f'三阶段 A{a}，提前等待C槽',9:'两阶段+C共享内存往返',
        10:'普通融合+末尾join4',11:'普通融合+1闲置warp+join5'}[v]
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']
        +['| '+' | '.join(map(str,r))+' |' for r in rows])
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def role(rank,mode):
    if mode>=10:return 'compute' if rank<4 else 'idle'
    return 'compute' if rank<4 else 'loader' if rank==4 else 'activation' if mode in (4,8) else 'idle'
def bar_name(bar,c):
    b=bar%8
    if c['mode']==2:return 'input_ready_or_reuse'
    if c['mode']>=10:return 'end_join' if b==7 else 'input_ready_or_reuse'
    if c['mode'] in (5,6,7) and b==7:return 'end_join'
    if c['mode']==9 and b==4:return 'C_roundtrip'
    return ('input_ready' if b<2 else 'input_free' if b<4 else 'C_ready' if b<6 else 'C_free')

def trace(c):
    timed=re.compile(r'^TRACE\s+(\d+): (.*)')
    pairs=re.compile(r'(\w+)=(\d+)')
    mem=re.compile(r'core0-lsu mem-req: op=(\w+), mask=([01]+), addr=\{([^}]+)\}.*\(#(\d+)\)')
    mathop=re.compile(r'(tcu execute: op=WGMMA|fpu execute: op=FMIN_MAX),.*wid=(\d+),')
    active={};lifetime={};res_events=[];waits=[];pending={};states=defaultdict(list)
    arrived={};arrival_hist=defaultdict(list);last_arrivals=Counter();missing_wait=Counter()
    gate=defaultdict(lambda:dict(blocks=set()));gate_ticks=set();gate_events=0
    gate_groups=Counter();gate_instructions=Counter();wait_groups=Counter();dependency=Counter()
    memory=Counter();compute=defaultdict(list);activation=defaultdict(list);transfers=0
    mm=c['memory_map'];num_addresses=Counter();max_tick=0
    with gzip.open(local(c['trace']),'rt',errors='replace') as src:
        for line in src:
            if 'dxa-core[' in line and ' start:' in line:transfers+=1
            if not ('COST_' in line or 'core0-lsu mem-req:' in line or 'tcu execute: op=WGMMA' in line or 'fpu execute: op=FMIN_MAX' in line):continue
            m=timed.match(line)
            if not m:
                assert line.startswith(('COST_MAP','COST_COUNTER')),line[:120]
                continue
            t,body=int(m[1]),m[2];max_tick=max(max_tick,t)
            if 'COST_' in body:
                d={n:int(v) for n,v in pairs.findall(body)}
                if body.startswith('COST_DISPATCH'):
                    wid=d['wid'];assert wid not in active
                    active[wid]=d
                    ident=(d['block'],d['rank']);assert ident not in lifetime
                    lifetime[ident]=dict(start=t,**d)
                    res_events.append((t,'dispatch',wid,d.copy()))
                elif body.startswith('COST_EXIT'):
                    wid=d['wid'];old=active.pop(wid);assert old['slot']==d['slot'] and wid not in pending
                    lifetime[(old['block'],old['rank'])]['end']=t
                    res_events.append((t,'exit',wid,old.copy()))
                elif body.startswith('COST_GATE'):
                    uid=d['uuid'];g=gate[uid]
                    if ' block ' in body:
                        assert t not in g['blocks'];g['blocks'].add(t);gate_ticks.add(t);gate_events+=1
                    else:g['pass']=t
                    g['wid']=d['wid']
                elif body.startswith('COST_OP'):
                    g=gate[d['uuid']];assert g['pass']==t and g['wid']==d['wid']
                    g.update(d)
                    old=active[d['wid']];group=bar_name(d['bar'],c)+'/'+role(old['rank'],c['mode'])
                    g['group']=group;g['block']=old['block']
                    gate_groups[group]+=len(g['blocks']);gate_instructions[group]+=1
                elif body.startswith('COST_ARR'):
                    old=active[d['wid']];bar=d['bar']
                    ph,seen=arrived.get(bar,(d['phase'],set()))
                    # DXA may finish before the first arrival, toggling the phase
                    # independently. A zero arrival count starts a new arrival group.
                    if ph!=d['phase'] or d['count']==0:seen=set()
                    assert len(seen)==d['count'] and old['rank'] not in seen,(c['mode'],t,d,ph,seen,old)
                    seen=seen|{old['rank']};arrived[bar]=(d['phase'],seen)
                    b=bar%8
                    expected=(set(range(d['target'])) if c['mode'] in (2,5,6,7,9,10,11) and (b<4 or b==7 or c['mode']==9)
                        else set(range(5)) if b<4 else set(range(4))|set(range(5,5+c['aw'])))
                    assert len(expected)==d['target'] and seen<=expected
                    missing=sorted({role(rank,c['mode']) for rank in expected-seen})
                    arrival_hist[bar].append((t,'+'.join(missing) or 'none'))
                    if not missing:last_arrivals[bar_name(bar,c)+'/'+role(old['rank'],c['mode'])+('/DXA_pending' if d['events'] else '/DXA_done')]+=1
                    if d['sync']:
                        assert d['wid'] not in pending
                        old=active[d['wid']]
                        pending[d['wid']]=dict(start=t,bar=d['bar'],wid=d['wid'],block=old['block'],rank=old['rank'])
                elif body.startswith('COST_WAIT'):
                    if d['wait']:
                        assert d['wid'] not in pending
                        old=active[d['wid']]
                        pending[d['wid']]=dict(start=t,bar=d['bar'],wid=d['wid'],block=old['block'],rank=old['rank'])
                elif body.startswith('COST_RESUME'):
                    q=pending.pop(d['wid']);assert q['bar']==d['bar']
                    waits.append(dict(q,end=t))
                elif body.startswith('COST_STATE'):
                    states[d['bar']].append((t,d['count'],d['events']))
                else:raise RuntimeError(body)
            elif 'core0-lsu mem-req:' in body:
                m=mem.search(body);assert m,body[:120]
                op,mask,addresses,uid=m.groups();wid=int(uid)>>32
                assert wid in active,(wid,t)
                old=active[wid];base=old['lmem'];r=role(old['rank'],c['mode'])
                vals=[int(x,16) for x in re.findall(r'0x[0-9a-f]+',addresses)]
                assert len(vals)==mask.count('1')
                for addr in vals:
                    region=next((n for n in ('A','B','Y') if mm[n]<=addr<mm[n]+mm[n+'_bytes']),None)
                    if region is None:
                        region='SMEM_input' if base<=addr<base+6144 else 'SMEM_C' if base+6144<=addr<base+14336 and c['smem']==14336 else 'other_stack_args'
                    memory[region+'/'+op]+=1;num_addresses[r+'/'+region+'/'+op]+=1
            else:
                m=mathop.search(body);assert m
                old=active[int(m[2])];ident=(old['block'],old['rank'])
                (compute if m[1].startswith('tcu') else activation)[ident].append(t)
    assert not active and not pending
    assert len(lifetime)==c['ctas']*c['cta_warps'] and all('end' in d for d in lifetime.values())
    assert all('group' in g and 'pass' in g for g in gate.values())
    # Reconstruct exact tick-weighted residency over first dispatch to last exit.
    active={};ctas=defaultdict(set);peak=Counter();area=Counter();prev=res_events[0][0]
    rows=[]
    for t,event,wid,d in res_events:
        metrics=dict(live_warps=len(active),started_ctas=sum(bool(v) for v in ctas.values()),
            full_live_ctas=sum(len(v)==c['cta_warps'] for v in ctas.values()),
            full_compute_ctas=sum(sum(active[w]['rank']<4 for w in v)==4 for v in ctas.values()))
        for n,v in metrics.items():area[n]+=v*(t-prev)
        if event=='dispatch':active[wid]=d;ctas[d['block']].add(wid)
        else:active.pop(wid);ctas[d['block']].remove(wid)
        now=dict(live_warps=len(active),started_ctas=sum(bool(v) for v in ctas.values()),full_live_ctas=sum(len(v)==c['cta_warps'] for v in ctas.values()),
            full_compute_ctas=sum(sum(active[w]['rank']<4 for w in v)==4 for v in ctas.values()))
        for n,v in now.items():peak[n]=max(peak[n],v)
        rows.append(dict(tick=t,event=event,wid=wid,block=d['block'],rank=d['rank'],**now));prev=t
    window=res_events[-1][0]-res_events[0][0]
    # Classify suspended warp-time using actual barrier state. These are warp-cycles,
    # not kernel elapsed cycles and must not be added to gate/scoreboard counters.
    for q in waits:
        name=bar_name(q['bar'],c)+'/'+role(q['rank'],c['mode'])
        wait_groups[name]+=q['end']-q['start']
        hist=states[q['bar']];ticks=[x[0] for x in hist]
        i=bisect_right(ticks,q['start'])-1;assert i>=0
        cur=q['start']
        while cur<q['end']:
            stop=min(q['end'],hist[i+1][0] if i+1<len(hist) else q['end'])
            count,events=hist[i][1:]
            reason='participants_and_DXA' if count and events else 'participants' if count else 'DXA' if events else 'neither'
            dependency[bar_name(q['bar'],c)+'/'+reason]+=stop-cur
            if count:
                ah=arrival_hist[q['bar']];ai=bisect_right([x[0] for x in ah],cur)-1
                assert ai>=0
                missing_wait[bar_name(q['bar'],c)+'/'+ah[ai][1]]+=stop-cur
            cur=stop;i+=1
    assert sum(dependency.values())==sum(wait_groups.values())
    # Verify identical useful work and the intended roundtrip payload.
    tiles=(c['m']//64)*(c['n']//32);per_comp=(c['k']//16)*32
    assert sum(map(len,compute.values()))==tiles*4*per_comp
    assert sum(map(len,activation.values()))==tiles*64
    assert transfers==tiles*(c['k']//16)*2
    assert memory['Y/ST']==c['m']*c['n']
    assert sum(n for name,n in memory.items() if name.endswith('/LD'))==c['core_counters'][str(0xB1C)]
    assert sum(n for name,n in memory.items() if name.endswith('/ST'))==c['core_counters'][str(0xB1E)]
    want_C=c['m']*c['n'] if c['mode'] in (4,8,9) else 0
    assert memory['SMEM_C/LD']==want_C and memory['SMEM_C/ST']==want_C,(memory,c)
    work=[];overlap=0;same_tick=0
    actr=list(range(5,5+c['aw'])) if c['mode'] in (4,8) else list(range(4))
    for block in range(c['ctas']):
        groups=[]
        count=1+(tiles-1-block)//c['ctas']
        for ordinal in range(count):
            co=[t for rank in range(4) for t in compute[block,rank][ordinal*per_comp:(ordinal+1)*per_comp]]
            pa=64//len(actr)
            ac=[t for rank in actr for t in activation[block,rank][ordinal*pa:(ordinal+1)*pa]]
            assert len(co)==4*per_comp and len(ac)==64
            if c['mode'] in (4,8):assert min(ac)>max(co)
            groups.append((co,ac))
            work.append(dict(block=block,ordinal=ordinal,compute_start=min(co),compute_end=max(co),activation_start=min(ac),activation_end=max(ac)))
        for (_,ac),(nco,_) in zip(groups,groups[1:]):
            overlap+=sum(min(nco)<=t<=max(nco) for t in ac);same_tick+=len(set(ac)&set(nco))
    if c['mode']==8:assert overlap==0 and same_tick==0
    stem=f"w{c['hardware_warps']}_g{c['ctas']}_v{c['mode']}_a{c['aw']}"
    write_csv(HERE/(stem+'_residency.csv'),rows);write_csv(HERE/(stem+'_activity.csv'),work)
    write_csv(HERE/(stem+'_waits.csv'),waits)
    byrole=Counter()
    for g in gate.values():byrole[g['group'].split('/')[1]]+=len(g['blocks'])
    return dict(gate_blocked_unique_ticks=len(gate_ticks),gate_block_events=gate_events,
        gate_blocked_instructions=sum(bool(g['blocks']) for g in gate.values()),bar_instructions=len(gate),
        gate_max_instruction_delay=max((g['pass']-min(g['blocks']) for g in gate.values() if g['blocks']),default=0),
        gate_blocks_by_role=dict(byrole),gate_blocks_by_bar_role=dict(gate_groups),bar_instructions_by_group=dict(gate_instructions),
        suspended_warp_cycles=dict(wait_groups),suspended_warp_cycles_by_dependency=dict(dependency),
        participant_wait_warp_cycles_by_missing_role=dict(missing_wait),last_participant_arrivals=dict(last_arrivals),
        residency=dict(window_ticks=window,first_dispatch=res_events[0][0],last_exit=res_events[-1][0],peak=dict(peak),average={n:area[n]/window for n in area}),
        lane_memory_operations=dict(memory),lane_memory_operations_by_role=dict(num_addresses),
        wgmma_events=sum(map(len,compute.values())),relu_events=sum(map(len,activation.values())),dxa_transfers=transfers,
        activation_in_next_compute_window=overlap,same_tick_with_next_compute=same_tick,
        ctas=[dict(block=b,start=min(v['start'] for (bb,_),v in lifetime.items() if bb==b),end=max(v['end'] for (bb,_),v in lifetime.items() if bb==b)) for b in range(c['ctas'])])

def main():
    runs={(w,p):json.loads(result_path(f'cost_w{w}_{p}_results.json').read_text()) for w in (8,16) for p in ('normal','validate','trace')}
    assert all(r['status']=='complete' for r in runs.values())
    assert len({r['binary_sha256'] for r in runs.values()})==1
    assert len({json.dumps(r['source_hashes'],sort_keys=True) for r in runs.values()})==1
    originals={(w,key(c)):c for (w,p),r in runs.items() if p=='normal' for c in r['cases']}
    # Old baselines and the original three-stage variants must remain exact.
    for w in (8,16):
        prior=json.loads(result_path(f'multicta_w{w}_matrix_results.json').read_text())['cases']
        for c in runs[w,'normal']['cases']:
            if c['mode'] in (2,3,4) and not c['reserve']:
                old=next(x for x in prior if (x['m'],x['n'],x['k'],x['ctas'],x['mode'],x['activation_warps'])==(c['m'],c['n'],c['k'],c['ctas'],c['mode'],c['aw']))
                assert c['chain']==old['chain'] and c['output_hash']==old['output_hash']
    hashes=defaultdict(set);measurements=[];traces=[]
    for (w,p),r in runs.items():
        for c in r['cases']:
            assert c['passed'] and c['hardware_warps']==w and len(c['core_counters'])==18
            hashes[c['m'],c['n'],c['k'],c['seed']].add(c['output_hash'])
            old=originals[w,key(dict(c,seed=1))]
            assert c['chain']==old['chain'] and c['core_counters']==old['core_counters']
            measurements.append(dict(hardware_warps=w,phase=p,**{n:c[n] for n in FIELDS},label=label(c),
                **c['chain'],smem=c['smem'],output_hash=c['output_hash']))
            if p=='trace':
                print(f"parse w{w} g{c['ctas']} mode{c['mode']} A{c['aw']}",flush=True)
                traces.append(dict(hardware_warps=w,**{n:c[n] for n in FIELDS},label=label(c),cycles=c['chain']['cycles'],
                    trace=c['trace'],detail=trace(c)))
    assert all(len(v)==1 for v in hashes.values())
    smoke=json.loads(result_path('cost_w8_smoke_results.json').read_text())
    assert smoke['status']=='complete' and len(smoke['cases'])==3 and all(c['passed'] for c in smoke['cases'])
    assert smoke['binary_sha256']==next(iter(runs.values()))['binary_sha256']
    result=dict(runs=sum(len(r['cases']) for r in runs.values())+len(smoke['cases']),formal_runs=sum(len(r['cases']) for r in runs.values()),
        binary_sha256=next(iter(runs.values()))['binary_sha256'],observer_counter_equality=True,
        source_hashes=next(iter(runs.values()))['source_hashes'],
        phases={f'w{w}_{p}':len(r['cases']) for (w,p),r in runs.items()},traces=traces,
        provenance={f'w{w}_{p}':r['logs'] for (w,p),r in runs.items()})
    capacity_runs={(w,p):json.loads(result_path(f'cost_capacity_w{w}_{p}_results.json').read_text()) for w in (8,16) for p in ('normal','validate','trace')}
    assert all(r['status']=='complete' for r in capacity_runs.values())
    assert len({r['binary_sha256'] for r in capacity_runs.values()})==1
    capacity_originals={(w,key(c)):c for (w,p),r in capacity_runs.items() if p=='normal' for c in r['cases']}
    capacity_traces=[]
    for (w,p),r in capacity_runs.items():
        for c in r['cases']:
            assert c['passed'] and c['smem']==6144
            hashes[c['m'],c['n'],c['k'],c['seed']].add(c['output_hash'])
            old=capacity_originals[w,key(dict(c,seed=1))]
            assert c['chain']==old['chain'] and c['core_counters']==old['core_counters']
            if c['mode']==2:
                old=originals[w,key(c)]
                assert c['chain']==old['chain'] and c['core_counters']==old['core_counters']
            measurements.append(dict(hardware_warps=w,phase='capacity_'+p,**{n:c[n] for n in FIELDS},label=label(c),**c['chain'],smem=c['smem'],output_hash=c['output_hash']))
            if p=='trace':
                print(f"parse capacity w{w} mode{c['mode']}",flush=True)
                capacity_traces.append(dict(hardware_warps=w,**{n:c[n] for n in FIELDS},label=label(c),cycles=c['chain']['cycles'],trace=c['trace'],detail=trace(c)))
    assert all(len(v)==1 for v in hashes.values())
    result['capacity']=dict(runs=sum(len(r['cases']) for r in capacity_runs.values()),
        binary_sha256=next(iter(capacity_runs.values()))['binary_sha256'],traces=capacity_traces,
        source_hashes=next(iter(capacity_runs.values()))['source_hashes'],
        provenance={f'w{w}_{p}':r['logs'] for (w,p),r in capacity_runs.items()})
    result['runs']+=result['capacity']['runs'];result['formal_runs']+=result['capacity']['runs']
    write_csv(HERE/'measurements.csv',measurements)
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(runs=result['runs'],observer_counter_equality=True),indent=2))

if __name__=='__main__':main()
