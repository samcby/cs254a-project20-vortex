#!/usr/bin/env python3
"""Reconstruct live CTA warp membership from existing scheduler trace events."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import re


def number(token):
    return int(token, 16 if any(ch in token.lower() for ch in 'abcdef') else 10)


def parse(path, hardware_warps, cta_warps, expected_ctas):
    opened = gzip.open(path, 'rt', errors='replace') if str(path).endswith('.gz') else path.open(errors='replace')
    dispatch = re.compile(r'dispatch CTA warp: cid=(\w+), wid=(\w+), cta_id=(\w+), rank=(\w+)/(\w+).*blockIdx=\((\w+),(\w+)\)')
    exit_ = re.compile(r'core(\d+).*warp-state: wid=(\w+), tmask=([01]+)\s*$')
    timestamp = re.compile(r'^TRACE\s+(\d+):')
    active = {}
    ctas = {}
    rows = []
    last_cycle = None
    peaks = dict(live_warps=0, started_live_ctas=0, fully_admitted_live_ctas=0, all_warps_live_ctas=0)
    dispatches = 0
    exits = 0
    for line in opened:
        m = timestamp.match(line)
        if m:
            last_cycle = int(m[1])
        m = dispatch.search(line)
        kind = None
        if m:
            cid, wid, slot, rank, size, x, y = map(number, m.groups())
            if cid != 0 or size != cta_warps or wid >= hardware_warps or wid in active:
                raise RuntimeError('Invalid warp dispatch: ' + line)
            key = (x, y)
            info = ctas.setdefault(key, dict(admitted=set(), live=set(), slot=slot, size=size))
            if rank in info['admitted']:
                raise RuntimeError('Duplicate CTA rank')
            info['admitted'].add(rank)
            info['live'].add(wid)
            active[wid] = key
            dispatches += 1
            kind = 'dispatch'
        elif 'dispatch CTA warp:' in line:
            raise RuntimeError('Unparsed dispatch: ' + line)
        else:
            m = exit_.search(line)
            if m and int(m[3], 2) == 0:
                cid, wid = int(m[1]), number(m[2])
                if cid != 0 or wid not in active:
                    raise RuntimeError('Unmatched warp termination: ' + line)
                key = active.pop(wid)
                info = ctas[key]
                info['live'].remove(wid)
                slot, rank = info['slot'], None
                exits += 1
                kind = 'exit'
        if kind:
            counts = dict(live_warps=len(active),
                started_live_ctas=sum(bool(c['live']) for c in ctas.values()),
                fully_admitted_live_ctas=sum(bool(c['live']) and len(c['admitted']) == cta_warps for c in ctas.values()),
                all_warps_live_ctas=sum(len(c['live']) == cta_warps for c in ctas.values()))
            for name, value in counts.items():
                peaks[name] = max(peaks[name], value)
            rows.append(dict(event=len(rows), preceding_trace_cycle=last_cycle, kind=kind,
                             wid=wid, cta_x=key[0], cta_y=key[1], slot=slot, rank=rank, **counts))
    opened.close()
    if active or len(ctas) != expected_ctas or dispatches != expected_ctas * cta_warps or exits != dispatches:
        raise RuntimeError('Incomplete CTA lifetime reconstruction')
    if any(info['admitted'] != set(range(cta_warps)) for info in ctas.values()):
        raise RuntimeError('Missing CTA ranks')
    return dict(hardware_warps=hardware_warps, cta_warps=cta_warps,
                ctas=len(ctas), warp_dispatches=dispatches, warp_exits=exits,
                peak=peaks, note='Peak membership is reconstructed in event order; dispatch timestamps lack an exact cycle.'), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path', type=Path)
    p.add_argument('--warps', type=int, required=True)
    p.add_argument('--cta-warps', type=int, required=True)
    p.add_argument('--ctas', type=int, required=True)
    p.add_argument('--csv', type=Path)
    args = p.parse_args()
    result, rows = parse(args.path, args.warps, args.cta_warps, args.ctas)
    if args.csv:
        with args.csv.open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
