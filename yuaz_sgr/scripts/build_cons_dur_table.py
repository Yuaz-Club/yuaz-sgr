"""从预处理数据统计「音素占音符时长」的比例表（供推理端使用） 用法: python scripts/build_cons_dur_table"""
import os
import glob
import json
import argparse

import numpy as np

def build(data_dir):
    rows = {}
    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if not files:
        raise SystemExit(f'no npz under {data_dir}')
    for f in files:
        d = np.load(f, allow_pickle=True)
        ids, durs, nl = d['phonemes'], d['durations'], d['note_lengths']
        if len(ids) != len(durs) or len(nl) == 0:
            continue
        idx = 0
        for n_len in nl:
            n_len = int(n_len)
            acc, phs = 0, []
            while idx < len(durs) and acc < n_len:
                phs.append((str(ids[idx]), int(durs[idx])))
                acc += int(durs[idx])
                idx += 1
            for j, (name, dur) in enumerate(phs):
                role = 'vowel' if (len(phs) == 1 or j > 0) else 'consonant'
                rows.setdefault(name, {'ratio': [], 'dur': [], 'role': []})
                rows[name]['ratio'].append(dur / max(1, n_len))
                rows[name]['dur'].append(dur)
                rows[name]['role'].append(role)

    out = {}
    for name, v in rows.items():
        r = np.array(v['ratio'])
        du = np.array(v['dur'])
        role = 'consonant' if v['role'].count('consonant') > len(r) / 2 else 'vowel'
        out[name] = {
            'role': role,
            'n': int(len(r)),
            'ratio_p50': float(np.median(r)),
            'ratio_p90': float(np.percentile(r, 90)),
            'ratio_max': float(r.max()),
            'frames_p50': float(np.median(du)),
            'frames_p90': float(np.percentile(du, 90)),
            'frames_max': float(du.max()),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='预处理后的 train/ 或 val/ 目录')
    ap.add_argument('--output', required=True, help='输出 json 路径')
    args = ap.parse_args()

    table = build(args.data_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False, indent=1)

    print('| 音素 | 角色 | 次数 | 占比P50 | 占比P90 | 帧数P50 | 帧数P90 |')
    print('|---|---|---|---|---|---|---|')
    for name in sorted(table):
        v = table[name]
        print('| %s | %s | %d | %.3f | %.3f | %.0f | %.0f |' % (
            name, v['role'], v['n'], v['ratio_p50'], v['ratio_p90'],
            v['frames_p50'], v['frames_p90']))
    print(f'\n写入 {args.output}')

if __name__ == '__main__':
    main()
