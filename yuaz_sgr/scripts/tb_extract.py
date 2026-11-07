"""从 TensorBoard 事件里导出验证集样音（训练结束也能事后试听各检查点）

用法:
    python scripts/tb_extract.py <tf_logdir> <out_dir> [step ...]
例:
    python scripts/tb_extract.py ../tf-logs/final ../tb_audio 105000 120000
"""
import os
import sys
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log_dir')
    ap.add_argument('out_dir')
    ap.add_argument('steps', nargs='*', type=int)
    ap.add_argument('--tags', default='val/syn', help='逗号分隔，如 val/syn,val/real')
    args = ap.parse_args()

    acc = EventAccumulator(args.log_dir, size_guidance={'audio': 0, 'images': 0, 'scalars': 0})
    acc.Reload()
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    wanted = set(args.steps)

    for tag in tags:
        if tag not in acc.Tags().get('audio', []):
            print('missing tag', tag)
            continue
        for ev in acc.Audio(tag):
            if wanted and ev.step not in wanted:
                continue
            name = tag.replace('/', '_')
            path = os.path.join(args.out_dir, f'{name}_{ev.step}.wav')
            with open(path, 'wb') as f:
                f.write(ev.encoded_audio_string)
            print('saved', path)


if __name__ == '__main__':
    main()
