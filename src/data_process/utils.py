import json
import random
from collections import Counter

def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)
    


def split_by_video(labels, labeled_ratio, seed):
    vids = [v['video'] for v in labels]
    rnd = random.Random(seed); rnd.shuffle(vids)
    n_lab = max(1, int(len(vids) * labeled_ratio))
    lab_set = set(vids[:n_lab])
    labeled   = [v for v in labels if v['video'] in lab_set]
    unlabeled = [v for v in labels if v['video'] not in lab_set]
    return labeled, unlabeled


def _print_info_helper(src_file, labels):
    num_frames = sum([x['num_frames'] for x in labels])
    num_events = sum([len(x['events']) for x in labels])
    print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
        src_file, len(labels), num_frames,
        num_events / num_frames * 100))
    print('Class distribution:')
    counter = Counter()
    for e in labels:
        for ev in e['events']:
            counter[ev['label']] += 1
    print(counter)
    print('---')
    print(f'In total {num_events} events across {len(labels)} videos ({num_frames} frames).')
