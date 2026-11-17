#!/usr/bin/env python3

import os
import argparse
import re

from utils.io import load_gz_json, load_json
from losses_metrics.metrics import compute_mAPs_advanced
from losses_metrics.eval import non_maximum_supression


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', help='Path to predictions or model dirs')

    parser.add_argument('gt_file', help='Path to ground truth file')

    parser.add_argument('--nms_window', type=int, default=1)

    parser.add_argument('-t', '--tolerances', type=int, nargs='+')

    return parser.parse_args()


def get_pred_file(pred_dir, split):
    regex = re.compile(r'pred-{}\.(\d+)\.recall\.json\.gz'.format(split))
    candidates = []
    for file_name in os.listdir(pred_dir):
        m = regex.match(file_name)
        if m:
            candidates.append((
                os.path.join(pred_dir, file_name), int(m.group(1))))
    if len(candidates) > 0:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]
    raise FileNotFoundError('No suitable prediction file!')


def main(pred_file, gt_file, nms_window, tolerances):

    pred = (load_gz_json if pred_file.endswith('.gz') else load_json)(pred_file)

    truth = load_json(gt_file)

    kwargs = {}
    if tolerances is not None:
        kwargs['tolerances'] = tolerances

    print('\n=== Results on (w/o NMS) ===')
    no_nms_result = compute_mAPs_advanced(truth, pred, **kwargs)

    pred = non_maximum_supression(pred, nms_window)
    print('\n=== Results on (w/ NMS) ===')
    nms_result = compute_mAPs_advanced(truth, pred, **kwargs)
    return no_nms_result, nms_result


if __name__ == '__main__':
    main(**vars(get_args()))