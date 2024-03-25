import os, sys
import tqdm
import numpy as np
import pickle

root_dir = sys.argv[1]  # path/to/your/submission
dst_path = sys.argv[2]  # path/to/your/submission.pkl

# Update method / authors / email / institution / country to your
#  Corresponding information.
sv_pkl = dict(
    method=None,
    team=None,
    authors=None,
    email=None,
    institution=None,
    country=None,
)

frame_prefix = 'frame'
frame_names = [f'{frame_prefix}_0',
               f'{frame_prefix}_1',
               f'{frame_prefix}_2',
               f'{frame_prefix}_3',
               f'{frame_prefix}_4',
               f'{frame_prefix}_5',
               f'{frame_prefix}_6',]

results = dict()

for fname in tqdm.tqdm(os.listdir(root_dir)):
    if not '.txt' in fname: continue
    res = []
    dt_path = os.path.join(root_dir, fname)
    with open(dt_path, 'r') as f:
        for line in f.readlines():
            cur_res = line.strip('\n').split(' ')
            res.append(cur_res)
    res = np.array(res).astype(np.float16)  # n, 1
    sample_idx, f_idx = fname.strip('.txt').split('_')
    assert int(f_idx) >= 1 and int(f_idx) <= 6, 'Future prediction should in 1-6 frames.'
    if sample_idx not in results:
        results[sample_idx] = dict()
    results[sample_idx][frame_names[int(f_idx)]] = res

sv_pkl['results'] = results

with open(dst_path, 'wb') as f:
    pickle.dump(sv_pkl, f)