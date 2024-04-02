import sys, os
import mmcv

train_pkl = 'data/nuscenes/nuscenes_infos_temporal_train.pkl'
train_pkl = mmcv.load(train_pkl)

test_pkl = 'data/nuscenes/nuscenes_infos_temporal_test.pkl'
test_pkl = mmcv.load(test_pkl)

dst_pkl = dict(
    metadata=train_pkl['metadata'],
    infos=train_pkl['infos'] + test_pkl['infos']
)
dst_fname = 'data/nuscenes/nuscenes_infos_temporal_traintest.pkl'
mmcv.dump(dst_pkl, dst_fname)