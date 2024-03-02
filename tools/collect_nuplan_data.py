import pickle
import os, sys


def save_infos(files, split):
    infos = []
    for file in files:
        with open(file, 'rb') as f:
            infos.extend(pickle.load(f))

    with open(f'data/openscene-v1.1/openscene_{split}.pkl', 'wb') as f:
        pickle.dump(infos, f)


if __name__ == '__main__':
    split = sys.argv[1]
    paths = os.listdir(f'data/openscene-v1.1/meta_datas/{split}')
    paths = [
        os.path.join(f'data/openscene-v1.1/meta_datas/{split}', each)
        for each in paths if each.endswith('.pkl')]

    if split == 'test':
        save_infos(paths, 'test')
    else:
        train_paths = paths[:int(len(paths) * 0.85)]
        save_infos(train_paths, f'{split}_train')

        val_paths = paths[int(len(paths) * 0.85):]
        save_infos(val_paths, f'{split}_val')