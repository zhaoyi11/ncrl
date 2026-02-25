import argparse
import io
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn.functional as F
import faiss
from omegaconf import OmegaConf 
import dreamer.nets as nets

# The model and data should be structured as follows:
# data_basedir
#   |-- dmcontrol
#   |   |-- domain1
#   |   |   |-- data1.npz
#   |   |   |-- data2.npz
#   |   |-- domain2
#   |   |   |-- data1.npz
#   |   |   |-- data2.npz
#   |-- metaworld
#   |   |-- task1
#   |   |   |-- data1.npz
#   |   |   |-- data2.npz
#   |   |-- task2
#   |   |   |-- data1.npz
#   |   |   |-- data2.npz

# The model and data should be structured as follows:
# model_basedir
#   |-- dmcontrol.pt
#   |-- metaworld.pt


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare KV dataset with encoder features.')
    parser.add_argument(
        '--model-basedir',
        type=Path,
        default=Path('/data/mpt_models'),
        help='Directory containing benchmark .pt encoder models (e.g. dmcontrol.pt, metaworld.pt)',
    )
    parser.add_argument(
        '--data-basedir',
        type=Path,
        default=Path('/data/ncrl_datasets'),
        help='Root directory of npz datasets (dmcontrol/, metaworld/ subdirs)',
    )
    parser.add_argument(
        '--path-to-save',
        type=Path,
        default=Path('/home/zhaoy13/rl/ncrl/dataset_db.npz'),
        help='Output path for the saved dataset DB .npz file',
    )
    return parser.parse_args()


def main(model_basedir: Path, data_basedir: Path, path_to_save: Path) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get nerual feature of the initial frame for each trajectory
    # initialize the encoder with the same configuration as the encoder in the world model
    cfg = OmegaConf.create({
        "cnn_depth": 96,
        "use_norm": True,
        "cnn_kernels": [4, 4, 4, 4],
        "act": "SiLU",
    })
    obs_shape = (3, 64, 64)
    encoder = nets.Encoder(obs_shape, **cfg)

    db_to_save = {}

    for benchmark in ['dmcontrol', 'metaworld']:
        data_path = data_basedir / benchmark
        filename = list(data_path.rglob('*.npz'))

        # load model
        if not str(model_basedir).endswith('.pt'):
            model_path = model_basedir / f'{benchmark}.pt'
        else:
            model_path = model_basedir
        param_dict = torch.load(model_path, weights_only=False)
        encoder.load_state_dict(param_dict['wm']['encoder'])
        encoder.to(device)
        encoder.eval()

        # collect the first frame of each trajectory
        first_frames = []
        for fn in tqdm(filename):
            data = np.load(fn)
            frame = data['observation'][0]
            first_frames.append(frame)

        first_frames = torch.from_numpy(np.stack(first_frames)).float() / 255. - 0.5

        # get feature
        first_frames = first_frames.to(device)
        with torch.no_grad():
            for i in range(0, first_frames.size(0), 100):
                feat = encoder(first_frames[i:i+100]) # preprocess 100 frames at a time to save memory
                feat = feat.cpu().numpy()
                if i == 0:
                    feats = feat
                else:
                    feats = np.concatenate((feats, feat), axis=0)
        if benchmark == 'dmcontrol':
            _benchmark = 'dmc'
        elif benchmark == 'metaworld':
            _benchmark = 'mw'
        else:
            raise ValueError(f'Unknown benchmark: {benchmark}')
        db_to_save[_benchmark] = {
            'filename': np.array(filename),
            'feature': feats,
        }

    # save the db
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **db_to_save)
        bs.seek(0)
        with path_to_save.open('wb') as f:
            f.write(bs.read())


if __name__ == '__main__':
    args = parse_args()
    main(args.model_basedir, args.data_basedir, args.path_to_save)