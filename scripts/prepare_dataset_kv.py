import io
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn.functional as F
import faiss
from omegaconf import OmegaConf 
import dreamer.nets_ as nets
# file name -- a list of strings
data_basedir = Path('/scratch/project_462000782/icml2025_dataset/all')
path_to_save = Path('/scratch/project_462000782/icml2025_dataset/dataset_db.npz')
model_basedir = Path('/flash/project_462000782/cont_pretrain/checkpoints')
# data_basedir = Path('/data/masrl/datasets/exploration')
# path_to_save = Path('/home/zhaoy13/rl/cont_pretrain/dataset_db.npz')
# model_basedir = Path('/home/zhaoy13/rl/cont_pretrain/logs/pretrain/dmc-walker-walk/65b97f21b0be47a9bc3427c19cebbe1a_42/snapshot_200000.pt')

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
    param_dict = torch.load(model_basedir)
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
            feat = encoder(first_frames[i:i+100])
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