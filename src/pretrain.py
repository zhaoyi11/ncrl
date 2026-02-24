import warnings
from time import time

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import uuid
import time
import random
from pathlib import Path
import hydra
from tqdm import tqdm
import numpy as np
import torch
import wandb
from collections import namedtuple, deque
import omegaconf

import envs
import utils
from replay_buffer import ReplayBufferStorage, make_replay_loader
from dreamer.dreamer import DreamerAgent

warnings.filterwarnings('ignore', category=DeprecationWarning)

torch.backends.cudnn.benchmark = True


class Workspace:
    def __init__(self, cfg):
        # set up workspace
        cfg.ws.run_id = str(uuid.uuid4().hex)
        if cfg.ws.seed == -1: # random sample a seed if ==-1
            cfg.ws.seed = random.randint(0, 1000)

        # set work_dir
        _run_path = Path('logs')/cfg.ws.experiment/f'{cfg.ws.run_id}_{cfg.ws.seed}'
        if cfg.ws.work_basedir is not None:
            self.work_dir = Path(cfg.ws.work_basedir)/_run_path
        else:
            self.work_dir = Path.cwd()/_run_path
        print(f'Workspace: {self.work_dir}')
        
        # set seed
        utils.set_seed_everywhere(cfg.ws.seed)
        self.device = torch.device(cfg.ws.device)
        print(f'Using device: {self.device}')

        # create envs
        self.train_env = envs.make_env(cfg.env, cfg.ws.seed)

        self.obs_shape = obs_shape = self.train_env.obs_space.image.shape
        self.act_dim = act_dim = self.train_env.action_space.shape[0]

        if cfg.ws.use_action_padding:
            padded_act_dim = envs.MAX_ACTION_DIM
        else:
            padded_act_dim = act_dim
        
        # Initialize agent
        self.agent = DreamerAgent(cfg.agent, obs_shape, act_dim, padded_act_dim, device=cfg.ws.device) 
        if cfg.snapshot.load_path is not None:
            snapshot_dir = Path(cfg.snapshot.load_path)
            self.agent.load(snapshot_dir, cfg.snapshot.models_to_load)
            print(f"--- Load snapshot from {snapshot_dir} ---")
 
        # create replay buffer
        assert cfg.buffer.offline_path is not None, "buffer.offline_path must be provided"
        self.replay_storage, self.replay_loader = self._setup_buffer(
                                cfg = cfg.buffer,
                                data_dir=Path(cfg.buffer.offline_path), 
                                obs_shape=obs_shape,
                                max_act_dim=padded_act_dim,
                                batch_size=int(cfg.buffer.batch_size))
        self._replay_iter = None
       
        
        self.cfg = cfg
        # create logger
        if self.cfg.ws.use_wandb and wandb.run is None:
            self._setup_wandb()

        # Globals
        self._start_time = time.time()
        self._global_step = 0
        self._global_episode = 0
        self._global_epoch = 0

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self.global_step,
            total_time=time.time() - self._start_time,
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def _setup_buffer(self, cfg, data_dir, obs_shape, max_act_dim, batch_size):
        # create replay buffer
        data_specs = (envs.specs(shape=obs_shape, dtype=np.uint8, name='observation'),
                    envs.specs(shape=None, dtype=np.float32, name='action'), # action is set to None to allow different action shapes
                    envs.specs(shape=(1,), dtype=np.float32, name='reward'),
                    envs.specs(shape=(1,), dtype=np.float32, name='discount'))

        replay_storage = ReplayBufferStorage(data_dir, data_specs)
        replay_loader = make_replay_loader(
            replay_dir=data_dir,
            data_specs=data_specs,
            max_size=cfg.capacity,
            batch_size=batch_size,
            num_workers=cfg.num_workers,
            nstep=cfg.chunk_length,
            max_act_dim=max_act_dim,
            pad_action=cfg.use_action_padding,
            save_buffer=True,
            )
        return replay_storage, replay_loader    

    def _setup_wandb(self):
        _cfg = self.cfg.ws
        wandb.init(project=_cfg.project_name, entity=_cfg.wandb_entity, 
                name=f'{_cfg.mode}-{_cfg.project_name}-{_cfg.experiment}-{_cfg.run_id}-{str(_cfg.seed)}',
                group='pretrain', 
                tags=[_cfg.project_name, 'pretrain', str(_cfg.seed)],
                config=omegaconf.OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True))

    def train(self):

        for step in tqdm(range(self.cfg.ws.num_pretrain_steps), ncols=80):
            self._global_step = step

            metrics = self.agent.update(
                online_data=None, 
                offline_data=next(self.replay_iter), 
                step=self.global_step)[1]
            
            if self.cfg.ws.use_wandb and self.global_step % self.cfg.ws.log_every_steps == 0:
                metrics.update(self.common_metrics())
                metrics = {f'train/{k}': v for k, v in metrics.items()}
                print(metrics)
                wandb.log(metrics, step=self.global_step)

            # log reconstrunction
            if self.cfg.ws.use_wandb and self.global_step > 0 and self.global_step % self.cfg.ws.recon_every_steps == 0:
                videos = self.agent.report(next(self.replay_iter))

                for k, v in videos.items():
                    v = np.uint8(v.cpu() * 255)
                    wandb.log({k: wandb.Video(v, fps=15, format="mp4")}, step=self.global_step)
                
            # save model
            if self.cfg.snapshot.save_snapshot and self.global_step > 0 and self.global_step % self.cfg.ws.save_every_steps == 0:
                self.work_dir.mkdir(parents=True, exist_ok=True)
                self.agent.save_model(self.work_dir/f'snapshot_{self.global_step}.pt')
                print(f'------- Save model at epoch {self.global_step} ------')

  
        if self.cfg.snapshot.save_snapshot:
            self.agent.save_model(self.work_dir/'snapshot.pt')
        

@hydra.main(config_path='../config', config_name='pretrain')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()
