import warnings
from time import time

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import time
import random
from pathlib import Path
import hydra
from tqdm import tqdm
import numpy as np
import torch
import wandb
from collections import namedtuple
import omegaconf
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from einops import rearrange

import envs
import utils
from replay_buffer import ReplayBufferStorage, make_replay_loader
from dreamer.dreamer import DreamerAgent
import dreamer.nets as nets

warnings.filterwarnings('ignore', category=DeprecationWarning)

torch.backends.cudnn.benchmark = True


class BCAgent(nn.Module):
    def __init__(self, obs_shape, act_dim, **kwargs):
        super().__init__()
        self._obs_horizon = kwargs.get('obs_horizon', 3)
        self._encoder = nets.Encoder(obs_shape)
        # Computing embed dim
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + obs_shape)
            outs = self._encoder(dummy_input)
            embed_dim = outs.shape[-1]
        self._actor = nn.Sequential(nn.Linear(embed_dim, 512), nn.LayerNorm(512), nn.ELU(),
                                    nn.Linear(512, 512), nn.LayerNorm(512), nn.ELU(),
                                    nn.Linear(512, act_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.aug = utils.RandomShiftsAug(3)

    @property
    def device(self):
        return next(self.parameters()).device

    def act(self, obs):
        obs = obs/255. - 0.5
        action = self._actor(self._encoder(obs[None]))
        return action
    
    def preprocess_data(self, data):
        #|o|o|o|     observations: 3
        #| | |p|     actions predicted: 8

        ret = {}
        obs = data['observation'][:, :self._obs_horizon] / 255. - 0.5
        obs = obs.to(device=self.device, dtype=torch.float32)
        b, t, c, h, w = obs.shape
        # reshape and augment observations
        aug_obs = self.aug(rearrange(obs, 'b t c h w -> b (t c) h w'))

        ret['observation'] = aug_obs
        ret['action'] = data['action'][:, self._obs_horizon:self._obs_horizon+1].to(device=self.device, dtype=torch.float32) 
        ret['mask'] = data['action_mask'][:, self._obs_horizon:self._obs_horizon+1].to(device=self.device, dtype=torch.float32)
        # remove the time dim of action
        ret['action'] = ret['action'].squeeze(-2)
        ret['mask'] = ret['mask'].squeeze(-2)
        return ret

    def update(self, data):
        data = self.preprocess_data(data)
        action_pred = self._actor(self._encoder(data['observation']))
        losses = F.mse_loss(action_pred, data['action'], reduction='none')
        
        loss = (losses * data['mask']).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'bc_loss': loss.item()}

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)



# class Workspace:
#     def __init__(self, cfg):
#         self.multitask = cfg.ws.multitask  
        
#         if self.multitask \
#             or (isinstance(cfg.env.task, str) and cfg.env.task.startswith('mt-')) \
#             or (not isinstance(cfg.env.task, str) and len(cfg.env.task) > 1):
#             assert self.multitask == True, "Multitask flag must be True if multiple tasks are provided."
        
#         # set up workspace
#         cfg.ws.run_id = int(time.time()) + random.randint(0, 10000)
#         if cfg.ws.seed == -1: # random sample a seed if ==-1
#             cfg.ws.seed = random.randint(0, 1000)

#         # set work_dir
#         _env_name = cfg.env.task if not self.multitask else "multitask"
#         _run_path = Path('logs')/cfg.ws.experiment/f'{_env_name}'/f'{cfg.ws.run_id}_{cfg.ws.seed}'
#         if cfg.ws.work_basedir is not None:
#             self.work_dir = Path(cfg.ws.work_basedir)/_run_path
#         else:
#             self.work_dir = Path.cwd()/_run_path
#         print(f'Workspace: {self.work_dir}')
        
#         # set seed
#         utils.set_seed_everywhere(cfg.ws.seed)
#         self.device = torch.device(cfg.ws.device)
#         print(f'Using device: {self.device}')

#         # create envs
#         self.train_env_dict = envs.make_env_dict(cfg.env, cfg.ws.seed)
#         self.eval_env_dict = envs.make_env_dict(cfg.env, cfg.ws.seed+10)

#         obs_shape = list(self.train_env_dict.values())[0].obs_space.image.shape
#         act_dim = list(self.train_env_dict.values())[0].action_space.shape[0] if not self.multitask else max(list(envs.TASK_ACT_DIM.values())) # get the max action dim over all envs
  
#         # Initialize agent
#         self.agent = BCAgent((9, 64, 64), act_dim).to(self.device)

#         if cfg.snapshot.load_path is not None:
#             _path = cfg.snapshot.load_path
#             snapshot_dir = Path(_path.base_dir)/_path.mid_dir/_path.snapshot_name
#             self.agent.load(snapshot_dir)
#             print(f'------- Load snapshot from {snapshot_dir} ------')
        
#         # create replay buffer
#         _path = cfg.buffer.offline_path
#         data_dir = Path(_path.base_dir)/_path.mid_dir/_path.buffer_name
#         self.offline_storage, self.offline_loader = self._setup_buffer(
#                             cfg=cfg.buffer,
#                             data_dir=data_dir,
#                             obs_shape=obs_shape,
#                             max_act_dim=act_dim,
#                             batch_size=cfg.buffer.batch_size)
#         self._offline_iter = None
#         assert len(self.offline_storage) > 0, "Offline data size must be greater than 0."
#         print(f'------- Load offline dataset from {data_dir} with #{len(self.offline_storage)} transitions. ------')     
    
#         self.cfg = cfg
#         # create logger
#         if wandb.run is None:
#             self._setup_wandb()

#         # Globals
#         self._start_time = time.time()
#         self._global_step = 0
#         self._global_episode = 0
#         self._global_epoch = 0

#     def common_metrics(self):
#         """Return a dictionary of current metrics."""
#         return dict(
#             step=self.global_step,
#             frame=self.global_frame,
#             epoch=self.global_epoch,
#             episode=self.global_episode,
#             total_time=time.time() - self._start_time,
#         )

#     @property
#     def global_step(self):
#         return self._global_step

#     @property
#     def global_episode(self):
#         return self._global_episode

#     @property
#     def global_epoch(self):
#         return self._global_epoch

#     @property
#     def global_frame(self):
#         return self.global_step * self.cfg.env.action_repeat
    
#     @property
#     def offline_iter(self):
#         if self._offline_iter is None:
#             self._offline_iter = iter(self.offline_loader)
#         return self._offline_iter
    
#     def _setup_buffer(self, cfg, data_dir, obs_shape, max_act_dim, batch_size):
#         # create replay buffer
#         data_specs = (envs.specs(shape=obs_shape, dtype=np.uint8, name='observation'),
#                     envs.specs(shape=None, dtype=np.float32, name='action'), # action is set to None to allow different action shapes
#                     envs.specs(shape=(1,), dtype=np.float32, name='reward'),
#                     envs.specs(shape=(1,), dtype=np.float32, name='discount'))

#         replay_storage = ReplayBufferStorage(data_dir, data_specs)
#         replay_loader = make_replay_loader(
#             replay_dir=data_dir,
#             data_specs=data_specs,
#             max_size=cfg.capacity,
#             batch_size=batch_size,
#             num_workers=cfg.num_workers,
#             # nstep=cfg.chunk_length,
#             nstep=4,
#             max_act_dim=max_act_dim,
#             pad_action=self.multitask)
#         return replay_storage, replay_loader    

#     def _setup_wandb(self):
#         _cfg = self.cfg.ws
#         wandb.init(project=_cfg.project_name, 
#                                     name=f'{_cfg.mode}-{self.cfg.env.task}-{_cfg.project_name}-{_cfg.experiment}-{_cfg.run_id}-{str(_cfg.seed)}',
#                                     group=f'{self.cfg.env.task}-{_cfg.project_name}', 
#                                     tags=[_cfg.project_name, f'{self.cfg.env.task}', _cfg.experiment, str(_cfg.seed)],
#                                     config=omegaconf.OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True))

#     @torch.no_grad()
#     def eval(self):
#         self.agent.eval()
#         for env_name, env in self.eval_env_dict.items():
#             timestep = env.reset()
#             obs_buffer = deque([timestep.observation] * 3, maxlen=3)
#             total_reward = 0
#             video = [timestep.observation]
#             while not timestep.done:
#                 action = self.agent.act(torch.from_numpy(
#                                         np.concatenate(list(obs_buffer))).to(self.device))
#                 timestep = env.step(action)
#                 obs_buffer.append(timestep.observation)
#                 video.append(timestep.observation)
#                 total_reward += timestep.reward
#             print(f'Env: {env_name}, Total Reward: {total_reward}')
        
#         self.agent.train()
#         return {'eval/total_reward': total_reward,
#                 'eval/video': np.stack(video)}

#     def train(self):
        
#         for ep in tqdm(range(self.cfg.ws.num_train_epochs), ncols=80):
#             self._global_epoch = ep

#             # update agent
#             for _ in range(self.cfg.ws.update_per_epoch * len(self.train_env_dict)):
#                 metrics = self.agent.update(data=next(self.offline_iter))
#             # logging
#             if self.cfg.ws.use_wandb:
#                 metrics.update(self.common_metrics())
#                 metrics = {f'train/{k}': v for k, v in metrics.items()}
#                 wandb.log(metrics, step=self.global_epoch)

#             # save model
#             if self.cfg.snapshot.save_snapshot and self.global_epoch > 0 and self.global_epoch % self.cfg.snapshot.save_every_epochs == 0:
#                 self.work_dir.mkdir(parents=True, exist_ok=True)
#                 self.agent.save_model(self.work_dir/f'snapshot_{self.global_epoch}.pt')
#                 print(f'------- Save model at epoch {self.global_epoch} ------')

#             # eval
#             if self.global_epoch % self.cfg.ws.eval_every_epochs == 0:
#                 eval_metrics = self.eval()
#                 if self.cfg.ws.use_wandb:
#                     if 'eval/video' in eval_metrics:
#                         video = eval_metrics.pop('eval/video')
#                         wandb.log({'eval/video': wandb.Video(video, fps=15, format='mp4')}, step=self.global_epoch)
#                     wandb.log(eval_metrics, step=self.global_epoch)
                    
#         if self.cfg.snapshot.save_snapshot:
#             self.agent.save_model(self.work_dir/'snapshot.pt')
        

#     @staticmethod
#     def load_snapshot(snap_dir, seed):
#         def try_load(seed):
#             if not snap_dir.exists():
#                 return None
#             with snap_dir.open('rb') as f:
#                 payload = torch.load(f)
#             return payload

#         # try to load current seed
#         payload = try_load(seed)
#         if payload is not None:
#             print(f"Snapshot loaded from: {snap_dir}")
#             return payload
#         else:
#             raise Exception(f"Snapshot not found at: {snap_dir}")


# @hydra.main(config_path='../config', config_name='default')
# def main(cfg):
#     workspace = Workspace(cfg)
#     workspace.train()

# if __name__ == '__main__':
#     main()
