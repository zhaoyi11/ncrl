import warnings
from time import time

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import uuid
import time
import random
from math import ceil
from pathlib import Path
import hydra
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import wandb
from collections import namedtuple, deque
import omegaconf
import faiss
from omegaconf import OmegaConf

import envs
import utils
import dreamer.nets as nets
from replay_buffer import ReplayBufferStorage, make_replay_loader
from dreamer.dreamer import DreamerAgent
from behavior_cloning import BCAgent

warnings.filterwarnings('ignore', category=DeprecationWarning)

torch.backends.cudnn.benchmark = True


class Workspace:
    def __init__(self, cfg):        
        # task-specific settings
        # for dmc hard version tasks -- sparse reward and action penalty
        if cfg.env.task.startswith('dmc-cheetah'):
            cfg.env.reward_threshold = 0.2
        else:
            cfg.env.reward_threshold = 0.6

        # set up workspace
        cfg.ws.run_id = str(uuid.uuid4().hex)
        if cfg.ws.seed == -1: # random sample a seed if ==-1
            cfg.ws.seed = random.randint(0, 1000)

        # set work_dir
        _env_name = cfg.env.task 
        _run_path = Path('logs')/cfg.ws.experiment/f'{_env_name}'/f'{cfg.ws.run_id}_{cfg.ws.seed}'
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
        self.eval_env = envs.make_env(cfg.env, cfg.ws.seed+10)

        # set warmup frames, no less than 5 episodes
        cfg.ws.warmup_frames = max(cfg.ws.warmup_frames, 5 * self.train_env.episode_length)

        self.obs_shape = obs_shape = self.train_env.obs_space.image.shape
        self.act_dim = act_dim = self.train_env.action_space.shape[0]

        if cfg.ws.use_action_padding: # TODO: check if this is correct
            padded_act_dim = envs.MAX_ACTION_DIM
        else:
            padded_act_dim = act_dim

        # Initialize agent
        self.agent = DreamerAgent(cfg.agent, obs_shape, act_dim, padded_act_dim, device=cfg.ws.device) 
        if cfg.snapshot.load_path is not None:
            snapshot_dir = Path(cfg.snapshot.load_path)
            if not str(snapshot_dir).endswith('.pt'):
                if cfg.env.task.startswith('dmc-'):
                    snapshot_dir = snapshot_dir / 'dmcontrol.pt'
                elif cfg.env.task.startswith('mw-'):
                    snapshot_dir = snapshot_dir / 'metaworld.pt'
                else:
                    raise ValueError(f"Unknown task: {cfg.env.task} for loading checkpoint.")

            self.agent.load(snapshot_dir, cfg.snapshot.models_to_load)
            print(f"--- Load snapshot from {snapshot_dir} ---")

        # setup replay buffer
        self.replay_storage, self.replay_loader = self._setup_buffer(
                                cfg = cfg.buffer,
                                data_dir=self.work_dir/'buffer', 
                                obs_shape=obs_shape,
                                max_act_dim=padded_act_dim, # for action padding
                                batch_size=int(cfg.buffer.online_ratio * cfg.buffer.batch_size),
                                save_buffer=cfg.buffer.save_buffer)
        self._replay_iter = None
        
        # setup retrieve buffer
        if cfg.buffer.online_ratio < 1.:
            if cfg.buffer.offline_path is not None:
                self.offline_buffer_path = Path(cfg.buffer.offline_path)
            else:
                self.offline_buffer_path = self.work_dir/'prior_buffer'
            self.offline_storage, self.offline_loader = self._setup_buffer(
                                cfg=cfg.buffer,
                                data_dir=self.offline_buffer_path,
                                obs_shape=obs_shape,
                                max_act_dim=padded_act_dim, #for action padding
                                batch_size=int((1.-cfg.buffer.online_ratio) * cfg.buffer.batch_size),
                                save_buffer=True)
            
            # pre-fill the replay buffer
            if cfg.buffer.kv_path is not None:
                print(f'--- Pre-filling the replay buffer with offline data ---')
                num_retrieved_episodes = 0   
                num_retrieve_iter = 0
                while num_retrieved_episodes < cfg.buffer.num_retrieve_episodes:
                    self.offline_storage, num_retrieved_episodes = self._retrieve_buffer(self.offline_storage, kv_path=cfg.buffer.kv_path, 
                                        query=self.train_env.reset().observation, # query frame is the first frame of the trajectory
                                        feature_extractor=self.agent.wm.encoder,
                                        env_name=cfg.env.task, 
                                        num_retrieve_episodes=min((cfg.buffer.num_retrieve_episodes - num_retrieved_episodes), 500)) # retrieve 500 episodes at most per iteration
                    num_retrieve_iter += 1
                    if num_retrieve_iter > 10:
                        print(f"[Warning] Cannot retrieve enough data from the offline dataset. Expect {cfg.buffer.num_retrieve_episodes} episodes, but only {num_retrieved_episodes} episodes are retrieved.")
                    print(f"------ Retrieved {num_retrieved_episodes} episodes using {num_retrieve_iter} iterations. ------")
            self._offline_iter = None

        self.cfg = cfg
        # create logger
        if self.cfg.ws.use_wandb and wandb.run is None:
            self._setup_wandb()

        # setup bc agent
        if cfg.ws.use_bc and cfg.buffer.online_ratio < 1.:
            # input of bc agent is the 3-frame stack
            self.bc_agent = BCAgent((9, 64, 64), padded_act_dim).to(self.device)
            # set up bc buffer
            self.bc_storage, self.bc_loader = self._setup_buffer(
                                cfg = cfg.bc_buffer,
                                data_dir=self.offline_buffer_path,
                                obs_shape=obs_shape,
                                max_act_dim=padded_act_dim, # for action padding
                                batch_size=int(cfg.ws.bc_batch_size),
                                save_buffer=True)
            self._bc_iter = None
            
            # train the bc agent on the prior buffer
            print(f'--- Training BC agent on the prior buffer ---')
            for bc_i in tqdm(range(cfg.ws.bc_train_steps)):
                metrics = self.bc_agent.update(next(self.bc_iter))
                if bc_i % 1000 == 0:
                    if cfg.ws.use_wandb:
                        wandb.log({f'bc/train/{k}': v for k, v in metrics.items()})
            print(f'--- BC agent training done | {metrics} ---')
            
        # Globals
        self._start_time = time.time()
        self._global_step = 0
        self._global_episode = 0
        self._global_epoch = 0

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self.global_step,
            frame=self.global_frame,
            episode=self.global_episode,
            epoch=self.global_epoch,
            total_time=time.time() - self._start_time,
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.env.action_repeat

    @property
    def global_epoch(self):
        return self._global_epoch

    # iterators for replay buffers
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    @property
    def offline_iter(self):
        if self._offline_iter is None:
            self._offline_iter = iter(self.offline_loader)
        return self._offline_iter

    @property
    def bc_iter(self):
        if self._bc_iter is None:
            self._bc_iter = iter(self.bc_loader)
        return self._bc_iter

    def _setup_buffer(self, cfg, data_dir, obs_shape, max_act_dim, batch_size, save_buffer):
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
            save_buffer=save_buffer)
        return replay_storage, replay_loader    

    @torch.no_grad()
    def _retrieve_buffer(self, storage, kv_path, query, feature_extractor, num_retrieve_episodes, env_name):
        # load kv pair of the offline dataset
        if isinstance(kv_path, str):
            kv_path = Path(kv_path)
        with kv_path.open('rb') as f:
            _data = np.load(f, allow_pickle=True)
            kv_dataset = {}
            for k in _data.keys():
                kv_dataset[k] = _data[k]
            kv_dataset = {k: _data[k].item() for k in _data.keys()}
        _benchmark = env_name.split('-')[0]

        filename = kv_dataset[_benchmark]['filename']
        features = kv_dataset[_benchmark]['feature']
        # faiss index
        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(features)

        # retrieve the most similar frame to the first frame of each trajectory
        # and save the retrieved data to the storage
        if query.ndim == 3:
            query = query[None]
        assert query.ndim == 4, "Query frame must be 4D tensor."
        
        # calcualte r3m feature of the query frame
        query = torch.from_numpy(query).to(device=self.device, dtype=torch.float32)
        # normalize the query frame
        query = query / 255. - 0.5
        with torch.no_grad():
            query_feat = feature_extractor(query) 
            query_feat = query_feat.cpu().numpy()
        # retrieve the most similar frame to the first frame of each trajectory
        D, I = index.search(query_feat, num_retrieve_episodes)
        I = I[0] # I has shape (B, num_retrieve_episodes), since we only have one query frame, I[0] is used to get the indices.
        
        for i in I:
            fn = filename[i]
            # load the trajectory
            _path = Path(str(fn))
            with _path.open('rb') as f:
                _data = np.load(f, allow_pickle=True)
                data = {k: _data[k] for k in _data.keys()}

            # filter out the data that has different action dimension
            if data['action'].shape[-1] != envs.TASK_ACT_DIM[env_name]:
                continue
            # add the data to the storage
            storage.copy_episode(data)
        return storage, storage._num_episodes

    def _setup_wandb(self):
        _cfg = self.cfg.ws
        wandb.init(project=_cfg.project_name, entity=_cfg.wandb_entity, 
                name=f'{_cfg.mode}-{self.cfg.env.task}-{_cfg.project_name}-{_cfg.experiment}-{_cfg.run_id}-{str(_cfg.seed)}',
                group=f'{self.cfg.env.task}-{_cfg.project_name}', 
                tags=[_cfg.project_name, f'{self.cfg.env.task}', _cfg.experiment, str(_cfg.seed)],
                config=omegaconf.OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True))

    def eval(self):
        metrics = {}
        score = []
        videos = {}
        
        episode_rewards, episode_successes = [], []
        for i in range(self.cfg.ws.num_eval_episodes):
            video = []
            episode_step, ep_reward = 0, 0
            timestep = self.eval_env.reset()
            agent_state = None
            # video recording
            if self.cfg.ws.save_video and i == 0:
                video.append(timestep.observation)
            while not bool(timestep.done):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, latent = self.agent.act(timestep, 
                                        self.global_step,
                                        eval_mode=True,
                                        state=agent_state,)
                    agent_state = (latent, action)
                    action = action.cpu().numpy()[0][:self.act_dim]
                timestep = self.eval_env.step(action)
                if self.cfg.ws.save_video and i == 0:
                    video.append(timestep.observation)

                ep_reward += timestep.reward
                episode_step += 1

            # video recording
            if video:
                videos[self.cfg.env.task] = np.stack(video)

            episode_rewards.append(ep_reward)
            
            if isinstance(timestep.info, dict) and 'success' in timestep.info: # for meta-world envs
                episode_successes.append(timestep.info.get('success', 0.))
        
        metrics[f'{self.cfg.env.task}/episode_reward'] = np.mean(episode_rewards)
        if len(episode_successes) > 0:
            metrics[f'{self.cfg.env.task}/episode_success'] = np.mean(episode_successes)

        if self.cfg.env.task.startswith('mw-'):
            score.append(metrics[f'{self.cfg.env.task}/episode_success']*100)
        elif self.cfg.env.task.startswith('dmc-'):
            score.append(metrics[f'{self.cfg.env.task}/episode_reward'] / 10) # dmc 1000 reward -> 100 score
        else:
            score.append(metrics[f'{self.cfg.env.task}/episode_reward'] / 2) # robosuite 200 reward -> 100 score
            
        metrics.update(self.common_metrics())
        metrics.update({'avg_episode_reward': np.mean([v for k, v in metrics.items() if 'episode_reward' in k])})
        metrics.update({'norm_score': np.mean(score)})
        metrics = {f'eval/{k}': v for k, v in metrics.items()}
        if videos:
            metrics['eval/video'] = videos
        return metrics

    def train(self):
        # In each epoch, we first collect training data from env and then do the update.
        FRAMES_PER_EPOCH = self.train_env.episode_length
        STEPS_PER_EPOCH = ceil(FRAMES_PER_EPOCH / self.cfg.env.action_repeat)
        NUM_EPOCH = ceil(self.cfg.ws.num_train_frames / FRAMES_PER_EPOCH)
        WARMUP_EPOCH = ceil(self.cfg.ws.warmup_frames / FRAMES_PER_EPOCH)
        GRAD_PER_EPOCH = ceil(STEPS_PER_EPOCH / (self.cfg.ws.batch_size * self.cfg.ws.chunk_length / self.cfg.ws.train_ratio))
        RECON_EVERY_EPOCH = ceil(self.cfg.ws.recon_every_frames / FRAMES_PER_EPOCH)
        EVAL_EVERY_EPOCH = ceil(self.cfg.ws.eval_every_frames  / FRAMES_PER_EPOCH)
        SAVE_EVERY_EPOCH = ceil(self.cfg.snapshot.save_every_frames / FRAMES_PER_EPOCH)
        print(f'------ Training Starts. Warmup epoch: {WARMUP_EPOCH} | Grad update per epoch: {GRAD_PER_EPOCH} | Eval every epoch {EVAL_EVERY_EPOCH} ------')
        for epoch in tqdm(range(1, NUM_EPOCH), ncols=80):
            self._global_epoch = epoch
            
            # collect data
            # for env_name, train_env in self.train_env_dict.items():
            episode_step, episode_reward = 0, 0
            timestep = self.train_env.reset()
            agent_state = None
            
            self.replay_storage.add(timestep)

            # prepare obs for behavior cloning agent if necessary
            if hasattr(self, 'bc_agent'): # for frame_stack
                FRAME_STACK = 3
                bc_obs_buffer = deque([timestep.observation] * FRAME_STACK, maxlen=FRAME_STACK)

                # setup execution guidance schedule
                if random.random() < utils.schedule(self.cfg.ws.jsrl_schedule, self.global_step):
                    _env_step = self.train_env.episode_length // self.cfg.env.action_repeat
                    _warmup_steps = [(_env_step / 10) * i for i in range(1, 11)]
                    BC_WARMUP_STEPS = random.choice(_warmup_steps)
                    BC_START_STEP = random.randint(0, max(0, _env_step - BC_WARMUP_STEPS))
                else:
                    BC_WARMUP_STEPS, BC_START_STEP = 0, 0

            # collection loop
            while not bool(timestep.done):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.global_epoch > WARMUP_EPOCH:
                        action, latent = self.agent.act(timestep, 
                                            self.global_step,
                                            eval_mode=False,
                                            state=agent_state,
                                            )

                        # replace action with bc action 
                        if hasattr(self, 'bc_agent') and BC_START_STEP < episode_step < BC_WARMUP_STEPS + BC_START_STEP:
                            action = self.bc_agent.act(torch.from_numpy(
                                                    np.concatenate(list(bc_obs_buffer))
                                                    ).to(self.device))
                        # update agent state
                        agent_state = (latent, action)
                        # convert to numpy
                        action = action.cpu().numpy()[0][:self.act_dim]
                    else: # use behavior cloning agent or random agent
                        if hasattr(self, 'bc_agent'):
                            action, agent_state = self.bc_agent.act(torch.from_numpy(
                                                    np.concatenate(list(bc_obs_buffer))
                                                    ).to(self.device))[0].cpu().numpy()[:self.act_dim], None
                        else: # random action
                            action = np.random.uniform(-1, 1, size=(self.act_dim,)).astype(np.float32)
                timestep = self.train_env.step(action)

                # save data
                self.replay_storage.add(timestep) 

                if hasattr(self, 'bc_agent'): # for frame_stack
                    bc_obs_buffer.append(timestep.observation)

                episode_reward += timestep.reward
                episode_step += 1
                self._global_step += 1
            self._global_episode += 1 

            # logging
            if self.cfg.ws.use_wandb:
                wandb.log({f'agent/{self.cfg.env.task}/episode_reward': episode_reward,
                            }, step=self.global_frame)
            print(f'Epoch: {self.global_epoch} | {self.cfg.env.task} | Return: {episode_reward} | Episode steps: {episode_step}')

            # update agent
            if self.global_epoch > min(WARMUP_EPOCH, 10):
                for _ in range(GRAD_PER_EPOCH):
                    metrics = self.agent.update(
                        online_data=None if self.cfg.buffer.online_ratio==0. else next(self.replay_iter), 
                        offline_data=None if self.cfg.buffer.online_ratio==1. else next(self.offline_iter), 
                        step=self.global_step)[1]
                # logging (every epoch)
                if self.cfg.ws.use_wandb:
                    metrics.update(self.common_metrics())
                    metrics = {f'train/{k}': v for k, v in metrics.items()}
                    wandb.log(metrics, step=self.global_frame)

            # log reconstrunction
            if self.global_epoch > min(WARMUP_EPOCH, 10) and self.global_epoch % RECON_EVERY_EPOCH == 0: 
                videos = self.agent.report(next(self.replay_iter))
                if self.cfg.ws.use_wandb:
                    for k, v in videos.items():
                        v = np.uint8(v.cpu() * 255)
                        wandb.log({k: wandb.Video(v, fps=15, format="mp4")}, step=self.global_frame)

            #  evaluate
            if self.global_epoch % EVAL_EVERY_EPOCH == 0:
                eval_metrics = self.eval()
                if self.cfg.ws.use_wandb:
                    if 'eval/video' in eval_metrics:
                        _video = eval_metrics.pop('eval/video')
                        for k, v in _video.items():
                            wandb.log({f'eval/video_{k}': wandb.Video(v, fps=15, format="mp4")}, step=self.global_frame)
                    wandb.log(eval_metrics, step=self.global_frame)
                
            # save model
            if self.cfg.snapshot.save_snapshot and self.global_epoch > 0 and self.global_epoch % SAVE_EVERY_EPOCH == 0:
                self.agent.save_model(self.work_dir/f'snapshot_{self.global_frame}.pt')
                print(f'------- Save model at epoch {self.global_frame} ------')

  
        if self.cfg.snapshot.save_snapshot:
            self.agent.save_model(self.work_dir/'snapshot.pt')
        
        # remove buffer for saving storage.
        if not self.cfg.buffer.save_buffer:
            import shutil
            try:
                shutil.rmtree(self.work_dir/"buffer")
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))


@hydra.main(config_path='../config', config_name='finetune')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()
