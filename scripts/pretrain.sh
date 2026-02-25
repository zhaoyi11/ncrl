export MUJOCO_GL=egl
DATASETS_PATH=<Data_Path>
python3 src/pretrain.py ws.experiment=pretrain-ablation ws.mode=pretrain \
        ws.use_wandb=false ws.wandb_entity=<WandB_Entity> \
        ws.batch_size=32 \
        buffer.offline_path=$DATASETS_PATH buffer.num_workers=4 ws.log_every_steps=1000 ws.recon_every_steps=5000 ws.save_every_steps=20000