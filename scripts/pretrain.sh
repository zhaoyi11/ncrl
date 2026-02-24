# WORK_BASEDIR="/flash/project_462000782/cont_pretrain"
export MUJOCO_GL=egl
WORK_BASEDIR="/home/zhaoy13/rl/ncrl"
#DATASETS_PATH="/scratch/project_462000782/icml2025_dataset/mixed/ablation"
# DATASETS_PATH="/scratch/project_462000782/icml2025_dataset/all"
DATASETS_PATH="/data/metaworld_datasets/mw-assembly"
# 400 M
python3 src/pretrain.py ws.experiment=pretrain-ablation ws.mode=pretrain \
        ws.use_wandb=true ws.wandb_entity=yi-zhao-aalto \
        ws.batch_size=32 ws.work_basedir=$WORK_BASEDIR \
        buffer.offline_path=$DATASETS_PATH buffer.num_workers=4 ws.log_every_steps=1000 ws.recon_every_steps=5000 ws.save_every_steps=20000