export MUJOCO_GL=egl
SNAPSHOT_BASEDIR="/flash/project_462000782/cont_pretrain/checkpoints"
DATA_DB_PATH="/scratch/project_462000782/icml2025_dataset/dataset_db.npz"
python3 src/finetune.py \
        ws.use_wandb=true ws.wandb_entity=yi-zhao-aalto ws.seed=-1 \
        ws.mode=finetune ws.num_train_frames=500_000 \
        ws.use_action_padding=true env.task=$ENV snapshot.load_path=$SNAPSHOT_BASEDIR \
        buffer.num_workers=4 buffer.online_ratio=0.75 buffer.kv_path=$DATA_DB_PATH \
        ws.use_bc=true agent.rssm.deter=12288 agent.rssm.hidden=1536 agent.rssm.classes=96 agent.encoder.cnn_depth=96