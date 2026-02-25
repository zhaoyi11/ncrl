export MUJOCO_GL=egl
SNAPSHOT_BASEDIR=<Model_Path>
DATA_DB_PATH=<DataBase_Path/dataset_db.npz>
ENV="dmc-quadruped-walk"
python3 src/finetune.py \
        ws.use_wandb=false ws.wandb_entity=<WandB_Entity> ws.seed=-1 \
        ws.mode=finetune ws.num_train_frames=500_000 \
        ws.use_action_padding=true env.task=$ENV snapshot.load_path=$SNAPSHOT_BASEDIR \
        buffer.num_workers=4 buffer.online_ratio=0.75 buffer.kv_path=$DATA_DB_PATH \
        ws.use_bc=false agent.rssm.deter=12288 agent.rssm.hidden=1536 agent.rssm.classes=96 agent.encoder.cnn_depth=96 ws.batch_size=4