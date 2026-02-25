## NCRL: Efficient Reinforcement Learning by Guiding World Models with Non-curated Data
PyTorch implementation of the NCRL algorithm from [Efficient Reinforcement Learning by Guiding World Models with Non-curated Data](https://openreview.net/forum?id=oBXfPyi47m&noteId=IyqKwANgzF). NCRL enables RL agents to effectively leverage reward-free and multi-embodiment offline data with world models.

>Leveraging offline data is a promising way to improve the sample efficiency of online reinforcement learning (RL). This paper expands the pool of usable data for offline-to-online RL by leveraging abundant non-curated data that is reward-free, of mixed quality, and collected across multiple embodiments. Although learning a world model appears promising for utilizing such data, we find that naive fine-tuning fails to accelerate RL training on many tasks. Through careful investigation, we attribute this failure to the distributional shift between offline and online data during fine-tuning. To address this issue and effectively use the offline data, we propose two techniques: i) experience rehearsal and ii) execution guidance. With these modifications, the non-curated offline data substantially improves RL's sample efficiency. Under limited sample budgets, our method achieves a 102.8\% relative improvement in aggregate score over learning-from-scratch baselines across 72 visuomotor tasks spanning 6 embodiments. On challenging tasks such as locomotion and robotic manipulation, it outperforms prior methods that utilize offline data by a decent margin.

### Installation
Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate ncrl_env
```
---
### Models and Datasets
Models and datasets used in the paper are hosted on Hugging Face https://huggingface.co/datasets/zhaoyi11/ncrl.
After downloading: a) unzip the files, and b) move the models and datasets to `<Model_Path>` and `<Data_Path>` respectively.

---
### Training Pipeline
1. Build the dataset database:
```python
python3 src/prepare_dataset_kv.py \
        --model-basedir <Model_Path> \
        --data-basedir <Data_Path> \
        --path-to-save <Database_Path>/dataset_db.npz
```

2. (Optional) World model pre-training
Edit `DATASETS_PATH` in `scripts/pretrain.sh` to `<Data_Path>`, then run:
```bash
./scripts/pretrain.sh
```

3. Task-specific fine-tuning
Edit the following fields in `scripts/finetune.sh`
    - `SNAPSHOT_BASEDIR` → `<Model_Path>` (or your pretrained world model in step 2)
    - `DATA_DB_PATH` → `<Database_Path>/dataset_db.npz`
    - `ENV` → Training env name (see `src/envs.py` for the full list)
Then run:
```bash
./scripts/finetune.sh
```
---
### BibTeX
If you find this repository useful for your research, please consider citing:
```latex
@inproceedings{zhaoefficient,
  title     = {Efficient Reinforcement Learning by Guiding World Models with Non-Curated Data},
  author    = {Zhao, Yi and Scannell, Aidan and Zhao, Wenshuai and Hou, Yuxin and Cui, Tianyu and Chen, Le and B{\"u}chler, Dieter and Solin, Arno and Kannala, Juho and Pajarinen, Joni},
  booktitle = {The Fourteenth International Conference on Learning Representations}
  year      = {2026}
}
```