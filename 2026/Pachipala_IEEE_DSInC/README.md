# DSInC

## Installation

The code was originally tested with Python 3.10.19, CUDA 12.1.
For RTX 5090 / Blackwell-class GPUs, use the 5090 setup script.

### Recommended 5090 Setup
DSInC includes the VLFM code needed for evaluation under `third_party/vlfm`. The setup below is intended for headless Docker/Linux environments with CUDA support.

Run these scripts from the DSInC repo root. `setup_env_5090.sh` creates the evaluation environment, `hb`. `setup_vlfm_server_env.sh` creates the VLM server environment, `cr_vlfm`, downloads public model weights, and prepares the YOLOv7/GroundingDINO runtime folders.
```
chmod +x setup_env_5090.sh
./setup_env_5090.sh

chmod +x setup_vlfm_server_env.sh
./setup_vlfm_server_env.sh
```

### Required VLFM weights
The setup scripts download the original VLFM PointNav checkpoint from:
```
https://github.com/rai-opensource/vlfm/blob/main/data/pointnav_weights.pth
```
and save it to:
```
third_party/vlfm/data/pointnav_weights.pth
```
If you already have a local original VLFM checkout, you can copy from it instead:
```
ORIGINAL_VLFM_ROOT=/path/to/vlfm ./setup_env_5090.sh
ORIGINAL_VLFM_ROOT=/path/to/vlfm ./setup_vlfm_server_env.sh
```
If the native downloaded PointNav checkpoint fails to load in your environment, use `utils/export_pointnav_state_dict.py` to export a compatible state-dict checkpoint.

The VLM server setup script auto-downloads these public server weights if they are missing:
```
third_party/vlfm/data/mobile_sam.pt
third_party/vlfm/data/groundingdino_swint_ogc.pth
third_party/vlfm/data/yolov7-e6e.pt
```
The setup script creates these local runtime checkouts when missing:
```
third_party/vlfm/yolov7/
third_party/vlfm/GroundingDINO/
```

### Download HM3D_v0.2 and MP3D datasets

#### Habitat Matterport
Download [HM3D_v0.2](https://aihabitat.org/datasets/hm3d/) and [MP3D](https://niessner.github.io/Matterport/) datasets using the download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).

### Setting up datasets
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
DSInC/
  data/
    scene_datasets/
        hm3d_v0.2/
            val/
            hm3d_annotated_basis.scene_dataset_config.json
            hm3d_annotated_val_basis.scene_dataset_config.json
        mp3d/
    matterport_category_mappings.tsv
    object_norm_inv_perplexity.npy
    versioned_data
    objectnav_hm3d_v2/
        train/
        val/
        val_mini/
```
The 2-agent HM3D benchmark uses `data/objectnav_hm3d_v2/val/val.json.gz` plus the per-scene files under `data/objectnav_hm3d_v2/val/content/`.

## Evaluation
### Start the VLM server
In a new terminal:
```
cd third_party/vlfm
conda activate cr_vlfm
./scripts/launch_vlm_servers.sh 
```

### Run 2-robot HM3D evaluation
From the DSInC repo root:
```
conda activate hb
```

Communication/Collaboration is enabled by default. The default benchmark settings are `--comm_fuse_window 8`, `--comm_novelty_threshold 0.85`, `--comm_novelty_beta 0.003`, `--max_episode_length 500`, and `--success_dist 1.0`.

You can run the local HM3D val set as 500-episode shards, for example:
```
python eval.py \
  -d ./DSInC_EXP/multi_hm3d_2-robot/ \
  --exp_name eval_s0 \
  --task_config tasks/multi_objectnav_hm3d_2agent.yaml \
  --num_shards 2 \
  --shard_index 0

python eval.py \
  -d ./DSInC_EXP/multi_hm3d_2-robot/ \
  --exp_name eval_s1 \
  --task_config tasks/multi_objectnav_hm3d_2agent.yaml \
  --num_shards 2 \
  --shard_index 1
```

For a no-communication baseline, add `--no_comm` and use separate experiment names:
```
python eval.py \
  -d ./DSInC_EXP/multi_hm3d_2-robot/ \
  --exp_name eval_s0_no_comm \
  --task_config tasks/multi_objectnav_hm3d_2agent.yaml \
  --num_shards 2 \
  --shard_index 0 \
  --no_comm

python eval.py \
  -d ./DSInC_EXP/multi_hm3d_2-robot/ \
  --exp_name eval_s1_no_comm \
  --task_config tasks/multi_objectnav_hm3d_2agent.yaml \
  --num_shards 2 \
  --shard_index 1 \
  --no_comm
```

To resume a shard from a local episode index (ex. 259):
```
python eval.py \
  -d ./DSInC_EXP/multi_hm3d_2-robot/ \
  --exp_name eval_s0_resume259 \
  --task_config tasks/multi_objectnav_hm3d_2agent.yaml \
  --num_shards 2 \
  --shard_index 0 \
  --resume_episode_index 259
```

### Summarize logs
After shard runs finish:
```
python compute_sr_spl.py --logs-dir ./DSInC_EXP/multi_hm3d_2-robot/logs
```
