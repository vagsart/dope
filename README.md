## Steps to run Experiments

### Preprocessing the data
Run the following command locally (not inside Docker):
```bash
python3 preprocessing/convert_to_dope_dataset.py \
  --data_folder data/scenes \
  --output_folder data/preprocessed \
  --models_path data/ \
  --obj_map data/ \
  --scenes "000006 000007"
```

### Build the docker image
From the directory containing the Dockerfile.noetic (`docker/`), run:

```bash
docker build -t nvidia-dope:noetic-v100 -f Dockerfile.noetic ..
```

### Training the model
From the project root (outside the Docker directory), run:

```bash
docker run --gpus all --rm -it \
  -v $(pwd)/Deep_Object_Pose:/dope \
  -v $(pwd)/data:/data \
  -w /dope/train2 \
  nvidia-dope:noetic-v100 \
  python3 -m torch.distributed.launch --nproc_per_node=1 train.py \
    --network dope \
    --epochs 2 \
    --batchsize 1 \
    --outf /data/training_log/ \
    --data /data
```

### Running inference

```bash
docker run --gpus all --rm -it \
  -v $(pwd)/Deep_Object_Pose:/dope \
  -v $(pwd)/data:/data \
  -v $(pwd)/config:/config \
  -w /dope/train2 \
  nvidia-dope:noetic-v100 \
  python3 inference.py --config /config/config_inference/epos_config_low_res/config_pose.yaml --data /data/rgb
```

### Evaluation using bop toolkit
