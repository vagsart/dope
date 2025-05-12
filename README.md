## Steps to run Experiments

### Preprocessing the data
Run the following command locally (not inside Docker):
```bash
DATASET_PATH=/home/vagsart/Datasets/IndustryShapes
python3 preprocessing/convert_to_dope_dataset.py \
  --data_folder $DATASET_PATH/train_primesense/ \
  --output_folder data/preprocessed \
  --models_path $DATASET_PATH/models_eval \
  --obj_map  obj_names.json\
  --scenes "000001 000002 000003 000004 000005 000006 000007 000008 000009 000010 000011"
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
    --epochs 100 \
    --batchsize 1 \
    --outf /data/training_log/ \
    --data /data/preprocessed
```

If you want to check training process in tensorboard:
```bash
tensorboard --logdir=data/training_log/runs --port=6006 --host=localhost
```

If raise error change `Deep_Object_Pose/train2/train.py:140` to:
```python
parser.add_argument("--local-rank", type=int)
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
