## Steps to run Experiments

### Preprocessing the data
Run the following command locally (not inside Docker):
```bash
DATASET_PATH=/home/vagsart/Desktop/IndustryShapesClassicv2
python3 utils/prepare_dataset.py \
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

Apply some minor modifications to `Deep_Object_Pose` repo:
```bash
cd Deep_Object_Pose
git apply ../diff.patch
```


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
    --batchsize 2 \
    --outf /data/training_log/ \
    --data /data/preprocessed/
```

If you want to check training process in tensorboard:
```bash
tensorboard --logdir=data/training_log/runs --port=6006 --host=localhost
```


### Running inference

```bash
docker run --gpus all --rm -it \
  -v $(pwd)/Deep_Object_Pose:/dope \
  -v $(pwd)/data:/data \
  -v $(pwd)/config:/config \
  -w /dope/train2 \
  nvidia-dope:noetic-v100 \
  python3 inference.py --config /config/config_pose.yaml --data /data/preprocessed/000002  --outf /data/results --camera /config/camera_info.yaml
```

### Evaluation using bop toolkit


```bash
python utils/results_to_bop.py \
  --root_folder /home/vagsart/code/dope_fork/scripts/train2/results/000001/ \
  --outf results/ \
  --modelpath /home/vagsart/Datasets/IndustryShapes/models_eval
```

(Optional) If you want to visualize the results, add `--visualize` flag to the command above.
```bash