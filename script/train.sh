cd ${PROJECT_PATH}

PIP="$(which python) -m pip"

# init with dependencies
echo "Install pip dependencies to $(which python)"
pip install pynvml
pip install bytedance.ndtimeline==2.2.7

pip install pytorch-lightning==1.5.0
pip install huggingface-hub==0.36.0
pip install deepspeed==0.14.5
pip install datasets==3.4.1
pip install open-clip-torch

pip install peft
sudo apt install fonts-noto-cjk
# sudo pip install transformers==4.54.0
sudo pip install accelerate==1.10.0
sudo pip install diffusers==0.32.2

export NCCL_DEBUG=WARN

# for deepspeed zero stage 3
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=${1:-$ARNOLD_WORKER_0_PORT}

echo "ACCELERATE_NUM_PROCESSES: $ACCELERATE_NUM_PROCESSES"
echo "ACCELERATE_NUM_MACHINES: $ACCELERATE_NUM_MACHINES"
echo "ACCELERATE_MACHINE_RANK: $ACCELERATE_MACHINE_RANK"
echo "ACCELERATE_MAIN_PROCESS_IP: $ACCELERATE_MAIN_PROCESS_IP"
echo "ACCELERATE_MAIN_PROCESS_PORT: $ACCELERATE_MAIN_PROCESS_PORT"

# init ckpt path
MODEL_PATH=${PRETRAIN_MODEL_PATH} # FLUX.1 pretrain path

BASEMODEL="UniGenFlux" # UniGenSD3

CONFIG="./config/unigen.yaml"

# init training parameter
CHECKPOINT_STEPS=1000

accelerate launch --config_file  ./config/accelerate_ds_config.yaml \
    --num_processes $ACCELERATE_NUM_PROCESSES \
    --num_machines $ACCELERATE_NUM_MACHINES \
    --machine_rank $ACCELERATE_MACHINE_RANK \
    --main_process_ip $ACCELERATE_MAIN_PROCESS_IP \
    --main_process_port $ACCELERATE_MAIN_PROCESS_PORT \
    ./train.py \
    --basemodel $BASEMODEL \
    --data_path $DATAPATH \
    --cn_config $CONFIG \
    --pretrained_model_name_or_path $MODEL_PATH \
    --work_dir $WORKDIR \
    --condition_types $CONDITION_TYPE \
    --guidance_scale 3.5 \
    --checkpointing_steps $CHECKPOINT_STEPS \
    --max_train_steps $TRAIN_STEPS \
    --train_batch_size 1 \
    --is_deepspeed True \
    ${@:3}