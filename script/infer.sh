cd ${PROJECT_PATH}

PIP="$(which python) -m pip"

# init with dependencies
echo "Install pip dependencies to $(which python)"
pip install pynvml
pip install bytedance.ndtimeline==2.2.7

pip install pytorch-lightning==1.5.0
pip install deepspeed==0.16.5
pip install datasets==3.4.1
pip install open-clip-torch

sudo apt install fonts-noto-cjk
sudo pip install accelerate==1.1.0
sudo pip install diffusers==0.32.2
pip install diffusers==0.32.2
sudo pip install --upgrade huggingface-hub
sudo pip install --upgrade tokenizers
sudo pip install transformers -U
pip install peft==0.17.0


echo "ACCELERATE_NUM_PROCESSES: $ACCELERATE_NUM_PROCESSES"
echo "ACCELERATE_NUM_MACHINES: $ACCELERATE_NUM_MACHINES"
echo "ACCELERATE_MACHINE_RANK: $ACCELERATE_MACHINE_RANK"
echo "ACCELERATE_MAIN_PROCESS_IP: $ACCELERATE_MAIN_PROCESS_IP"
echo "ACCELERATE_MAIN_PROCESS_PORT: $ACCELERATE_MAIN_PROCESS_PORT"

# init ckpt path
MODEL_PATH=${PRETRAIN_MODEL_PATH} # FLUX.1 pretrain path
BASEMODEL="UniGenFlux"

CONFIG="./config/unigen.yaml"

TRAINED_TRANSFORMER=${TRAINED_WORKDIR}
STEP=$(cat "$TRAINED_TRANSFORMER/latest")

WORKDIR="${TRAINED_TRANSFORMER}/generate_res/${STEP}"

echo "*************************************"
echo "load step ${STEP} ckpt to fp32"
if [ ! -f "${TRAINED_TRANSFORMER}/${STEP}/pytorch_model_fp32.bin" ]; then
    python $TRAINED_TRANSFORMER/zero_to_fp32.py ${TRAINED_TRANSFORMER} ${TRAINED_TRANSFORMER}/${STEP}/pytorch_model_fp32.bin
fi
echo "*************************************"

accelerate launch --config_file  ./config/accelerate_ds_config.yaml \
    --num_processes $ACCELERATE_NUM_PROCESSES \
    --num_machines $ACCELERATE_NUM_MACHINES \
    --machine_rank $ACCELERATE_MACHINE_RANK \
    --main_process_ip $ACCELERATE_MAIN_PROCESS_IP \
    --main_process_port $ACCELERATE_MAIN_PROCESS_PORT \
    ./infer.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --transformer "${TRAINED_TRANSFORMER}/${STEP}/pytorch_model_fp32.bin" \
    --cn_config $CONFIG \
    --basemodel $BASEMODEL \
    --data_path $DATAPATH \
    --condition_types $CONDITION_TYPE \
    --batch_size 1 \
    --dataloader_num_workers 4 \
    --num_inference_steps 50 \
    --max_sequence_length 512 \
    --guidance_scale 3.5 \
    --work_dir $WORKDIR \
    ${@:3}