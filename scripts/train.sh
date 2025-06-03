CONFIG=configs/lift_vit_b16_128m.yaml # Your training config file
DIR=LIFT
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-55565}
PORT=9007
DIR=${DIR:-"LIFT"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
HOSTE_NODE_ADDR=${MASTER_ADDR}:${PORT}
TIMESTAMP=$(date +%Y_%m_%d-%H_%M_%S)
export TOKENIZERS_PARALLELISM=true
echo "$DIR"
cd $DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Automatically determine nproc_per_node based on CUDA_VISIBLE_DEVICES
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

torchrun --nproc_per_node=$NPROC_PER_NODE \
        --rdzv_endpoint=$HOSTE_NODE_ADDR \
       	--nnodes=$NNODES --node_rank=$NODE_RANK \
        -m training.main \
        --config="${CONFIG}" \