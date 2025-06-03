CONFIG=evaluation/configs/lift_vit_b16.yaml # Your evaluation config file
export CUDA_VISIBLE_DEVICES=0 

# You can run the evaluation on other datasets by changing the next line to
# torchrun -m evaluation.zeroshot.<coco, imagenet, flickr, sugarcrepe>
torchrun -m evaluation.evaluate.sugarcrepe \
  --config=$CONFIG \