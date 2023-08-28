#!/bin/bash

cd ..

# denovo train
#python3 train.py \
#  --weights  weights/yolov7-lite-s.pt \
#  --cfg configs/control/yolov7-lite-s.yaml \
#  --data  configs/control/dataset.yaml \
#  --hyp  configs/control/hyp.yaml \
#  --epochs  2000 \
#  --batch-size  2 \
#  --device  0  \
#  --project runs/control/train

# distributed
python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 8585 train.py \
  --weights  weights/yolov7-lite-s.pt \
  --cfg configs/control/yolov7-lite-s.yaml \
  --data  configs/control/dataset.yaml \
  --hyp  configs/control/hyp.yaml \
  --epochs  2000 \
  --batch-size  256 \
  --device  0,1  \
  --project runs/control/train 


# distribute + load pretrained model
#python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 8585 train.py \
#  --weights  runs/control/train/exp/weights/last.pt \
#  --cfg configs/control/yolov7-lite-s.yaml \
#  --data  configs/control/dataset.yaml \
#  --hyp  configs/control/hyp.yaml \
#  --epochs  2000 \
#  --batch-size  384 \
#  --device  0,1  \
#  --project runs/control/train \
#  --resume "/cv/xyc/yolov7_plate/runs/control/train/exp2/weights/last.pt"