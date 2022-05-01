DATASET=imagenet_may26 #ssd_300_reduced cifar10_july8 kws_chu_july19 imagenet_may26 vit_cifar100
INPUT_MODEL_DIR=/home/mendel/disk-USB/model-c
OUTPUT_LOG_DIR=/home/mendel/disk-USB/log-EdgeTPU
WARM_COUNT=3
COUNT=10
PRINT_INTERVAL=1000
BREAK_FLAG=0
START_INDEX=0
END_INDEX=100000

python3 examples/benchmark_tflite.py \
-d $DATASET \
-i $INPUT_MODEL_DIR \
-o $OUTPUT_LOG_DIR \
-w $WARM_COUNT \
-c $COUNT \
-p $PRINT_INTERVAL \
-b $BREAK_FLAG \
-s $START_INDEX \
-e $END_INDEX
