DATASET=kws_chu_july19 #ssd_300_reduced cifar10_july8 kws_chu_july19 imagenet_may26 vit_cifar100
INPUT_DATASET_DIR=/home/mendel/disk-USB/model-c/$DATASET
OUTPUT_DATASET_LOG_DIR=/home/mendel/disk-USB/log-EdgeTPU/$DATASET
WARM_COUNT=3
COUNT=10
PRINT_INTERVAL=1000

COUNTER=0
for f in $INPUT_DATASET_DIR; do
    MODEL_PATH=$INPUT_DATASET_DIR/$f
    if [ -f "$MODEL_PATH" ]; then
        python3 examples/benchmark_tflite_single.py -m $f -i $INPUT_DATASET_DIR -o $OUTPUT_DATASET_LOG_DIR -w $WARM_COUNT -c $COUNT 
        let COUNTER=COUNTER+1 
    else 
        echo "$MODEL_PATH does not exist."
    fi

    REMAINDER=$(($COUNTER%$PRINT_INTERVAL))
    if [ $REMAINDER -eq 0 ]; then
        echo "Processed $COUNTER Models."
    fi
done