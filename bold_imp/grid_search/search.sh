#!/bin/bash

source /usr3/graduate/wfchen/ml2/venv/bin/activate

cd ..


# --dataset cifar10
# --arch-custom 
# --opt-momentum 
# --opt-probabilistic 
# --input-int 
# --loss-int-l1 
# --spread-std
# --epochs 100 
# --loss-int-l1-scale -50.0 50.0
# --loss-int-l1-salient 50
# --activation-range -1 1
# --prob-flip-ratio-decay 0.9
# --prob-flip-ratio-decay-epochs 5 
# --prob-flip-ratio-batch-acc-aware 0.3 
# --batch-size 384

# Define arrays for different parameter values
l1_scales=("-1.0 1.0" "-10.0 10.0" "-50.0 50.0" "-100.0 100.0" "-500.0 500.0" "-1000.0 1000.0")
l1_salients=("1" "10" "50" "100" "500" "1000")
act_ranges=("-1 1" "-10 10" "-100 100" "-1000 1000")
flip_ratios=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
flip_epochs=("1" "5" "10")
flip_baas=("0.1" "0.3" "0.6" "0.9")
batch_sizes=("64" "128" "256" "350")

id=0

# Loop through all combinations of parameters
for l1_scale in "${l1_scales[@]}"; do
    for l1_salient in "${l1_salients[@]}"; do
        for act_range in "${act_ranges[@]}"; do
            for flip_ratio in "${flip_ratios[@]}"; do
                for flip_epoch in "${flip_epochs[@]}"; do
                    for flip_baa in "${flip_baas[@]}"; do
                        for batch_size in "${batch_sizes[@]}"; do
                            echo "(ID=$id) Running with parameters: l1_scale=$l1_scale, l1_salient=$l1_salient, act_range=$act_range, flip_ratio=$flip_ratio, flip_epoch=$flip_epoch, flip_baa=$flip_baa, batch_size=$batch_size"

                            python3 bold_int.py --dataset cifar10 --arch-custom --opt-momentum --opt-probabilistic --input-int --loss-int-l1 \
                            --spread-std \
                            --epochs 10 \
                            --loss-int-l1-scale $l1_scale \
                            --loss-int-l1-salient $l1_salient \
                            --activation-range $act_range \
                            --prob-flip-ratio-decay $flip_ratio \
                            --prob-flip-ratio-decay-epochs $flip_epoch \
                            --prob-flip-ratio-batch-acc-aware $flip_baa \
                            --batch-size $batch_size \
                            --arch `cat ./archs/test | tr -d '\n'` > result_${id}.txt

                            echo "(ID=$id) Finished! Result saved to result_${id}.txt"
                            id=$((id+1))
                        done
                    done               
                done
            done
        done
    done
done
