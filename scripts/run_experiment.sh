#!/bin/bash

cd ../

first_or_second=$1
data_limit=1000
# method=lora
# dataset=ghc
# device=2
# balance_ratio=0.5
# warmup_ratio=0.06
weight_decay=0.01
gpus=(0 1 2 3 4 5 7)
num_gpus=${#gpus[@]}

LM=xlm-roberta-base
EPOCHS=10


datasets=(hindi ukraninan turkish portuguese latvian italian greek german estonian danish albanian arabic english russian chinese)
label_cols=(hate abusive offensive hate moderated hate offensive offensive moderated offensive offensive abuse hate toxicity offensive)
num_datasets=${#datasets[@]}


if [ "$first_or_second" == "first" ]; then
    echo "Running first stage"
    #---------------------------FIRST---------------------------
    for gpu_index in "${!gpus[@]}"; do       
        gpu=${gpus[$gpu_index]}
        dataset_index=$gpu_index
        concatenated_cmd=""

        while [ "$dataset_index" -lt "$num_datasets" ]; do
            
            dataset=${datasets[$dataset_index]}
            label_col=${label_cols[$dataset_index]}
            SESSION_NAME="$dataset-noise-$noise_ratio"


            experiment_subdir=first-$dataset-$label_col-$data_limit-$LM
            cmd="WANDB_PROJECT=crosscultural CUDA_VISIBLE_DEVICES=$gpu python train.py \
                    --project_name crosscultural \
                    --experiment_subdir $experiment_subdir \
                    --LM $LM \
                    --dataset_name $dataset \
                    --label_col $label_col \
                    --weight_decay $weight_decay \
                    --limited_data $data_limit \
                    --EPOCHS $EPOCHS \
                    --stratify \
                    --seed 0 ; "
            concatenated_cmd+=$cmd
            ((dataset_index = dataset_index + num_gpus))
        done 
        echo "Running model with gpu: $gpu, dataset: $dataset, session_name: $SESSION_NAME"
        echo $concatenated_cmd
        screen -dmS "$SESSION_NAME" bash -c "$concatenated_cmd"                  
    done

elif [ "$first_or_second" == "second" ]; then
    echo "Running second stage"
    for gpu_index in "${!gpus[@]}"; do       
        gpu=${gpus[$gpu_index]}
        dataset_index=$gpu_index
        concatenated_cmd=""

        while [ "$dataset_index" -lt "$num_datasets" ]; do

     
            prev_dataset=${datasets[$dataset_index]}
            prev_label_col=${label_cols[$dataset_index]}
            SESSION_NAME="prev-$prev_dataset"

            for dataset_index2 in "${!datasets[@]}"; do      
                dataset=${datasets[$dataset_index2]}
                label_col=${label_cols[$dataset_index2]}
                if [ "$dataset" == "$prev_dataset" ]; then
                    echo "Skipping dataset $dataset as it is the same as the previous dataset $prev_dataset"
                    continue
                fi
                prev_model_dir=first-$prev_dataset-$prev_label_col-$data_limit-$LM
                experiment_subdir=first-$prev_dataset-second-$dataset-$label_col-$data_limit-$LM
                cmd="WANDB_PROJECT=crosscultural CUDA_VISIBLE_DEVICES=$gpu python train.py \
                        --project_name crosscultural \
                        --experiment_subdir $experiment_subdir \
                        --LM $LM \
                        --dataset_name $dataset \
                        --label_col $label_col \
                        --weight_decay $weight_decay \
                        --limited_data $data_limit \
                        --prev_model $prev_model_dir\
                        --EPOCHS $EPOCHS \
                        --stratify \
                        --seed 0 ; "
                concatenated_cmd+=$cmd

                

            done
            ((dataset_index = dataset_index + num_gpus))
        done 
        echo "Running model with gpu: $gpu, dataset: $dataset, session_name: $SESSION_NAME"
        echo $concatenated_cmd
        screen -dmS "$SESSION_NAME" bash -c "$concatenated_cmd"   
    done

else
    echo "Please specify first or second stage"
    exit 1
fi



# wait

#---------------------------Second---------------------------


# > experiments/$experiment_subdir/finetune-$label_col-$noise_ratio/output.txt
#                 --limited_data 1000 \
#                 --DEV \
