PROJECT_DIR=''
SOURCE_CODE_DIR=''
MODEL_STORAGE_DIR=$PROJECT_DIR/models #where pre-trained huggingface models are stored
datetime=`date +%Y-%m-%d_%H-%M-%S`

# system hyperparamters
walltime=15:00:00
nNode=1
nCPU=12
nGPU=4
gpuMem=32gb
mem=100gb

quicktest=false
model_name='t5-base' 
# -------------------------------------------------------------------------------------- #
# below are settings for hyperformer config file
# -------------------------------------------------------------------------------------- #
model_name_or_path="$MODEL_STORAGE_DIR/$model_name"
tokenizer_name="$MODEL_STORAGE_DIR/$model_name"
output_dir=''
overwrite_output_dir=true
cache_dir=''

# model/architecture hyperparameters
learning_rate='0.0003' #bash cannot natively handle float, so make it string and convert later
max_source_length=128
max_target_length=128
val_max_target_length=128
test_max_target_length=128
warmup_steps=500
label_smoothing='0.1' #bash cannot natively handle float, so make it string and convert later
per_device_train_batch_size=36
per_device_eval_batch_size=36
temperature=10
non_linearity='gelu_new'
dropout_rate='0.1' #bash cannot natively handle float, so make it string and convert later

if [ $quicktest = true ] 
then
    eval_steps=10
    save_steps=10
    logging_steps=10
    max_steps=30
    n_train=300 #specify how many training datapoints to use, -1 means all of them
else
    eval_steps=1000
    save_steps=1000
    logging_steps=1000
    max_steps=30000
    n_train=-1 #specify how many training datapoints to use, -1 means all of them
fi
logging_first_step=true
save_total_limit=1
do_train=true
do_eval=true
do_test=true
split_validation_test=false
load_best_model_at_end=true

#evaluation settings
evaluation_strategy='steps'
metric_for_best_model='average_metrics'
greater_is_better=true
predict_with_generate=true

# Hyperformer related settings
tasks="americasnlp2021-es_XX-aym_XX americasnlp2021-es_XX-gn_XX americasnlp2021-es_XX-quy_XX americasnlp2021-es_XX-shp_XX americasnlp2021-es_XX-cni_XX" 
eval_tasks="americasnlp2021-es_XX-aym_XX americasnlp2021-es_XX-gn_XX americasnlp2021-es_XX-quy_XX americasnlp2021-es_XX-shp_XX americasnlp2021-es_XX-cni_XX"
task_embedding_dim=64
train_adapters=true
train_task_embeddings=true
reduction_factor=32
projected_task_embedding_dim=64
conditional_layer_norm=true
unfreeze_lm_head=false
unfreeze_layer_norms=true
efficient_unique_hyper_net=true
adapter_config_name='meta-adapter'
# -------------------------------------------------------------------------------------- #
# above are settings for hyperformer config file
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
# dynamically create config file
# -------------------------------------------------------------------------------------- #
config_folder_name=''
config_file_name='config.json'
python3 $SOURCE_CODE_DIR/hyperformer/scripts/create_config_file.py --config_folder_name $config_folder_name \
                            --config_file_name $config_file_name \
                            --model_name_or_path $model_name_or_path \
                            --tokenizer_name $tokenizer_name \
                            --output_dir $output_dir \
                            --overwrite_output_dir $overwrite_output_dir \
                            --cache_dir $cache_dir \
                            --learning_rate $learning_rate \
                            --max_source_length $max_source_length \
                            --max_target_length $max_target_length \
                            --val_max_target_length $val_max_target_length \
                            --test_max_target_length $test_max_target_length \
                            --warmup_steps $warmup_steps \
                            --per_device_train_batch_size $per_device_train_batch_size \
                            --label_smoothing $label_smoothing \
                            --per_device_eval_batch_size $per_device_eval_batch_size \
                            --temperature $temperature \
                            --non_linearity $non_linearity \
                            --dropout_rate $dropout_rate \
                            --eval_steps $eval_steps \
                            --save_steps $save_steps \
                            --logging_first_step $logging_first_step \
                            --logging_steps $logging_steps \
                            --save_total_limit $save_total_limit \
                            --do_train $do_train \
                            --do_eval $do_eval \
                            --do_test $do_test \
                            --split_validation_test $split_validation_test \
                            --load_best_model_at_end $load_best_model_at_end \
                            --n_train $n_train \
                            --max_steps $max_steps \
                            --evaluation_strategy $evaluation_strategy \
                            --metric_for_best_model $metric_for_best_model \
                            --greater_is_better $greater_is_better \
                            --predict_with_generate $predict_with_generate \
                            --tasks $tasks \
                            --eval_tasks $eval_tasks \
                            --train_adapters $train_adapters \
                            --train_task_embeddings $train_task_embeddings \
                            --task_embedding_dim $task_embedding_dim \
                            --reduction_factor $reduction_factor \
                            --projected_task_embedding_dim $projected_task_embedding_dim \
                            --conditional_layer_norm $conditional_layer_norm \
                            --unfreeze_lm_head $unfreeze_lm_head \
                            --unfreeze_layer_norms $unfreeze_layer_norms \
                            --efficient_unique_hyper_net $efficient_unique_hyper_net \
                            --adapter_config_name $adapter_config_name 

# -------------------------------------------------------------------------------------- #
# main
# -------------------------------------------------------------------------------------- #
pretrained_model_path=$MODEL_STORAGE_DIR/$model_name
out_file_path=''
err_file_path=''

qsub -o $out_file_path \
    -e $err_file_path \
    -l walltime=$walltime,select=$nNode:ncpus=$nCPU:ngpus=$nGPU:gpu_mem=$gpuMem:mem=$mem \
    -A $groupName \
    -N $datetime-$trainWithTrainAndPartialDev \
    -v config_file_path=$config_folder_name/$config_file_name,nGPU=$nGPU \
    $SOURCE_CODE_DIR/hyperformer/scripts/runner.sh