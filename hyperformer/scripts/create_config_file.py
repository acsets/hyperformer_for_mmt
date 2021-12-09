import argparse
import json
parser = argparse.ArgumentParser()

parser.add_argument('--config_folder_name', type=str)
parser.add_argument('--config_file_name', type=str)

parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--tokenizer_name', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--overwrite_output_dir', type=bool)
parser.add_argument('--cache_dir', type=str)

# model/architecture hyperparameters
parser.add_argument('--learning_rate', type=str)
parser.add_argument('--max_source_length', type=int)
parser.add_argument('--max_target_length', type=int)
parser.add_argument('--val_max_target_length', type=int)
parser.add_argument('--test_max_target_length', type=int)
parser.add_argument('--warmup_steps', type=int) 
parser.add_argument('--label_smoothing', type=str) 
parser.add_argument('--per_device_train_batch_size', type=int)
parser.add_argument('--per_device_eval_batch_size', type=int)
parser.add_argument('--temperature', type=int) 
parser.add_argument('--non_linearity', type=str)
parser.add_argument('--dropout_rate', type=str)

# recording frequency and training hyperparamters
parser.add_argument('--eval_steps', type=int)
parser.add_argument('--save_steps', type=int)
parser.add_argument('--logging_first_step', type=bool) 
parser.add_argument('--logging_steps', type=int) 
parser.add_argument('--save_total_limit', type=int)
parser.add_argument('--do_train', type=bool)
parser.add_argument('--do_eval', type=bool)
parser.add_argument('--do_test', type=bool)
parser.add_argument('--split_validation_test', type=bool)
parser.add_argument('--load_best_model_at_end', type=bool)
parser.add_argument('--n_train', type=int)
parser.add_argument('--max_steps', type=int)

# evaluation settings
parser.add_argument('--evaluation_strategy', type=str)
parser.add_argument('--metric_for_best_model', type=str)
parser.add_argument('--greater_is_better', type=bool)
parser.add_argument('--predict_with_generate', type=bool)

# Hyperformer related settings
# bash list to python argparse: https://www.kite.com/python/answers/how-to-pass-a-list-as-an-argument-using-argparse-in-python
parser.add_argument("--tasks", nargs="+")
parser.add_argument("--eval_tasks", nargs="+")
parser.add_argument('--train_adapters', type=bool)
parser.add_argument('--train_task_embeddings', type=bool)
parser.add_argument('--task_embedding_dim', type=int)
parser.add_argument('--reduction_factor', type=int)
parser.add_argument('--projected_task_embedding_dim', type=int)
parser.add_argument('--conditional_layer_norm', type=bool)
parser.add_argument('--unfreeze_lm_head', type=bool)
parser.add_argument('--unfreeze_layer_norms', type=bool)
parser.add_argument('--efficient_unique_hyper_net', type=bool)
parser.add_argument('--adapter_config_name', type=str)

args, unknown = parser.parse_known_args()
config_dict = {
    # paths
    "model_name_or_path": args.model_name_or_path,
    "tokenizer_name": args.tokenizer_name,
    "output_dir": args.output_dir,
    "overwrite_output_dir": args.overwrite_output_dir,
    "cache_dir": args.cache_dir,
    
    # learning hyperparameters
    "learning_rate": float(args.learning_rate),
    "max_source_length": args.max_source_length,
    "max_target_length": args.max_target_length,
    "val_max_target_length": args.val_max_target_length,
    "test_max_target_length": args.test_max_target_length,
    "warmup_steps": args.warmup_steps,
    "label_smoothing": float(args.label_smoothing),
    "per_device_train_batch_size": args.per_device_train_batch_size,
    "per_device_eval_batch_size": args.per_device_eval_batch_size,
    "temperature": args.temperature,
    "non_linearity": args.non_linearity,
    "dropout_rate": float(args.dropout_rate),

    # recording hyperparamters
    "eval_steps": args.eval_steps,
    "save_steps": args.save_steps,
    "logging_first_step": args.logging_first_step,
    "logging_steps": args.logging_steps,
    "save_total_limit": args.save_total_limit,
    
    # procedure settings
    "do_train": args.do_train,
    "do_test": args.do_test,
    "do_eval": args.do_eval,
    "split_validation_test": args.split_validation_test,
    "load_best_model_at_end": args.load_best_model_at_end,
    "n_train": args.n_train,
    "max_steps": args.max_steps,

    # evaluation related
    "evaluation_strategy": args.evaluation_strategy,
    "metric_for_best_model": args.metric_for_best_model,
    "greater_is_better": args.greater_is_better,
    "predict_with_generate": args.predict_with_generate,

    # Hyperformer related
    "tasks": args.tasks,
    "eval_tasks": args.eval_tasks,
    "task_embedding_dim": args.task_embedding_dim,
    "train_adapters": args.train_adapters,
    "train_task_embeddings": args.train_task_embeddings,
    "reduction_factor": args.reduction_factor,
    "projected_task_embedding_dim": args.projected_task_embedding_dim,
    "conditional_layer_norm": args.conditional_layer_norm,
    "unfreeze_lm_head": args.unfreeze_lm_head,
    "unfreeze_layer_norms": args.unfreeze_layer_norms,
    "efficient_unique_hyper_net": args.efficient_unique_hyper_net,
    "adapter_config_name": args.adapter_config_name,
}

with open(f"{args.config_folder_name}/{args.config_file_name}", "w") as outfile:
    json.dump(config_dict, outfile)