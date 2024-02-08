CUDA_VISIBLE_DEVICES=7 python eval.py \
    --model_path finetuned_models/fingpt-forecaster-dow-30-20221231-20231231-1-4-08-modified_202402022043 \
    --dataset_path data/fingpt-forecaster-dow-30-20221231-20231231-1-4-08-modified \
    --eval_model peft