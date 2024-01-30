CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 python eval.py \
    --model_path finetuned_models/crpyto-2023-4-llama2-5e-5lr-qkvogud_202401281103 \
    --dataset_path data/fingpt-forecaster-crypto-20230125-20240125-1-4-065 \
    --eval_model peft