CUDA_VISIBLE_DEVICES=0,1 python run_src/do_generate.py \
    --dataset_name LOGICCAT \
    --test_json_filename test_all \
    --model_ckpt /home/admin/Temp/Model/Qwen3-1.7B \
    --note default \
    --num_rollouts 4
