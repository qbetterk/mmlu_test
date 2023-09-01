#!/bin/bash
set -xue
# A simple script
# python run_clm_no_trainer_instruction_tuning.py \


sample_num=20000
# # export CUDA_VISIBLE_DEVICES=4
# # # # finetune on share gpt
# data_name=alpaca
# # data_name=wizardlm
# data_name=sharegpt_xc
# model_name=llama_7B_${data_name}_${sample_num}

# export CUDA_VISIBLE_DEVICES=0
# model_name=llama_7B_alpaca_${sample_num}_2
# export CUDA_VISIBLE_DEVICES=1
# model_name=llama_7B_alpacalong_${sample_num}_2
# export CUDA_VISIBLE_DEVICES=6
# model_name=llama_7B_sharegpt_xc_${sample_num}
# export CUDA_VISIBLE_DEVICES=4
# model_name=llama_7B_wizardlm_${sample_num}
# export CUDA_VISIBLE_DEVICES=5
# model_name=llama_7B_firstturn_${sample_num}
# export CUDA_VISIBLE_DEVICES=6
# model_name=llama_7B_followturn_${sample_num}
# for ntrain in 5; do
#     for ck in 25; do
#         python run_eval.py --ckpt /export/share/kqian/trainer_fsdp/${model_name}/checkpoint-${ck}000 \
#                         --model ${model_name}/ckp${ck}k --ntrain ${ntrain}
#     done
# done

# export CUDA_VISIBLE_DEVICES=4
# for data_name in wizardlm; do
#     for ntrain in 0 5; do
#         for ck in 50 75 100; do
#             model_name=llama_7B_${data_name}_${sample_num}_ga
#             python run_eval.py --ckpt /export/home/ckpt/trainer_fsdp/${model_name}/checkpoint-${ck}00 \
#                             --model llama_7B_ga_${data_name}_${sample_num}/ckp${ck}00 --ntrain ${ntrain}
#         done
#     done
# done

# # # # # mmlu for deepspeed ckpt
# export CUDA_VISIBLE_DEVICES=4
# for data_name in alpacagpt4; do
#     for ck in 1; do
#         for ntrain in 0 5; do
#             model_name=llama_7B_${data_name}_${sample_num}
#             python run_eval.py --ckpt /export/home/ckpt/instruction-tuning-models/${model_name}/epoch_${ck} \
#                             --model llama_7B_ds_${data_name}_${sample_num}/epoch${ck} --ntrain ${ntrain}
#         done
#     done
# done
# export CUDA_VISIBLE_DEVICES=5
# for data_name in alpacagpt4; do
#     for ck in 2; do
#         for ntrain in 0 5; do
#             model_name=llama_7B_${data_name}_${sample_num}
#             python run_eval.py --ckpt /export/home/ckpt/instruction-tuning-models/${model_name}/epoch_${ck} \
#                             --model llama_7B_ds_${data_name}_${sample_num}/epoch${ck} --ntrain ${ntrain}
#         done
#     done
# done
# export CUDA_VISIBLE_DEVICES=6
# for data_name in alpacagpt4; do
#     for ck in 3; do
#         for ntrain in 0 5; do
#             model_name=llama_7B_${data_name}_${sample_num}
#             python run_eval.py --ckpt /export/home/ckpt/instruction-tuning-models/${model_name}/epoch_${ck} \
#                             --model llama_7B_ds_${data_name}_${sample_num}/epoch${ck} --ntrain ${ntrain}
#         done
#     done
# done


# # # # # mmlu for alpaca ckpt
# base_model=llama2_7B
# base_model=llama_7B
base_model=llama_7B_llamatokenizer
sample_num=52000
export CUDA_VISIBLE_DEVICES=2
data_name=sharegpt_qy_vic
# data_name=sharegpt_xc
# data_name=alpaca
# data_name=alpaca_combine_26k
for ntrain in 0; do
    model_name=${base_model}_${data_name}_${sample_num}
    python run_eval.py --ckpt /export/home/ckpt/alpaca_code/${model_name} \
                    --model ${base_model}_alpacacode_${data_name}_${sample_num} --ntrain ${ntrain}
done



# # # # # # mmlu for deepspeed ckpt
# # base_model=llama2_7B
# # base_model=llama_7B
# base_model=llama_7B_llamatokenizer
# sample_num=52000
# export CUDA_VISIBLE_DEVICES=6
# # data_name=sharegpt_qy_vic
# # data_name=sharegpt_xc
# # data_name=alpaca
# data_name=alpaca_combine_6k
# for ck in 3; do
#     for ntrain in 5; do
#         model_name=${base_model}_${data_name}_${sample_num}
#         python run_eval.py --ckpt /export/home/ckpt/instruction-tuning-models/${model_name}/epoch_${ck} \
#                         --model ${base_model}_ds_${data_name}_${sample_num}/epoch${ck} --ntrain ${ntrain}
#     done
# done


# # # # # # mmlu for deepspeed2 ckpt
# sample_num=52000
# export CUDA_VISIBLE_DEVICES=7
# data_name=alpaca
# for ck in 2 3 1; do
#     for ntrain in 0 5; do
#         model_name=llama2_7B_ds2_${data_name}_${sample_num}
#         python run_eval.py --ckpt /export/home/ckpt/instruction-tuning-models/${model_name}/epoch_${ck} \
#                         --model llama2_7B_ds2_${data_name}_${sample_num}/epoch${ck} --ntrain ${ntrain}
#     done
# done


# for ntrain in 0 5; do
#     for ck in 20; do
#         python run_eval.py --ckpt gpt2-xl \
#                         --model gpt2-xl --ntrain ${ntrain}
#     done
# done

# export CUDA_VISIBLE_DEVICES=2
# model_name=llama_7B_alpaca_${sample_num}
# python run_eval.py --ckpt /export/share/kqian/instruction-tuning-models/${model_name} \
#                 --model ${model_name}/ --ntrain 0
                
# python run_eval_openchat.py \
#     --ntrain 0 \
#     --model_name_or_path /export/share/kqian/trainer_fsdp/${model_name}/checkpoint-20000 \
#     --eval_batch_size 8 \
#     --use_chat_format

# python run_eval.py --ckpt /export/share/kqian/trainer_fsdp/${model_name}/checkpoint-14000 \
#                 --model ${model_name}

# export CUDA_VISIBLE_DEVICES=7
# sample_num=20000
# # data_name=alpaca
# # # data_name=sharegpt_xc
# # model_name=dp_llama_7B_${data_name}_${sample_num}
# # python run_eval.py --ckpt /export/share/kqian/instruction-tuning-models-bkp/llama_7B_${data_name}_${sample_num}_length \
# #                 --model ${model_name}
# data_name=sharegpt_xc
# model_name=dp_llama_7B_${data_name}_${sample_num}
# python run_eval.py --ckpt /export/share/kqian/instruction-tuning-models-bkp/llama_7B_${data_name}_${sample_num}_first_turn \
#                 --model ${model_name}
