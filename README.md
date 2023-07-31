# [TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a Domain-Specific Expert in Transportation Safety](https://arxiv.org/abs/2307.15311)
Large Language Models (LLMs) have shown remarkable effectiveness in various general-domain natural language processing (NLP) tasks. However, their performance in transportation safety domain tasks has been suboptimal, primarily attributed to the requirement for specialized transportation safety expertise in generating accurate responses. To address this challenge, we introduce TrafficSafetyGPT, a novel LLAMA-based model, which has undergone supervised fine-tuning using TrafficSafety-2K dataset which has human labels from government produced guiding books and ChatGPT-generated instruction-output pairs.

Ou Zheng<sup>1</sup>, Mohamed Abdel-Aty<sup>2</sup>, Dongdong Wang<sup>3</sup>, Chenzhu Wang<sup>4</sup>, Shengxuan Ding<sup>5</sup>

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/HUANGLIZI/ChatDoctor/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Page](https://img.shields.io/badge/Web-Page-yellow)](https://www.yunxiangli.top/ChatDoctor/) 
## Resources List

UCF Traffic Safety data from NSTHA Model Minimum Uniform Crash Criteria (MMUCC) Guideline Fourth edition, FHWA The Highway Safety Manual (HSM)[TrafficSafety-2K ](https://docs.google.com/spreadsheets/d/1PTztJw3pq1Eau0ZM2uL7N_yilv6H36QC/edit?usp=sharing&ouid=105044560872530659805&rtpof=true&sd=true).

Stanford Alpaca data for basic conversational capabilities. [Alpaca link](https://github.com/Kent0n-Li/ChatDoctor/blob/main/alpaca_data.json).

 ## How to fine-tuning

 ```python
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./HealthCareMagic-100k.json \
    --bf16 True \
    --output_dir pretrained \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
    --tf32 True
 ```
 
 
Fine-tuning with Lora 
```python
WORLD_SIZE=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 --master_port=4567 train_lora.py \
  --base_model './weights-alpaca/' \
  --data_path 'HealthCareMagic-100k.json' \
  --output_dir './lora_models/' \
  --batch_size 32 \
  --micro_batch_size 4 \
  --num_epochs 1 \
  --learning_rate 3e-5 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
 ```
 
 ## How to inference
 You can build a ChatDoctor model on your own machine and communicate with it.
 ```python
python chat.py
 ```

## Reference

TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a Domain-Specific Expert in Transportation Safety
```
@misc{zheng2023trafficsafetygpt,
      title={TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a Domain-Specific Expert in Transportation Safety}, 
      author={Ou Zheng and Mohamed Abdel-Aty and Dongdong Wang and Chenzhu Wang and Shengxuan Ding},
      year={2023},
      eprint={2307.15311},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
