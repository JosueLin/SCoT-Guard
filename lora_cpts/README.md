---
library_name: peft
license: other
base_model: /root/autodl-fs/llm/Qwen2.5-1.5B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: rag_3consistent_4w_0513
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# rag_3consistent_4w_0513

This model is a fine-tuned version of [/root/autodl-fs/llm/Qwen2.5-1.5B-Instruct](https://huggingface.co//root/autodl-fs/llm/Qwen2.5-1.5B-Instruct) on the rag_3consistent_4w dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 5.0

### Training results



### Framework versions

- PEFT 0.15.1
- Transformers 4.51.3
- Pytorch 2.5.1+cu124
- Datasets 3.5.0
- Tokenizers 0.21.1