# -*- coding: utf-8 -*-
import json
import pandas as pd
import torch
import logging
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


TRAIN_DATA_PATH = "../data/datasets.jsonl"
TEST_DATA_PATH = "../data/datasets_eval.jsonl"

MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None

cls_instruction = (
    "判断以下词汇是否属于政务领域词汇。政务领域词汇要求具备以下特征："
    "领域性：任何与政策实施相关或综合政务处理中专有的词汇，视为政务词汇。通常指向具体的部门机构、项目或政策措施，但也可以是抽象的术语口号、政务概念或战略目标；"
    "语义完整性：检查词汇结构，避免出现过程和行为状态的描述以及词汇是其他词组的一部分的情况。"
)

def load_data(origin_path):
    data = []
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line.strip())
            if "word" in sample and "label" in sample:
                    data.append(sample)
    return data

def dataset_for_train(data, include_sentence=True, cls_instruction=""):
    transformed_data = []
    for sample in data:
        word = sample["word"]
        sentence = sample.get("sentence", "")
        label = sample["label"]

        instruction = cls_instruction
        if include_sentence:
            input_text = f"词汇：{word}\n文本：{sentence}"
        else:
            input_text = f"词汇：{word}"
        output = label

        message = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "label": label
        }
        transformed_data.append(message)
    return transformed_data

def dataset_for_eval(data, include_sentence=True, cls_instruction=""):

    transformed_data = []

    for sample in data:
        word = sample["word"]
        sentence = sample.get("sentence", "")
        label = sample["label"]
        instruction = cls_instruction
        if include_sentence:
            input_text = f"词汇：{word}\n文本：{sentence}"
        else:
            input_text = f"词汇：{word}"

        output = label

        message = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "label": label,
            "word": word,
            "sentence": sentence
        }
        transformed_data.append(message)
    return transformed_data

def process_func(example, index=None, debug_sample=False):

    instruction = example['instruction']
    input_text = example['input']
    output_text = example['output']
    eos_token = tokenizer.eos_token

    # 构建训练时的 prompt 格式
    system_prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n" \
                    f"<|im_start|>user\n{input_text}<|im_end|>\n" \
                    f"<|im_start|>assistant\n"
    user_completion = f"{output_text}{eos_token}"

    full_text = system_prompt + user_completion

    encoding = tokenizer(
        full_text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )

    input_ids = encoding['input_ids'].squeeze()
    attention_mask = encoding['attention_mask'].squeeze()

    labels = input_ids.clone()
    prompt_length = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    labels[:prompt_length] = -100

    pad_token_id = tokenizer.pad_token_id
    labels = torch.where(input_ids == pad_token_id, torch.tensor(-100), labels)

    # if debug_sample and index == 0:
    #     logging.debug(f"\n[Sample {index}]")
    #     logging.debug(f"Instruction: {instruction}")
    #     logging.debug(f"Input Text: {input_text}")
    #     logging.debug(f"Output Text: {output_text}")
    #     logging.debug(f"Full Text: {full_text}")
    #     logging.debug(f"Input IDs: {input_ids.tolist()}")
    #     logging.debug(f"Attention Mask: {attention_mask.tolist()}")
    #     logging.debug(f"Labels: {labels.tolist()}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_experiments = [
    {
        "experiment_id": 1,
        "include_sentence": True,
        "train_model": True,
        "instruction": cls_instruction
    }
]

model_dir = './model/Qwen2.5-7B-Instruct'

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model.eval()

for exp in train_experiments:
    experiment_id = exp["experiment_id"]
    include_sentence = exp["include_sentence"]
    instruction = exp["instruction"]
    OUTPUT_DIR = f"./model/lora_model/experiment{experiment_id}"
    BEST_MODEL_DIR = f"./model/lora_model/experiment{experiment_id}_best"

    train_data = load_data(TRAIN_DATA_PATH)

    transformed_train_data = dataset_for_train(
        train_data,
        include_sentence=include_sentence,
        instruction=instruction
    )

    if MAX_TRAIN_SAMPLES:
        transformed_train_data = transformed_train_data[:MAX_TRAIN_SAMPLES]

    train_ds = Dataset.from_pandas(pd.DataFrame(transformed_train_data).reset_index(drop=True))

    train_dataset = train_ds.map(
        lambda x, idx: process_func(x, idx, debug_sample=True),
        remove_columns=train_ds.column_names,
        num_proc=4,
        with_indices=True
    )

    model_train = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.2,
        bias='none',
    )
    model_train = get_peft_model(model_train, lora_config)
    model_train.print_trainable_parameters()
    model_train.config.use_cache = False


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=5,
        num_train_epochs=3,
        learning_rate=5e-4,
        logging_steps=50,
        save_steps=50,
        eval_strategy="no",
        save_total_limit=3,
        fp16=False,
        # report_to="tensorboard",
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        remove_unused_columns=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model_train,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    model_train.save_pretrained(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)

    test_data = load_data(TEST_DATA_PATH)

    if MAX_TEST_SAMPLES:
        test_data = test_data[:MAX_TEST_SAMPLES]

    transformed_test_data = dataset_for_eval(
        test_data,
        include_sentence=include_sentence,
        instruction=instruction
    )

    test_ds = Dataset.from_pandas(pd.DataFrame(transformed_test_data).reset_index(drop=True))


    model_eval = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model_eval = PeftModel.from_pretrained(model_eval, BEST_MODEL_DIR)
    model_eval.eval()

    results = []
    predicted_labels = []
    true_labels = []

    for i, sample in enumerate(test_ds):
        try:
            word = sample['word']
            sentence = sample.get('sentence', "")
            label = sample['label']
        except KeyError as e:
            continue

        messages = [
            {"role": "system", "content": cls_instruction},
            {"role": "user", "content": sample['input']}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )

        gen_kwargs = {
            "max_length": inputs['input_ids'].shape[1] + 50,
            "do_sample": True,
            "top_k": 1
        }

        try:
            with torch.no_grad():
                outputs = model_eval.generate(**inputs, **gen_kwargs)
                generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        except Exception as e:
            generated_text = "error"

        results.append({
            'Sample': i + 1,
            'Instruction': cls_instruction,
            'Input': sample['input'],
            'Expected Output': label,
            'Generated Output': generated_text
        })

        pred_label = 0 if "不是" in generated_text else 1 if "是" in generated_text else -1
        true_label = 0 if label == "不是" else 1 if label == "是" else -1

        predicted_labels.append(pred_label)
        true_labels.append(true_label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary', zero_division=1
    )

    logging.info(f"\n Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    evaluation_output = {
        'results': results,
        'metrics': {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    }

    result_file = f'test_evaluation_experiment{experiment_id}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_output, f, ensure_ascii=False, indent=4)

    logging.info(f"实验 {experiment_id} 完成。")

