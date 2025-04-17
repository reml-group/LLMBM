# LLMBM

This repository contains the code for **Enhancing Governmental New Word Discovery with Large Language Models**, as well as the **GovDict** and **GovTerm** datasets.

## Corpus

We begin with the [PolicyBank](https://pan.baidu.com/s/1ca9kpHGxmgeo1mB1qr3cCA) as our raw corpus, and supplement it with additional government‑related texts scraped from various public websites. After preprocessing, we obtain the **GovTerm**  dataset—an instruction‑tuning collection for large models to improve their knowledge of governmental terminology and their ability to extract new terms. We also provide **GovDict**— a glossary of government terms with partial explanations and a plain list of government terms without explanations.

## Usage

### Getting Started

```bash
pip install -r requirements.txt
```

### Model

We use **QWEN2.5‑7B** as our base model. You can download it from [Qwen2.5‑7B](https://huggingface.co/Qwen/Qwen2.5-7B) and place it under `model/Qwen2.5-7B-Instruct`.

### Dataset

Under `/data` you will find:

- **GovTerm**: 

  the fine‑tuning and evaluation data.

  - The fine‑tuning set includes three task types: ***classification***, ***correction***, and ***explanation***. You can select task types and proportions before or during training.`datasets`          
  - The evaluation set contains only the ***classification*** task.`datasets_eval`

- **GovDict**: 

  Our curated glossary of government terms with explanations.`govdict.json`  

  The plain list of government terms.`govdict.txt` 

## Discovering New Words

You can unzip your documents by year/month into the `data` directory, following the [GOVGLM](https://github.com/reml-group/GovGLM) documents. We also include a sample test case `2024-08.txt`. Run `discovery.py`to extract new words from that document.

## Training and Evaluation

### Training

After preparing your model files and datasets, run`scripts/train.py` to start training the Lora model.The configurations used in our experiments were:

#### Qwen

```bash
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
    gradient_checkpointing=False,
    max_grad_norm=1.0,
    remove_unused_columns=True,
)
```

#### LoRA

```bash
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.2,
    bias='none',
)
```

Our experiments used full‑precision`fp32`. If your hardware supports it, you may switch to `bf16` to significantly speed up training with minimal impact on accuracy.

### Evaluation

We provide built‑in evaluation routines to measure the model’s classification accuracy on our selected data. To assess overall new‑word discovery performance and coverage against our term lists, run `eval.py` under `/scripts`.

