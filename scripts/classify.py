# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_classification_model(base_model_dir, lora_model_dir):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir, use_fast=False, trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_dir
    )
    lora_model.eval()
    return lora_model, tokenizer


def classify_word(model, tokenizer, word, sentence, instruction):
    if sentence and sentence != "无":
        input_text = f"文本：{sentence}\n词汇：{word}"
    else:
        input_text = f"文本：无\n词汇：{word}"

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text}
    ]

    try:
        encoded_input = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=True
        )
    except Exception as e:
        return {
            'Word': word,
            'Sentence': sentence,
            'Generated Output': "",
            'Predicted Label': "未分类"
        }

    # 设置生成参数
    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": True,
        "top_k": 1
    }

    try:
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                **gen_kwargs
            )
    except Exception as e:
        return {
            'Word': word,
            'Sentence': sentence,
            'Generated Output': "",
            'Predicted Label': "未分类"
        }

    try:
        generated_text = tokenizer.decode(
            generated_tokens[0][encoded_input['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
    except Exception as e:
        generated_text = ""

    if "不是" in generated_text:
        pred_label = "不是"
    elif "是" in generated_text:
        pred_label = "是"
    else:
        pred_label = "未分类"

    return {
        'Word': word,
        'Sentence': sentence,
        'Generated Output': generated_text.strip(),
        'Predicted Label': pred_label
    }


def load_new_words(new_words_file):
    new_words = []
    with open(new_words_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                new_words.append(word)
    return new_words


def load_source_sentences(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


def find_sentence(sentences, word):
    for sentence in sentences:
        if len(sentence) > 150:
            continue
        if word in sentence and sentence.count(word) == 1:
            return sentence
    return "无"


NEW_WORDS_FILE = "results.txt"
DATA_FILE = "source.txt"
OUTPUT_FILE = "new_words.txt"


# ------------------------------
# 主流程：分类模块
# ------------------------------
def main():
    BASE_MODEL_DIR = '../model/Qwen2.5-7B-Instruct'

    LORA_MODEL_DIR = '../model/lora_model'

    INSTRUCTION = (
        "判断以下词汇是否属于政务领域词汇。政务领域词汇要求具备以下特征："
        "领域性：任何与政策实施相关或综合政务处理中专有的词汇，视为政务词汇。通常指向具体的部门机构、项目或政策措施，但也可以是抽象的术语口号、政务概念或战略目标；"
        "语义完整性：检查词汇结构，避免出现过程和行为状态的描述以及词汇是其他词组的一部分的情况。"
    )

    new_words = load_new_words(NEW_WORDS_FILE)

    sentences = load_source_sentences(DATA_FILE)

    new_word_sentences = []
    for word in new_words:
        sentence = find_sentence(sentences, word)
        new_word_sentences.append(sentence)

    model, tokenizer = load_classification_model(BASE_MODEL_DIR, LORA_MODEL_DIR)

    classification_results = []
    for word, sentence in tqdm(zip(new_words, new_word_sentences), total=len(new_words), desc="Classifying new words", unit="word"):
        result = classify_word(model, tokenizer, word, sentence, INSTRUCTION)
        classification_results.append(result)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in classification_results:
            f.write("Word: {}\n".format(result['Word']) + "Predicted Label: {}\n".format(result['Predicted Label']))
            f.write("\n")


if __name__ == '__main__':
    main()
