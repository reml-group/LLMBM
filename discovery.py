# -*- coding: utf-8 -*-

import os
from scripts.trie import Trie, process_txt_files
from scripts.candi_n import candi_new_word
from scripts.classify import (
    load_classification_model,
    classify_word,
    find_sentence
)

def build_pkl_from_txt(input_txt_dir, output_pkl_path):
    trie, all_sentences = process_txt_files(input_txt_dir)
    trie.save(output_pkl_path)
    return all_sentences

def candidates(pkl_path, thresholds):
    trie_loaded = Trie.load(pkl_path)
    candidates = candi_new_word(trie_loaded, thresholds)
    return candidates

def classify(candidates, all_sentences, base_model_dir, lora_model_dir, instruction, output_file):

    model, tokenizer = load_classification_model(base_model_dir, lora_model_dir)
    classification_results = []
    for (cand_word, _, _, _) in candidates:
        sentence = find_sentence(all_sentences, cand_word)
        result = classify_word(model, tokenizer, cand_word, sentence, instruction)
        classification_results.append(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in classification_results:
            if item['Predicted Label'] == "是":
                f.write(item['Word'] + "\n")


def main():
    input_txt_dir = './data/your files'
    output_pkl_path = './data/trie.pkl'
    thresholds = {
        2: (30, 15, 2),
        3: (28, 18, 2),
        4: (26, 21, 2),
        5: (24, 24, 2),
        6: (22, 27, 2),
        7: (20, 30, 2),
        8: (18, 33, 2),
        9: (16, 36, 2),
        10: (14, 39, 2),
    }

    lora_model_dir = './model/lora_model'
    base_model_dir = './model/Qwen2.5-7B-Instruct'


    instruction = (
        "判断以下词汇是否属于政务领域词汇。政务领域词汇要求具备以下特征："
        "领域性：任何与政策实施相关或综合政务处理中专有的词汇，视为政务词汇。通常指向具体的部门机构、项目或政策措施，但也可以是抽象的术语口号、政务概念或战略目标；"
        "语义完整性：检查词汇结构，避免出现过程和行为状态的描述以及词汇是其他词组的一部分的情况。"
    )
    final_output_file = './data/new_words.txt'

    all_sentences = build_pkl_from_txt(input_txt_dir, output_pkl_path)

    candi_new_words = candidates(output_pkl_path, thresholds)

    classify(
        candidates=candi_new_words,
        all_sentences=all_sentences,
        base_model_dir=base_model_dir,
        lora_model_dir=lora_model_dir,
        instruction=instruction,
        output_file=final_output_file
    )

if __name__ == '__main__':
    main()
