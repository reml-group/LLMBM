# -*- coding: utf-8 -*-

import logging
import os

def load_evaluation_dictionary(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r', encoding='utf-8') as f:
        eval_dict = set()
        for line in f:
            parts = line.strip().split(',')
            word = parts[0].strip()
            if word:
                eval_dict.add(word)
    return eval_dict

def load_filtering_dictionary(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r', encoding='utf-8') as f:
        filtering_dict = set()
        for line in f:
            parts = line.strip().split(',')
            word = parts[0].strip()
            if word:
                filtering_dict.add(word)
    return filtering_dict

def evaluate_classification(newword_dict, evaluation_dict):
    newword_set= newword_dict
    evaluation_set = evaluation_dict

    TP = len(newword_set & evaluation_set)
    FP = len(newword_set - evaluation_set)
    FN = len(evaluation_set - newword_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logging.info(f"评估结果:")
    logging.info(f"TP: {TP}")
    logging.info(f"FP: {FP}")
    logging.info(f"FN: {FN}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    evaluation_metrics = {
        'True Positives (TP)': TP,
        'False Positives (FP)': FP,
        'False Negatives (FN)': FN,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    return evaluation_metrics

def main():

    newword_file = "../data/new_words.txt"
    eval_file = "../data/dict_eval.txt"

    newword_set = load_evaluation_dictionary(newword_file)
    evaluation_set = load_evaluation_dictionary(eval_file)

    metrics = evaluate_classification(newword_set, evaluation_set)

    print("评估完成:", metrics)


if __name__ == '__main__':
    main()
