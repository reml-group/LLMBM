import os
import pickle
import math
from tqdm import tqdm
from trie import Trie


def calculate_entropy(neighbors):
    total = sum(neighbors.values())
    entropy = -sum((count / total) * math.log(count / total, 2) for count in neighbors.values()) if total > 0 else 0
    return entropy


def calculate_pmi(trie, ngram, total_count):
    ngram_prob = trie.search(ngram) / total_count
    if ngram_prob == 0:
        return -float('inf')
    char_probs = [trie.search(char) / total_count for char in ngram]
    individual_prob = math.prod(char_probs)
    pmi = math.log(ngram_prob / individual_prob, 2) if individual_prob > 0 else -float('inf')
    return pmi


def traverse_nodes(node):
    yield node
    for child in node.children.values():
        yield from traverse_nodes(child)


def candi_new_word(trie, thresholds):
    candidates = []
    total_count = trie.get_total_count()
    total_nodes = sum(1 for _ in traverse_nodes(trie.root))

    with tqdm(total=total_nodes, desc="Processing Trie nodes", unit="node") as pbar:
        def traverse(node, prefix=""):
            n = len(prefix)
            if n in thresholds:
                freq_threshold, pmi_threshold, entropy_threshold = thresholds[n]
                if node.is_end_of_word and node.frequency >= freq_threshold:
                    pmi = calculate_pmi(trie, prefix, total_count)
                    if pmi >= pmi_threshold:
                        left_entropy = calculate_entropy(node.prev_chars)
                        right_entropy = calculate_entropy(node.next_chars)
                        min_entropy = min(left_entropy, right_entropy)
                        if min_entropy >= entropy_threshold:
                            candidates.append((prefix, node.frequency, pmi, min_entropy))
            for char, child_node in node.children.items():
                traverse(child_node, prefix + char)
            pbar.update(1)

        traverse(trie.root)
    return candidates

def save_results_to_file(new_words, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, freq, pmi, entropy in new_words:
            file.write(f"Word: {word}, Frequency: {freq}, PMI: {pmi}, Entropy: {entropy}\n")


def process_directories(directory, thresholds):
    all_candidates = {}
    num_subdirectories = sum(len(dirs) for _, dirs, _ in os.walk(directory))
    with tqdm(total=num_subdirectories, desc="Processing Subdirectories", unit="folder") as pbar:
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.pkl'):
                    pkl_file_path = os.path.join(subdir, file)
                    trie = Trie.load(pkl_file_path)
                    candidates = candi_new_word(trie, thresholds)
                    results_filename = os.path.join(subdir, 'results.txt')
                    save_results_to_file(candidates, results_filename)
                    all_candidates[pkl_file_path] = candidates
            pbar.update(1)
    return all_candidates



if __name__ == '__main__':
    directory = '../data/2024/08'
    # 自行设定阈值
    thresholds = {n: (30 - 2*(n-2), 15 + 3*(n-2), 2) for n in range(2, 11)}
    # thresholds = {
    #     2: (30, 15, 2),
    #     3: (28, 18, 2),
    #     4: (26, 21, 2),
    #     5: (24, 24, 2),
    #     6: (22, 27, 2),
    #     7: (20, 30, 2),
    #     8: (18, 33, 2),
    #     9: (16, 36, 2),
    #     10: (14, 39, 2),
    # }
    process_directories(directory, thresholds)

