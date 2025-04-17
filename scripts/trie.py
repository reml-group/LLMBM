import os
import re
import pickle
import sys
from tqdm import tqdm
import copy

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.next_chars = {}
        self.prev_chars = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, sentence, right_char=None, left_char=None):
        current = self.root
        for char in sentence:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
        current.frequency += 1

        if right_char:
            if right_char in current.next_chars:
                current.next_chars[right_char] += 1
            else:
                current.next_chars[right_char] = 1

        if left_char:
            if left_char in current.prev_chars:
                current.prev_chars[left_char] += 1
            else:
                current.prev_chars[left_char] = 1

    # def display(self, node=None, prefix=""):
    #     if node is None:
    #         node = self.root
    #     for char, child_node in node.children.items():
    #         if child_node.is_end_of_word:
    #             print(f"{prefix + char}: {child_node.frequency}")
    #             if child_node.prev_chars:
    #                 print(f"    左邻字: {child_node.prev_chars}")
    #             if child_node.next_chars:
    #                 print(f"    右邻字: {child_node.next_chars}")
    #         self.display(child_node, prefix + char)

    def merge(self, other_trie):
        self._merge_nodes(self.root, other_trie.root)

    def _merge_nodes(self, node1, node2):
        node1.frequency += node2.frequency
        node1.is_end_of_word = node1.is_end_of_word or node2.is_end_of_word

        for char, freq in node2.next_chars.items():
            if char in node1.next_chars:
                node1.next_chars[char] += freq
            else:
                node1.next_chars[char] = freq

        for char, freq in node2.prev_chars.items():
            if char in node1.prev_chars:
                node1.prev_chars[char] += freq
            else:
                node1.prev_chars[char] = freq

        for char, child_node2 in node2.children.items():
            if char in node1.children:
                self._merge_nodes(node1.children[char], child_node2)
            else:
                node1.children[char] = copy.deepcopy(child_node2)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            loaded_trie = pickle.load(f)
        return loaded_trie

    def get_total_count(self):
        return sum(child.frequency for child in self.root.children.values())

    def search(self, word):
        current = self.root
        for char in word:
            if char in current.children:
                current = current.children[char]
            else:
                return 0
        return current.frequency if current.is_end_of_word else 0


# 分句与清洗
def split_sentences(content):
    sentences = re.split(
        r'[。！？,.;:!?，；：——]|(?:\d+\.\d+|\d+\.|\d+、|[a-zA-Z]+\)|\([a-zA-Z]+\)|（[一二三四五六七八九十]+）|\([一二三四五六七八九十]+\)|[一二三四五六七八九十]+、|〈[一二三四五六七八九十]+〉|〔\d+〕|[\d+〕|[一二三四五六七八九十]+）|[一二三四五六七八九十]+）)',
        content
    )

    punctuation_to_remove = r'[（）()【】｛｝{}《》〈〉‹›«»“”‘’""\'\'\[\]<>、_-]'
    sentences = [
        re.sub(punctuation_to_remove, '', sentence).strip()
        for sentence in sentences
        if sentence.strip()
    ]

    sentences = [re.sub(r'\s+', '', sentence) for sentence in sentences]

    return sentences

def generate_ngrams(sentence, min_len, max_len):
    ngrams = []
    length = len(sentence)
    max_len = min(max_len, length)
    for n in range(min_len, max_len + 1):
        for i in range(length - n + 1):
            ngram = sentence[i:i + n]
            left_char = sentence[i - 1] if i > 0 else None
            right_char = sentence[i + n] if i + n < length else None
            ngrams.append((ngram, right_char, left_char))
    return ngrams

def process_txt_files(path):
    trie = Trie()
    all_sentences = []

    if os.path.isfile(path):
        txt_paths = [path]
    elif os.path.isdir(path):
        txt_paths = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith('.txt')
        ]
    else:
        raise ValueError(f"{path} ")

    for file_path in tqdm(txt_paths, desc=f"processing {path}", unit="文件"):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
            except:
                continue

        sentences = split_sentences(content)
        all_sentences.extend(sentences)

        for sentence in sentences:
            if not sentence:
                continue
            max_len = min(10, len(sentence))
            for ngram, right_char, left_char in generate_ngrams(sentence, 1, max_len):
                trie.insert(ngram, right_char, left_char)

    return trie, all_sentences

def process_directory(directory):
    if not os.path.exists(directory):
        return

    trie = process_txt_files(directory)
    trie_file_path = os.path.basename(directory) + '.pkl'
    trie.save(trie_file_path)

def main():
    directories = []
    start_year = 2021
    end_year = 2024
    data_directory = './data'

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_str = f"{month:02}"
            directory = os.path.join(data_directory, f"{year}/{month_str}")
            if os.path.exists(directory):
                directories.append(directory)

    if not directories:
        sys.exit(1)

    for directory in directories:
        process_directory(directory)


if __name__ == '__main__':
    main()
