import re
import collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)

    # Iterate through the vocabulary and count character pairs
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq

    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    # Merge character pair in all words in the vocabulary
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out

# Vocabulary with initial subword units
# I made it randomly with this text to make it changable and the coder can see
# each time the changes in the tokens
import random
text = "مرحبا بك في هذه المقالة المفيدة"

words_list = text.split()

new_text = [" ".join(word) for word in words_list]

vocab = {word:random.randint(1,100) for word in new_text}

num_merges = 10

# Perform BPE tokenization for the specified number of merges
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
