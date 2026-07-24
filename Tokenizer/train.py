import json
from collections import Counter


DATASET = "TinyStories.txt"
VOCAB_SIZE = 4096


# ----------------------------
# Load dataset
# ----------------------------

text = open(DATASET, "r", encoding="utf-8").read()


# ----------------------------
# Create initial corpus
# Each word becomes list of characters
# ----------------------------

words = text.split()

corpus = []

for word in words:
    corpus.append(list(word))


# ----------------------------
# Initial vocabulary
# ----------------------------

vocab = set()

for word in corpus:
    for char in word:
        vocab.add(char)


# Give IDs to initial tokens

token_to_id = {}

for idx, token in enumerate(sorted(vocab)):
    token_to_id[token] = idx


next_token_id = len(token_to_id)



# ----------------------------
# Count adjacent pairs
# ----------------------------

def get_pair_counts(corpus):

    pairs = Counter()

    for word in corpus:

        for i in range(len(word)-1):

            pair = (word[i], word[i+1])

            pairs[pair] += 1

    return pairs



# ----------------------------
# Merge pair in corpus
# ----------------------------

def merge_pair(corpus, pair):

    new_corpus = []

    first, second = pair

    merged = first + second


    for word in corpus:

        new_word = []

        i = 0

        while i < len(word):

            if (
                i < len(word)-1
                and word[i] == first
                and word[i+1] == second
            ):
                new_word.append(merged)
                i += 2

            else:
                new_word.append(word[i])
                i += 1


        new_corpus.append(new_word)


    return new_corpus



# ----------------------------
# BPE Training
# ----------------------------

merges = []


while len(token_to_id) < VOCAB_SIZE:


    pair_counts = get_pair_counts(corpus)


    if len(pair_counts) == 0:
        break


    best_pair = pair_counts.most_common(1)[0][0]


    new_token = best_pair[0] + best_pair[1]


    print(
        "merge:",
        best_pair,
        "->",
        new_token
    )


    merges.append(best_pair)


    token_to_id[new_token] = next_token_id

    next_token_id += 1


    corpus = merge_pair(
        corpus,
        best_pair
    )



# ----------------------------
# Save vocabulary
# ----------------------------

with open(
    "vocab.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        token_to_id,
        f,
        ensure_ascii=False,
        indent=2
    )


# ----------------------------
# Save merges
# ----------------------------

with open(
    "merges.txt",
    "w",
    encoding="utf-8"
) as f:

    for a,b in merges:

        f.write(
            f"{a} {b}\n"
        )


print(
    "Training finished"
)

print(
    "Vocabulary size:",
    len(token_to_id)
)