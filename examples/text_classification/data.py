from joblib import Memory
from tqdm import tqdm

import torch

import os.path as osp
import random
CACHE_DIR = "tmp/cache"
MEMORY = Memory(CACHE_DIR, verbose=2)
VALID_DATASETS = ["20ng", "r8", "r52", "ohsumed", "mr"]

@MEMORY.cache(ignore=["n_jobs"])
def load_data(
    key,
    tokenizer,
    max_length=None,
    construct_textgraph=False,
    n_jobs=1,
    force_lowercase=False,
    raw=False,
):
    assert key in VALID_DATASETS, f"{key} not in {VALID_DATASETS}"
    print("Loading raw documents")
    with open(osp.join("data", "corpus", key + ".txt"), "rb") as f:
        raw_documents = [line.strip().decode("latin1") for line in tqdm(f)]

    N = len(raw_documents)

    # print("First few raw_documents", *raw_documents[:5], sep='\n')

    labels = []
    train_mask, test_mask = torch.zeros(N, dtype=torch.bool), torch.zeros(
        N, dtype=torch.bool
    )
    print("Loading document metadata...")
    doc_meta_path = osp.join("data", key + ".txt")
    with open(doc_meta_path, "r") as f:
        for idx, line in tqdm(enumerate(f)):
            __name, train_or_test, label = line.strip().split("\t")
            if "test" in train_or_test:
                test_mask[idx] = True
            elif "train" in train_or_test:
                train_mask[idx] = True
            else:
                raise ValueError(
                    "Doc is neither train nor test:"
                    + doc_meta_path
                    + " in line: "
                    + str(idx + 1)
                )
            labels.append(label)

    assert len(labels) == N
    # raw_documents, labels, train_mask, test_mask defined

    if raw:
        return raw_documents, labels, train_mask, test_mask

    if max_length:
        print(f"Encoding documents with max_length={max_length}...")
        # docs = [tokenizer.encode(raw_doc, max_length=max_length) for raw_doc in raw_documents]
        # docs = tokenizer(raw_documents, truncation=True, max_length=max_length)

        # Now use truncation=True (continued experiments with seq2mat)
        docs = [
            tokenizer.encode(raw_doc, truncation=True, max_length=max_length)
            for raw_doc in raw_documents
        ]
    else:
        print(f"Encoding documents without max_length")
        docs = [tokenizer.encode(raw_doc) for raw_doc in raw_documents]

    print("Encoding labels...")
    label2index = {label: idx for idx, label in enumerate(set(labels))}
    label_ids = [label2index[label] for label in tqdm(labels)]

    return docs, label_ids, train_mask, test_mask, label2index

