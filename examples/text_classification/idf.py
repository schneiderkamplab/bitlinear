
import torch
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

from tqdm import tqdm

def inverse_document_frequency(encoded_docs, vocab_size):
    """Returns IDF scores in shape [vocab_size]"""
    num_docs = len(encoded_docs)
    counts = sp.dok_matrix((num_docs, vocab_size))
    for i, doc in tqdm(enumerate(encoded_docs), desc="Computing IDF"):
        for j in doc:
            counts[i, j] += 1

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)

    tfidf.fit(counts)

    return torch.FloatTensor(tfidf.idf_)
