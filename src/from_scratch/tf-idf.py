from collections import Counter

import torch


class TFIDF:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.idf = {}

    def fit(self, documents: list[str]) -> None:
        tokenized_documents = [doc.lower().split() for doc in documents]
        vocab = {word for document in tokenized_documents for word in document}
        self.vocab = {word: i for i, word in enumerate(vocab)}
        self.vocab_size = len(self.vocab)

        df = {}
        for document in tokenized_documents:
            unique_words = set(document)
            for word in unique_words:
                if word not in df:
                    df[word] = 0
                df[word] += 1

        N = len(documents)
        for word in self.vocab:
            self.idf[word] = torch.log(N / df[word])

    def transform(self, documents: list[str]) -> torch.Tensor:
        tokenized_documents = [doc.lower().split() for doc in documents]

        tfidf_matrix = torch.zeros(len(documents), self.vocab_size)
        for i, document in enumerate(tokenized_documents):
            counts = Counter(document)
            for word, count in counts.items():
                if word in self.vocab:
                    tf = count / len(document)
                    tfidf_matrix[i, self.vocab[word]] = tf * self.idf[word]
        return tfidf_matrix

    def fit_transform(self, documents: list[str]) -> torch.Tensor:
        self.fit(documents)
        return self.transform(documents)
