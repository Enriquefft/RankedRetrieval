"""Cosine method for the Ranked retrieval model."""

from math import log10
from ..Index import InverseIndex
from ..tokenizer import tokenize, normalize, is_relevant
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from collections import defaultdict


class CosineMethod(InverseIndex):
    """Cosine Index Ranked Retrieval class."""

    def __init__(self, documents: str, stop_words: Optional[str] = None):
        """Initialize the cosine index."""
        super().__init__(documents, stop_words)

        # Matrix with len(self.documents) rows and empty vectors
        self.tf_idf_matrix: list[NDArray[np.float]] = [
            np.empty(0) for _ in range(len(self.documents))
        ]
        print(self.tf_idf_matrix)

        self.term_idx: dict[str, int] = {}

        self.term_documents: dict[str, set[Path]] = defaultdict(set)

        tf: dict[tuple[str, Path], int] = defaultdict(int)
        word_count: int = 0
        for doc_idx, doc in enumerate(self.documents):

            with open(doc, 'r') as file:
                document: str = file.read()
                for word in tokenize(document):
                    normalized_word: str = normalize(word)
                    if not is_relevant(normalized_word):
                        continue

                    # Assign word_count to term_idx[word] if word is not in term_idx
                    if word not in self.term_idx:
                        # Word hasnt been seen before
                        self.term_idx[word] = word_count
                        word_count += 1

                    tf[word, doc] += 1
                    self.term_documents[word].add(doc)

                # for word, freq in tf.items():
                #     self.tf_idf_matrix[doc_idx][term_idx[word]] = freq * log10(
                #         len(self.documents) / len(self.term_documents[word]))

        # Get the tf_idf_matrix
        for doc_idx, doc in enumerate(self.documents):
            self.tf_idf_matrix[doc_idx] = np.zeros((word_count, ))
            for word, idx in self.term_idx.items():
                self.tf_idf_matrix[doc_idx][idx] = tf[word, doc] * log10(
                    len(self.documents) / len(self.term_documents[word]))

    def compare(self, query: str, doc: Path) -> float:
        """Compare the query and the document."""
        query_vector: NDArray[np.float] = np.zeros(
            (len(self.tf_idf_matrix[0]), ))

        tf: dict[str, int] = defaultdict(int)
        for word in tokenize(query):
            normalized_word: str = normalize(word)
            if not is_relevant(normalized_word):
                continue
            tf[word] += 1

        for word, freq in tf.items():
            query_vector[self.term_idx[word]] = freq * log10(
                len(self.documents) / len(self.term_documents[word]))

        doc_vector: NDArray[np.float] = self.tf_idf_matrix[
            self.documents.index(doc)]
        return np.dot(query_vector, doc_vector)
