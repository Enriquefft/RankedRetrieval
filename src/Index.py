"""File that implements an inverse index class."""

from .tokenizer import tokenize, normalize, is_relevant
from pathlib import Path
from collections import defaultdict

from typing import Optional

import logging

from Config import DOCUMENT_FOLDER, STOP_WORDS_FOLDER


def key_format(key: str) -> str:
    """Format a keyword to be used in a search."""
    if not is_relevant(key):
        raise ValueError(f'"{key}" is not a valid search term.')
    return normalize(key)


class termInfo:
    """Stores the info stored for each term found in the documents."""

    def __init__(self, docsId: set[Path], tf: list[int]):
        """Create a new termInfo object."""
        self.docsId = docsId
        self.tf = tf


class InverseIndex:
    """A dictionary that maps words to a list of documents."""

    def __init__(self, documents: str, stop_words: Optional[str] = None):
        """Create an inverse index from a list of documents and stop words."""
        # https://docs.python.org/3/library/collections.html#collections.defaultdict

        self.documents: list[Path] = []

        documents_path = Path(documents)
        for idx, file in enumerate(documents_path.iterdir()):
            if file.is_file():
                self.documents.append(file)
                # with open(file, "r") as r_file:
                #     doc = r_file.read()
                #     for word in tokenize(doc):
                #         self.terms[word].tf[idx] += 1
                #         self.terms[word].docsId.add(file)


idx = InverseIndex(documents=DOCUMENT_FOLDER, stop_words=STOP_WORDS_FOLDER)
