"""Term Frequency Ranked retrieval model."""

from math import log10

USE_DOT_PRODUCT = True


def dot_product(query_idx: dict[str, int], document: dict[str, int]) -> float:
    """Return the log frecuency using dot product."""
    sum: float = 0

    for word in query_idx.keys():
        if document[word] > 0:
            qi = log10(1 + query_idx[word])
            di = log10(1 + document[word])
            sum += qi * di
    return sum


def log_frequency(query_idx: dict[str, int], document: dict[str,
                                                            int]) -> float:
    """Return the log frecuency using simple log."""
    sum: float = 0

    for word in query_idx.keys():
        if document[word] > 0:
            sum += 1 + log10(query_idx[word])
    return sum


def term_frequency(query_idx: dict[str, int], document: dict[str,
                                                             int]) -> float:
    """Call the term frequency method."""
    if USE_DOT_PRODUCT:
        return dot_product(query_idx, document)
    else:
        return log_frequency(query_idx, document)
