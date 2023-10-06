"""Module for the Jaccard coefficient ranked retrieval method."""

from math import sqrt


def jaccard(query: set[str], document: set[str]) -> float:
    """Return the Jaccard coefficient of two sets."""
    return len(query & document) / sqrt(len(query | document))
