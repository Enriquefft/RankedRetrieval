from sklearn.feature_extraction.text import TfidfVectorizer

# Define the document collection
docs = [
    "The sky is blue.", "The sun is bright today.",
    "The sun in the sky is bright.",
    "We can see the shining sun, the bright sun."
]

# Create the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Compute the TF-IDF vectors for the document collection
tfidf_matrix = vectorizer.fit_transform(docs)
print(tfidf_matrix)

# Convert the TF-IDF matrix to an array
tfidf_array = tfidf_matrix.toarray()
print(tfidf_array)
