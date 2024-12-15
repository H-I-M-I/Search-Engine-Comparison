import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Load the dataset
file_path = r'G:\Academic\4thYear\4-2_term\Current_term\Data_Mining_Lab\Review_Dataset_Lab3,4,5,' \
            r'6\British_Airway_Review.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Step 1: Extract the "reviews" column and preprocess the text
reviews = data['reviews'].dropna()  # Drop missing values


# Preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data."""
    text = re.sub(r"✅ Trip Verified \|", "", text)  # Remove special markers
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                  "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                  "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                  "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                  "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                  "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
                  "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                  "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
                  "now"}
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Apply preprocessing
cleaned_reviews = reviews.apply(preprocess_text)

# Step 2: Create TF-IDF Matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_reviews)

# Step 3: Create Term-Term Correlation Matrix
term_correlation_matrix = cosine_similarity(tfidf_matrix.T)

# Step 4: Apply Latent Semantic Indexing (LSI)
svd = TruncatedSVD(n_components=2, random_state=42)  # Reduce to 2 components
lsi_matrix = svd.fit_transform(tfidf_matrix)


# Step 5: Define Search Engine Function
def search_engine(qu, matrix, method="TF-IDF"):
    """Search for documents using various methods."""
    qu = preprocess_text(qu)
    query_vector = vectorizer.transform([qu])

    if method == "TF-IDF":
        similarities = cosine_similarity(query_vector, matrix).flatten()
    elif method == "Term-Term":
        term_scores = query_vector.dot(term_correlation_matrix)  # Term scores
        similarities = tfidf_matrix.dot(term_scores.T).flatten()  # Document scores
    elif method == "LSI":
        query_lsi = svd.transform(query_vector)
        similarities = cosine_similarity(query_lsi, lsi_matrix).flatten()

    # Rank documents
    ranked_indices = similarities.argsort()[::-1][:5]  # Top 5 results
    return ranked_indices, similarities[ranked_indices]


# Example Query
query = "great service and experience"
tfidf_results, tfidf_scores = search_engine(query, tfidf_matrix, method="TF-IDF")
term_corr_results, term_corr_scores = search_engine(query, tfidf_matrix, method="Term-Term")
lsi_results, lsi_scores = search_engine(query, tfidf_matrix, method="LSI")


# Step 6: Display Results
def display_results(results, scores, method):
    """Display the search results."""
    print(f"\nTop 5 results for {method}:")
    for i, idx in enumerate(results):
        print(f"{i + 1}. {reviews.iloc[idx]} (Score: {scores[i]:.4f})")


display_results(tfidf_results, tfidf_scores, "TF-IDF")
display_results(term_corr_results, term_corr_scores, "Term-Term")
display_results(lsi_results, lsi_scores, "LSI")
