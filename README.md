# Search Engine Comparison with TF-IDF, Term Correlation, and LSI

This program compares the performance of three search engine methods: **TF-IDF**, **Term-Term Correlation**, and **Latent Semantic Indexing (LSI)**. It processes a dataset of reviews to identify the top relevant documents for a user-provided query. The program evaluates results across these methods and highlights commonalities and differences.

## Features

- **TF-IDF Search Engine**: Matches user queries with document terms using term frequency-inverse document frequency.
- **Term-Term Correlation Search**: Scores documents by aggregating the contributions of query terms present in the documents.
- **Latent Semantic Indexing (LSI)**: Maps documents and queries into a lower-dimensional semantic space for similarity comparison.
- **Rank Comparison**: Analyzes top results from each method to find common documents, unique documents, and rank displacements.
- **Rank Displacement Details**: Calculates displacement metrics for result ranking differences across methods.

## Dataset

Place the dataset file (`British_Airway_Review.csv`) in the root directory. The file should contain a `reviews` column with textual reviews for analysis.

## Example Output

**Query**: `Enter your search query: "business class experience"`

**Results**:
- Top-ranked reviews for TF-IDF, Term Correlation, and LSI.
- A detailed table comparing document rankings across methods.
- Metrics for rank displacement and commonality.

## Dependencies

- Python 3.7+
- `pandas`
- `scikit-learn`
- `numpy`
- `nltk`

## Code Highlights

- **TF-IDF Vectorization**: Converts text data into a weighted term-document matrix.
- **LSI Modeling**: Uses `TruncatedSVD` to reduce dimensionality for semantic analysis.
- **Cosine Similarity**: Measures query-document similarity in the LSI space.
- **Rank Comparison**: Evaluates the overlap and differences between the search results.

## Improvements

- Add support for stemming/lemmatization for better query processing.
- Enable visualization of rank comparisons using plots.
- Extend functionality to process larger datasets efficiently.

