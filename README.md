# Graph-Based Semantic Clustering using NLP & Shortest Path Optimization
This project explores how semantic relationships between words can be learned directly from raw text by combining **natural language processing, graph theory, and unsupervised learning**.

Instead of relying on pre-trained embeddings, the approach constructs a **custom semantic space** using co-occurrence patterns and refines it through **shortest-path analysis**, enabling deeper insights into how words are contextually connected.

## Project Overview
Understanding relationships between words is a fundamental problem in NLP. Traditional approaches often rely on vector embeddings, but this project takes a more interpretable route by:
* Building semantic similarity from **sentence-level co-occurrence**
* Representing relationships as a **weighted graph**
* Enhancing semantic structure using **Dijkstra's shortest path algorithm**
* Applying clustering to uncover **latent word groupings**

The result is a system that demonstrates how meaning can emerge from both **direct and indirect relationships in text**.

## Methodology
The project follows a structured pipeline, implemented across Jupyter notebooks:
1. **Text Collection & Preprocessing**
A corpus is constructed by combining multiple literary texts. The data is cleaned and prepared through:
* Sentence tokenization
* Text normalization (lowercasing, punctuation handling)
* Removal of stopwords
* Preservation of sentence boundaries for contextual analysis
This ensures that the downstream analysis captures meaningful linguistic structure.

2. **Vocabulary Selection**
A curated set of ~100 meaningful words is selected, focusing on content-bearing terms such as nouns, verbs, and adjectives.

This step reduces noise and ensures that the analysis focuses on **semantically rich tokens**, rather than high-frequency function words.

3. **Semantic Distance Construction**
A custom distance metric is designed to quantify relationships between word pairs based on:
* Co-occurrence within the same sentence
* Relative positional distance in the text

This produces a **pairwise distance matrix**, forming the foundation of the semantic space. Unlike standard similarity measures, this approach is fully interpretable and tailored to the dataset.

4. **Baseline Clustering**
Unsupervised learning techniques are applied to the initial distance matrix to group words into clusters.

These clusters reflect **direct co-occurrence relationships**, providing an initial view of how words are organized semantically. However, due to sparsity in language, some relationships remain weak or fragmented.

5. **Graph Construction**
To address limitations of direct co-occurrence, the problem is reformulated as a graph:
* Words are treated as nodes
* Pairwise distances define weighted edges

This creates a **semantic network**, where relationships between words can be explored beyond immediate proximity.

6. **Shortest Path Computation (Dijkstra's Algorithm)**
Dijkstra’s algorithm is applied to compute the shortest path between all word pairs in the graph.

This step captures **indirect relationships**, allowing words to be connected through intermediate terms. As a result, the semantic distances become more representative of the overall structure of the language.

7. **Refined Clustering**
Clustering is repeated using the updated shortest-path distance matrix.

This leads to:
* More coherent and interpretable clusters
* Stronger grouping of conceptually related words
* Reduced impact of sparse or missing co-occurrences

The refined representation better captures the underlying semantic landscape of the corpus.

### Key Insights
* Direct co-occurrence alone is insufficient for capturing semantic similarity
* Graph-based shortest paths significantly improve relationship modeling
* Representation of data has a greater impact than the choice of clustering algorithm
* Language can be effectively modeled as a **network of interconnected concepts**

### Tech Stack
* Python
* NumPy & Pandas
* NLTK (text processing)
* Scikit-learn (clustering)
* NetworkX (graph modeling & shortest path computation)
* Matplotlib / Seaborn (visualization)

## Conclusion
This project demonstrates how combining **NLP with graph algorithms** can lead to a richer and more nuanced understanding of language. By transforming text into a network and refining relationships through shortest-path analysis, it highlights the importance of **indirect connections in semantic modeling**.

## Author
Nandhitha Sivakumar
