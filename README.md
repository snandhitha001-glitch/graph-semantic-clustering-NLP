# Climate Sentiment Analysis & Semantic Topic Discovery (NLP + AI)
## Problem Overview

Climate-related corporate disclosures are complex, unstructured, and highly contextual. The same language can signal **risk, opportunity, or neutrality depending on phrasing**, making automated analysis non-trivial.

This project investigates how different NLP approaches perform across two key tasks:
* **Supervised Learning**: Classifying climate sentiment (Risk / Opportunity / Neutral)
* **Unsupervised Learning**: Extracting latent themes using topic modeling and graph-based clustering

The focus is on **representation learning, model comparison, and interpretability trade-offs** across classical and modern NLP methods.

### Dataset
* **Source**: ClimateBERT (`climatebert/climate_sentiment`)
* **Type**: Annotated corporate disclosure text
* **Classes**:
  * Risk
  * Opportunity
  * Neutral
* **Split Strategy**:
  * Custom train-validation split using `train_test_split`

## Part 1: Climate Sentiment Classification
### Preprocessing & Representation
* Tokenization using **NLTK** (`word_tokenize`)
* Stopword removal and text normalization
* Feature extraction using:

**Bigram-Based Vectorization**
```
CountVectorizer(ngram_range=(2,2))
```

**Why Bigrams?**
* Capture phrase-level semantics ("climate risk", "carbon emissions")
* Improve signal over unigram-based representations
* Reduce ambiguity in domain-specific text

### Models Implemented
* **Multinomial Naïve Bayes** (baseline + modified)
* **Logistic Regression / Linear classifiers**

#### Naïve Bayes Enhancement
**Modifications**
* Transition from unigram → **bigram features**
* Controlled vocabulary using vectorization
* Improved feature representation for probability estimation

**Impact**
* Better capture of contextual phrases
* Improved class separability in sparse feature space

**Trade-offs**
* High dimensionality from bigrams
* Increased sparsity → potential overfitting
* Conversion to dense matrix (`toarray()`) → memory inefficiency

**Model Behavior & Observations**
* Linear models outperform Naïve Bayes in:
  * Handling correlated features
  * Weighting important terms
* Persistent challenges:
  * **Risk vs Opportunity overlap**
  * Context-dependent meaning of keywords
  * Subtle neutral statements

**Key Limitation**
All models rely on **frequency-based representations (Bag-of-Words / TF-IDF)**
- No true semantic understanding
- Cannot capture contextual similarity between phrases

## Part 2: Topic Modeling (Unsupervised Learning)
### 1. Latent Dirichlet Allocation (LDA)
**Implementation**
* Preprocessing:
  * Lowercasing
  * Stopword removal
  * Token filtering (`isalpha`)
* Built:
  * Gensim dictionary
  * Bag-of-Words corpus
* Model:
```
LdaModel(num_topics=5, passes=10, random_state=42)
```

**Strengths**
* Interpretable topic-word distributions
* Simple probabilistic framework

**Limitations**
* Assumes word independence
* Weak performance on short, context-heavy text
* Topic coherence highly sensitive to preprocessing

### 2. BERTopic (Transformer-Based Topic Modeling)
**Implementation**
```
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(text_data)
```

**Why BERTopic?**
* Uses **transformer embeddings** instead of BoW
* Captures **semantic similarity** across documents
* Automatically clusters embeddings into topics

**Key Advantage**: Introduces context-aware topic modeling, overcoming limitations of LDA

### 3. Clustering-Based Approach
* Applied KMeans clustering on vectorized text
* Evaluated using Silhouette Score

**Insight**
* Performance highly dependent on feature representation
* Works better with structured embedding space

## Part 3: Graph-Based Semantic Clustering (Advanced)
**Objective**: Move beyond feature-based grouping to **relationship-based clustering of documents**

### Methodology
#### Step 1: Vector Representation
* TF-IDF or embedding-based vectors
#### Step 2: Similarity Computation
* Pairwise cosine similarity
#### Step 3: Graph Construction
* Nodes → documents
* Edges → similarity above threshold
#### Step 4: Community Detection / Clustering
* Identify clusters based on graph connectivity

### Why This Matters
Unlike traditional methods:
* Captures **inter-document relationships**
* Models **semantic neighborhoods**
* Introduces **network-based learning into NLP**

### Design Trade-Offs
|**Parameter** |**Effect** |
|--------------|-----------|
|High similarity threshold |Sparse graph, precise clusters |
|Low threshold |Dense graph, noisy clusters |

**Key Insight**
* Graph structure provides an additional layer of information beyond feature space — enabling more flexible and interpretable clustering.

**Limitations**
* Sensitive to similarity threshold selection
* Computationally expensive for large datasets
* Dependent on quality of vector representation

## Key Technical Insights
* **Feature representation is the dominant factor in NLP performance**
* Bigrams improve classification but remain limited
* Transformer embeddings significantly enhance semantic understanding
* Graph-based methods introduce structure-aware learning
* Combining supervised + unsupervised methods provides deeper analytical capability

## Tech Stack
* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* Gensim (LDA)
* BERTopic
* HuggingFace Datasets
* Jupyter Notebook
* Git

## Author
Nandhitha Sivakumar
