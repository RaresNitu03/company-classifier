
# Company Classifier – Insurance Taxonomy Project

This project uses semantic similarity and machine learning to classify companies into a predefined insurance taxonomy. It leverages company descriptions, business tags, and sector metadata to assign relevant insurance-related labels.

---

## Recommended: Run the Project Online (Colab)

To explore the project interactively **without setting up anything locally**, use the link below:

▶️ **[Open in Google Colab](https://colab.research.google.com/gist/RaresNitu03/ba62580f76321d4985e4dca78c302ee1/insurance_taxonomy_classification.ipynb)**  

This will open the notebook in a cloud-based environment, allowing you to:

- Run each cell step by step
- See intermediate outputs and visualizations
- Test the full classification pipeline in real time

 **Note**: The Colab version includes annotations, cell outputs, and is ready to run without local setup.


## Files Included

- `Company_Classifier.ipynb`: Main notebook with the full classification pipeline.
- `Last_100.csv`: Subset of companies used for manual validation.
- `insurance_taxonomy.csv`: Static taxonomy used as label reference.
- `ml_insurance_challenge.csv`: Raw dataset with company details.
- `classified_companies_with_labels.csv`: Final output dataset with predicted insurance labels.
- `requirements.txt`: Python package dependencies required to run the notebook.

---

## Project Overview

The notebook follows a 10-step pipeline:

0. **Introduction** - Project goal and taxonomy context
1. **Setup & Imports** – Load libraries and install dependencies
2. **Data Loading** – Import raw data and taxonomy files
3. **Data Preparation & Text Processing** – Build full text from multiple fields
4. **Semantic Similarity & Pseudo-Label Generation** – Generate labels via similarity scoring
5. **Training Classifier with Pseudo-Labels** – Train classifier using pseudo-labels
6. **Prediction Function & Re-labeling** – Predict labels with trained model
7. **Model Validation & Performance Analysis** – Compare model vs manual labels
8. **Visual Check & Sample Evaluation** – Manually inspect random samples
9. **Model Saving & Export** – Save results and trained model
10. **Analysis and Conclusions** – Summarize results and insights

---

## Technical Overview

The project pipeline combines classic machine learning techniques with modern semantic representation to enable multi-label classification.

### 🟣 Data Preparation
- Each company record includes description, tags, and metadata (sector, category, niche).
- These fields are concatenated into a single `full_text` string per company.

### 🟣 Embedding & Similarity
- The `full_text` is encoded using a transformer model from `sentence-transformers` to generate a semantic embedding (dense vector).
- Each insurance label from the taxonomy is also embedded in the same vector space.
- Cosine similarity is computed between each company vector and all label vectors to determine label relevance.

### 🟣 Pseudo-Labeling Strategy
- Top `k` labels (default: 5) are assigned as pseudo-labels if their similarity exceeds a threshold (default: 0.3).
- If no label passes the threshold, the fallback mechanism selects the top 3 most similar labels.

### 🟣 Model Training
- A TF-IDF vectorizer transforms the `full_text` into sparse feature vectors.
- A `LogisticRegression` classifier is trained in a `OneVsRestClassifier` wrapper, enabling multi-label learning.
- The model is trained only on companies that received pseudo-labels.

### 🟣 Prediction Function
- The trained model predicts label probabilities for each company.
- If no label passes the probability threshold, the fallback logic selects top 3 labels with highest probabilities.

### 🟣 Evaluation
- Final predictions are compared to manually labeled data.
- Accuracy is computed as the percentage of companies for which at least one predicted label overlaps with a true label.

## Results

The model was evaluated on a manually labeled subset (100 companies) with the goal of achieving high relevance, even in the absence of a complete ground truth.  
A prediction is considered correct if at least one label overlaps with human annotations.

## Local Setup Instructions

To run the project locally, follow these steps:

```bash
git clone https://github.com/username/company-classifier.git
cd company-classifier
pip install -r requirements.txt
jupyter notebook
```

Make sure the `.csv` files are in the same directory as the notebook when running locally.

---

## Requirements

Required Python libraries:

```
sentence-transformers
joblib
scikit-learn
pandas
numpy
```

Install via pip:

```bash
pip install sentence-transformers joblib scikit-learn pandas numpy
```

Or use the provided `requirements.txt`.

---

## Author

Built by **Rareș Nițu**  
This project was developed as part of an AI challenge for insurance sector classification.


---

## References & Useful Links

Here are some useful resources and tools referenced or related to this project:

- [Sentence Transformers Documentation](https://www.sbert.net/) – Used for semantic embeddings
- [scikit-learn MultiLabel Classification](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-classification) – Multi-label classification strategies
- [Google Colab](https://colab.research.google.com/) – Run Python notebooks in the cloud
- [Understanding TF-IDF](https://towardsdatascience.com/a-practical-guide-to-tf-idf-83df764f4f6a) – How TF-IDF works in text processing
- [cosine_similarity in NLP](https://en.wikipedia.org/wiki/Cosine_similarity) – Metric used for comparing embeddings
- **ChatGPT (OpenAI)** – Used as a support tool for manually labeling the 100 validation examples in the dataset.

These helped shape the approach and methods used in building the classifier.

