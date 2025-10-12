# Alfie's Data Science Portfolio

# [Project 1: Fine-grained Sentiment Analysis](https://www.github.com/acaudwell03/Sentiment_Analysis)

## Project Overview
This project compares **traditional machine learning, deep learning, and transformer-based models** for **fine-grained sentiment analysis** across 27 distinct emotions using the GoEmotions dataset by Google Research.

The models evaluated were:
* **Random Forest (RF)** – baseline traditional approach
* **Convolutional Neural Network (CNN)** – custom deep learning model
* **RoBERTa** – pre-trained transformer fine-tuned on emotion labels
The goal was to understand how each type of model handles nuanced emotional expressions and dataset imbalances, providing actionable insights for business and research applications in NLP.

## Key Results
| Model             | Avg. F1-Score | Notes                                                |
| :---------------- | :------------ | :--------------------------------------------------- |
| **Random Forest** | 0.47          | **Robust** with small or imbalanced datasets             |
| **CNN**           | 0.49          | Learns **basic semantic structures**, limited context    |
| **RoBERTa**       | **0.54**      | Best overall performance; captures **contextual nuance** |

<br>

* Performance **positively correlated** with sample size (r = 0.284)
* RoBERTa effectively identified subtle distinctions between similar emotions (e.g., love vs joy)

## Visual Results

### Model Performance by Emotion
| Emotion         | RF   | CNN  | RoBERTa | Average |
|-----------------|------|------|----------|----------|
| Admiration      | 0.68 | 0.69 | **0.74** | 0.70 |
| Amusement       | 0.82 | 0.82 | **0.87** | 0.84 |
| Anger           | 0.49 | 0.51 | **0.57** | 0.52 |
| Annoyance       | 0.40 | 0.40 | 0.40 | 0.40 |
| Approval        | 0.38 | 0.37 | **0.49** | 0.41 |
| Caring          | 0.39 | 0.43 | **0.44** | 0.42 |
| Confusion       | 0.34 | 0.36 | **0.52** | 0.41 |
| Curiosity       | 0.29 | 0.32 | **0.68** | 0.43 |
| Desire          | 0.54 | **0.57** | 0.55 | 0.55 |
| Disappointment  | 0.26 | 0.31 | **0.37** | 0.31 |
| Disapproval     | 0.28 | 0.35 | **0.53** | 0.39 |
| Disgust         | 0.45 | 0.50 | **0.51** | 0.49 |
| Embarrassment   | 0.58 | **0.67** | 0.51 | 0.59 |
| Excitement      | 0.30 | 0.35 | **0.47** | 0.37 |
| Fear            | 0.63 | 0.69 | **0.70** | 0.67 |
| Gratitude       | 0.90 | **0.93** | **0.93** | 0.92 |
| Grief           | **0.31** | 0.12 | 0.00 | 0.14 |
| Joy             | 0.54 | 0.59 | **0.61** | 0.58 |
| Love            | 0.78 | 0.82 | **0.83** | 0.81 |
| Nervousness     | 0.29 | 0.37 | **0.42** | 0.36 |
| Optimism        | 0.58 | **0.63** | 0.59 | 0.60 |
| Pride           | 0.64 | **0.67** | 0.58 | 0.63 |
| Realization     | **0.32** | 0.34 | 0.31 | 0.32 |
| Relief          | 0.22 | 0.19 | **0.27** | 0.23 |
| Remorse         | 0.74 | **0.78** | 0.66 | 0.73 |
| Sadness         | 0.50 | 0.50 | **0.60** | 0.53 |
| Surprise        | 0.54 | 0.57 | **0.64** | 0.57 |

> **Note:** Bold values indicate the best result for each emotion.

<br>

* **Gratitude (0.92)** was the **highest-scoring** emotion; **Grief (0.14)** the **lowest**.
* Performance was tied closely to emotion frequency within the dataset.

<br>

<p align="center"> <img src="images/emotion_frequency.png" alt="Emotion Frequency Distribution" width="500"> </p>

## Methodology
### Data
* **Dataset**: GoEmotions (27 emotions)
* **Preprocessing**: Tokenisation, TF-IDF, Word Embeddings, Stopword Removal, Stemming, Multi-Hot Encoding
* **Split**: Train (70%), Validation (15%), Test (15%)

### Models
* **Random Forest** (scikit-learn) → Optimised with Random Search
* **CNN** (PyTorch) → 1D convolutional layers, dropout, embedding layer → Bayesian Optimization for tuning
* **RoBERTa** (Hugging Face Transformers) → Pre-trained on the dataset, Adam, and learning rate scheduling

### Evaluation
* **F1-score** (macro)
* **Correlation** between sample size & model performance
* **ANOVA** with Greenhouse–Geisser correction + Holm pairwise comparisons
* **SHAP** for interpretability

## Statistcial Findings
* **ANOVA**: F(1.26, 32.76) = 5.04, **p = 0.024** → **significant model effect**
* **Pairwise comparisons**:
  * RF vs DL: marginally non-significant (p ≈ 0.054)
  * CNN vs RoBERTa: not significant (p = 0.096)
* **Spearman's Rho**: **r = 0.284** → **weak positive** correlation between sample size & performance
### Correlation Between Sample Size and Avergage Performance
<p align="center"> <img src="images/correlation.png" alt="Emotion Frequency Distribution" width="500"> </p>

* **Interpretation**: **RoBERTa performs best**, though differences are moderate due to sample imbalance.

## Insights & Applications
* **Business impact**: Detect nuanced emotions in **reviews, social media posts, or chatbots**.
* **Model choice**:
  * Use **RF for small** or imbalanced datasets
  * Use **Transformers (RoBERTa) for large-scale**, context-rich data
* **Future work**:
  * Evaluate **multilingual** or cross-platform data
  * Improve class balance via **data augmentation**
  * Deploy as an **interactive app** for real-time emotion detection

## Tech Kit
**Languages**: Python, R
<br>
**Libraries**: scikit-learn, PyTorch, Transformers, Optuna, SHAP, Pandas, Matplotlib, Seaborn
<br>
**Tools**: Jupyter Notebook, RStudio, VS Code

<br>

# [Project 2: Music Exploration](https://github.com/acaudwell03/Music_Exploration)

This project demonstrates capabililties to work with SQL/SQLite3 and data exploration. Using a dataset from Kaggle about songs, I performed data exploration and visualisation regarding many features, including functions to interact with the database and an interactive menu which presents different elements of the database chosen by the user.
* Functions were created to interact with an SQLite3 database from python, which includes adding and deleting entries, creating new tables, and functions to perform general SQL queries
* Interactive tasks include retrieving data about different artists, genres, and finding top artists based on a scoring system
* Interactive menu using widgets to perform these tasks from a simple user-friendly interface

## Example of Visualisation from a Selected Year
![](images/music_figure.png)

