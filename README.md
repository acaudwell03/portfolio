# Alfie's Data Science Portfolio

# [Project 1: Fine-grained Sentiment Analysis](https://www.github.com/acaudwell03/Sentiment_Analysis)

This is my final project completed during my MSc Data Science course titled 
'From Traditional to Transformers - A Comparative Approach to Fine-Grained Sentiment Analysis'.

* The dataset used is the GoEmotions dataset by Google's Research Team
* Data was imported, cleaned, preprocessed, trained, and evaluated for each model
* Preprocessing algorithms include: TF-IDF, Word Embeddings, Tokenisation, Stemming
* Trained models include: Random Forest, CNN, RoBERTa
* Models were optimised using Random Search and Bayesian Optimisation
* F1-scores were compared using ANOVA per emotion and overall
* Correlation analysis found a weak positive relationship between sample size for each emotion and average perforamance
* Demonstrations of each model involve a custom sentence input which outputs highest predicted emotions
* SHAP analysis was done on the RoBERTa model

## Emotion Distribution
![](images/emotion_frequency.png)

## Emotion Frequency ![](images/Emotion_Count.pdf)
  
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

## Correlation Between Sample Size and Avergage Performance
![](images/correlation.png)

