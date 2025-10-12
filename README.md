# Alfie's Data Science Portfolio

# [Project 1: Fine-grained Sentiment Analysis](https://www.github.com/acaudwell03/Sentiment_Analysis)

## Project Summary
Using Python and R, I performed a sentiment analysis with a Random Forest (RF), Convolution Neural Network (CNN), and RoBERTa model on fine-grained emotions to. After training, optimisation, and evaluation, the key findings were as follows:
* RoBERTa performed the best with an average F1-score of 54%, compared to the CNN and RF models performing at 49% and 47%, respectively
* RF performed better which emotions with a lower sample size
* Performance positively correlated with sample size (0.284)
* RoBERTa is able to use context to identify subtle nuances between emotions

The results demonstrate the importance of model selection for sentiment analysis, reccomending RF models for smaller datasets and RoBERTa for larger datasets. It fills the gap within sentiment analysis research by comparing traditional, deep learning, and transformer models on a fine-grained dataset. This can be applied to businesses, social medias, or advertisement companies wanting to understand their users' feelings towards products and services
  
## Motivation
This is my final project completed during my MSc Data Science course titled 'From Traditional to Transformers - A Comparative Approach to Fine-Grained Sentiment Analysis'. Sentiment analysis, the prediction of emotions or polarity based on data, has become more useful within businesses by applying it to product reviews, customer reccomendations, and chatbots. Understanding how customers or clients feel about a service or product can lead to actionable improvements to boost productivity and satisfaction. This can be done by analysing text, using deep learning models, to identify positive or negative sentiments and picking out key words to help identify a common issue. Many studies have demonstrated models that can accurately predict sentiments from Amazon reviews and online comments.

However, there is little research on the number of emotions these models are able to accurately predict as they predict only positive and negetive seneiments. Understanding fine-grained emotions, such as happiness or frustration, can lead to more targetted interventions and solutions which can be tailored to the customer, potentially improving satisfaction and productivity. This project aims to fill this gap.

# Methodology
* The dataset used is the GoEmotions dataset by Google's Research Team
* Data was imported, cleaned, preprocessed, trained, and evaluated for each model using Python and R
* Preprocessing algorithms include: TF-IDF, Word Embeddings, Tokenisation, Stemming, Stopword Removal, Multi-Hot Encoding
* Trained models include: Random Forest, CNN, RoBERTa
* Models were optimised using Random Search (RF) and Bayesian Optimisation (CNN, RoBERTa)
* F1-scores were compared per emotion
* Post-Hoc analyses included a correlational analysis and SHAP analysis

## Results
### Emotion Distribution
![](images/emotion_frequency.png)

<br>

### Emotion Frequency ![](images/Emotion_Count.pdf)
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

### Correlation Between Sample Size and Avergage Performance
![](images/correlation.png)

<br>

## Discussion
As the RoBERTa model performed the best out of all three algorithms, it is reccomended for this model or other transformer-based models to be used on large datasets for fine-grained sentiment analysis. However, it would be sensible to use RF for smaller sample sizes as this algorithm is more robust with class imbalances and smaller datasets; deep learning models require more data in order to find meaningful relationships and embeddings. Some applications for this task can include:
* Analysing product reviews
* Understanding sentiment during chatbot messaging
* Targetted ads on social medias for users displaying certain emotions

Future work should be done looking into the following:
* Larger, more balanced datasets
* Data using different demographics/platforms
* Text in different languages
* The effects in its practical applications within businesses

<br>

# [Project 2: Music Exploration](https://github.com/acaudwell03/Music_Exploration)

This project demonstrates capabililties to work with SQL/SQLite3 and data exploration. Using a dataset from Kaggle about songs, I performed data exploration and visualisation regarding many features, including functions to interact with the database and an interactive menu which presents different elements of the database chosen by the user.
* Functions were created to interact with an SQLite3 database from python, which includes adding and deleting entries, creating new tables, and functions to perform general SQL queries
* Interactive tasks include retrieving data about different artists, genres, and finding top artists based on a scoring system
* Interactive menu using widgets to perform these tasks from a simple user-friendly interface

## Example of Visualisation from a Selected Year
![](images/music_figure.png)

