# Natural Language Processing: Beer review text classification

[Code](https://github.com/SoumyaO/beer-review-text-classification/tree/main/code) | [Report](https://github.com/SoumyaO/beer-review-text-classification/blob/main/Report.pdf) | [Data](https://github.com/SoumyaO/beer-review-text-classification/tree/main/data)

### Contributors
[Soumya Ogoti](https://github.com/SoumyaO)  
[Wenxu Tian](https://github.com/Wayne599)  
[Kartikey Bhalla](https://github.com/KartikeyBhalla)  
[Preeti Pothireddy](https://github.com/preethi799)  
[George Krivosheev](https://github.com/GeorgeKrivosheev)  


## Description
This project consists of a text classification task for user reviews from the BeerAdvocate community/website. The goal is to classify the reviews as *okay*, *good* and *excellent*. The training dataset contains 21,057 labelled reviews and the test data contains 8,943 reviews.

The data was preprocessed by *lemmatizing* the text (root words), removing stop words, punctuations, special characters, numbers (dates), emails and urls and converting all the text to lowercase. Using the **Gensim** library bigrams and trigrams were accounted for.

Both Machine learning and Deep learning models were used in this analysis.

### Machine Learning Models
Several hand-engineered features were used for training the machine learning models. These include TFIDF (Term Frequency and Inverse Document Frequency) with *n_grams* and dimensionality reduction, Topic Modeling with Latent Dirichlet allocation, and Doc2Vec. The classifiers used were Logistic Regression, Multinomial Naive Bayes classifier, Random Forest, OneVsOne classifier and OneVsRest classifier, Support Vector Machine and Voting Classifier. Grid search was used to arrive at the optimal set of hyperparameters. A combination of the above mentioned features and classifiers were used and the macro-average F1 scores were used for evaluating the models. Of the models evaluated, the model with the OneVsOne classifier with a combination of TFIDF, Doc2Vec and Topic modeling gave the best F1 score of 60.6%.

### Deep Learning Models
Bidirectional LSTM, FastText as a classifier, FastText word representations with BiLSTM classifier and BERT were used for the task of text classification. For the Bidirectional LSTM, the input text was tokenized into individual words, cleaned and lemmatised. The Keras tokeniser tokenized the cleaned text into sequences of tokens which were input to the LSTM. In order to capture subword information, extract morphological features and handling out-of-vocabulary words the FastText model was used. In order to combine the benefits of the above two models, FastText embeddings with a BiLSTM model were also explored. Additionally, a BERT (Bidirectional ENcoder Representations from Transformers) model was also investigated. A smart padding approach was used to speed up the training times, where the padding is dynamically changed based on the length of the sequences. Of these models, the BERT model with the smart padded tokens resulted in the best F1 value of 61.3%.

## Conclusion
BERT model with smart padding was chosen as the best model. As BERT is trained on a large and diverse corpus, it helps generalise well to various domains such as beer reviews. BERTs pre-training followed by fine-tuning on the training data allowed it to adapt to the task at hand and aling its representations with specific characteristics of beer reviews.
