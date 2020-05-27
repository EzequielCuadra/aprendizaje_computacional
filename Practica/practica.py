# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:37:15 2020

@author: Ezequiel Cuadra
"""

import nltk  # for text manipulation 
import pandas as pd # for data manipulation 

# =============================================================================
# ~ Prepare the dataset
# =============================================================================

# Load the dataset
data = pd.read_csv("tweet_dataset.csv", encoding = "latin-1")

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
# Rename the columns
data.columns = DATASET_COLUMNS
# Drop the columns without importance for us
data.drop(['ids','date','flag','user'],axis = 1,inplace = True)

# Lets work with a subset from the dataset ## Updated to entire dataset 
positif_data = data[data.target==4]
negative_data = data[data.target==0]

# Now, we have a dataset with 200000 rows
data = pd.concat([positif_data,negative_data],axis = 0)
print(data.shape)

# =============================================================================
# ~ Clean the text column, adding a new column with the clean text
# =============================================================================

# Remove the @ from the @users
data['Clean_TweetText'] = data['text'].str.replace("@", "") 
# Remove the URL's
data['Clean_TweetText'] = data['Clean_TweetText'].str.replace(r"http\S+", "") 
# Remove special characters
data['Clean_TweetText'] = data['Clean_TweetText'].str.replace("[^a-zA-Z]", " ") 
# Remove words with len under 3
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# Load stopwords from nltk
stopwords=nltk.corpus.stopwords.words('english')

# Function to remove stopwords from each tweet
def remove_stopwords(text):
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text

# Remove stop words and lowercasing
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda text : remove_stopwords(text.lower()))

# Tokenize the words
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: x.split())

# Stemming the words
from nltk.stem.porter import PorterStemmer 
stemmer = PorterStemmer() 
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: [stemmer.stem(i) for i in x])

# Join the stemmed words and tokens
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x]))
print(data.head())

# =============================================================================
# ~ Data vectorization
# =============================================================================

# Tf-idf vectorization 
from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english")
tfv = tfv.fit_transform(data['Clean_TweetText'])
'''
# Bag of Words vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
cv = cv.fit_transform(data['Clean_TweetText'])
'''
# =============================================================================
# ~ Building the model
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Split the data into train and test set 
X_train,X_test,y_train,y_test = train_test_split(tfv,data['target'] , test_size=.2,stratify=data['target'], random_state=42)

# =============================================================================
# ~ LogisticRegression model
# =============================================================================

from sklearn.linear_model import LogisticRegression 

lr = LogisticRegression(max_iter=1000,solver='lbfgs')
print("...fitting data...")
lr.fit(X_train,y_train)
print("===== LOGISTIC REGRESSION: ======")
prediction_lr = lr.predict(X_test)
print("-----ACCURACY: -----")
print(accuracy_score(y_test,prediction_lr)*100, '%')
print("-----CONFUSION MATRIX: -----")
print(confusion_matrix(y_test,prediction_lr))

# =============================================================================
# ~ Naive Bayes model
# =============================================================================

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB (alpha = 10)
print("...fitting data...")
nb.fit(X_train, y_train)
print("===== NAIVE BAYES : ======")
prediction_nb = nb.predict(X_test)
print("-----ACCURACY: -----")
print(accuracy_score(y_test,prediction_nb)*100, '%')
print("-----CONFUSION MATRIX: -----")
print(confusion_matrix(y_test,prediction_nb))


# =============================================================================
# ~ LinearSVC model
# =============================================================================

from sklearn.svm import LinearSVC

lsvc = LinearSVC(max_iter=20000)
print("...fitting data...")
lsvc.fit(X_train, y_train)
print("===== LINEAR SVC : ======")
prediction_lsvc = lsvc.predict(X_test)
print("-----ACCURACY: -----")
print(accuracy_score(y_test,prediction_lsvc)*100, '%')
print("-----CONFUSION MATRIX: -----")
print(confusion_matrix(y_test,prediction_lsvc))






