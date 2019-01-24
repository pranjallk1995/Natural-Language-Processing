#Natural Language Processing

"""Restaurant review analyzer"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

#importing data
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)      #csv file is not used in NLP since text can contain commas

#cleaning the texts
#nltk.download('stopwords')         used if stepwords needs to be updated.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

N = len(dataset)
corpus = []
for i in range(0, N):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    stemmed_review = []
    for word in review:
        if word not in set(stopwords.words('english')):         #set is required since it is faster to search.
            stemmed_review.append(ps.stem(word))
            review = ' '.join(stemmed_review)
    corpus.append(review)

#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)                      #can also be used to perform advanced cleaning like removing br from text from html codes (<,/ and > was already removed above etc.
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1]

#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.2)

#fitting Naive Bayesian classifier                  
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, Y_train.values.ravel())

#predicting values
Y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
