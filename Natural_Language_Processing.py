#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:29:27 2020

@author: kiran
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(0, 1000):
    
    review = re.sub( '[^A-Za-z]' , ' ', dataset['Review'][i])
    review = review.lower() 
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review  =' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words={'english'}, max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split

X_train, X_test , y_train , y_test = train_test_split(X, y , test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train , y_train)


y_pred = nb.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)