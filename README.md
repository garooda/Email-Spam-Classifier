# Gankara

### Email Spam Classifier Using Naive Bayes Algorithm
Gankara is an email security system that automatically checks your incoming emails and classifies them into spam, phishing, frauds and safe mails. Gankara lets you focus on what really matters, the good stuff - family, friends, and that promotion you've been waiting for. Gankara is an AI powered email spam classification system that works behind the scenes to do what you already do; read your emails and mark safe or unsafe. Except we do it better! We protect you from online fraud, scams, spam and all those unwanted emails.

<p align="center">
  <img src="https://cdn.pixabay.com/photo/2014/09/28/10/38/road-sign-464657_960_720.png" width=300 height=300>
</p>
Photo By - [geralt-9301](https://pixabay.com/users/geralt-9301/)

## Table of Content
- [Introduction to Naive Bayes](#introduction-to-naive-bayes)
- [Conditional Probability](#condtional-probability)
- [Code](#code)

#### Introduction to Naive Bayes
Classification is one of the core task of machine learning that is widely used in various applications like spam filtering, medical diagnosis, fraud detection, website navigation path prediction etc. Naive Bayes algorithm belongs to the family of probabilistic classifiers and it assumes that each feature/ attribute (vector) is conditionally independent given the class value, and that the conditional distributions are Gaussian.

#### Conditional Probability
Conditional probability is defined as the likelihood of an event or outcome occurring, based on the occurrence of a previous event or outcome. Conditional probability is calculated by multiplying the probability of the preceding event by the updated probability of the succeeding, or conditional, event. 

For example:
* Event A is that an individual applying for college will be accepted. There is an 80% chance that this individual will be accepted to college.
* Event B is that this individual will be given dormitory housing. Dormitory housing will only be provided for 60% of all of the accepted students.
* P (Accepted and dormitory housing) = P (Dormitory Housing | Accepted) P (Accepted) = (0.60)*(0.80) = 0.48.

#### Code
Jupyter python notebook is available at [nbviewer](https://nbviewer.jupyter.org/github/garooda/Email-Spam-Classifier/blob/main/email_classifier.ipynb)

Download the dataset from [here](https://github.com/garooda/Email-Spam-Classifier/blob/main/emails.csv)

##### loading the important libraries

```python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
```

##### loading the dataset

```python3
df = pd.read_csv("emails.csv")
```

##### data preprocessing and gathering info about the dataset
```python3
df.shape      #checking the number of the data points

df[df.label == 0].count() #checking the number of data points belonging to class 0 i.e. ham

df[df.label == 1].count() #checking the number of data points belonging to class 1 i.e. spam
```
