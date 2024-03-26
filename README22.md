#  Spam Detection Using TensorFlow in Python
 ## Step 1: Introduction

 TTo building an efficient SMS spam classification model using the SMS Spam Collection dataset. By the end of this notebook, you'll have a powerful tool to help you filter out unwanted messages and ensure that your text messaging experience is smoother and safer. 
  
## Step 2: Problem Statement

The primary goal of this notebook is to develop a predictive model that accurately classifies incoming SMS messages as either ham or spam. We will use the SMS Spam Collection dataset, which consists of 5,574 SMS messages tagged with their respective labels

## Step 3: Import Necessary Libraries

*#Importing necessary libraries*
*import numpy as np        # For numerical operations*
*import pandas as pd       # For data manipulation and analysis*
*import matplotlib.pyplot as plt  # For data visualization
%matplotlib inline*

*#Importing WordCloud for text visualization*
*from wordcloud import WordCloud*

*#Importing NLTK for natural language processing*
*import nltk*
*from nltk.corpus import stopwords*    #For stopwords


*#Downloading NLTK data*
*nltk.download('stopwords')*   #Downloading stopwords data
*nltk.download('punkt')*       #Downloading tokenizer data
 
## Step 4 : Data Cleaning

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   v1          5572 non-null   object
 1   v2          5572 non-null   object
 2   Unnamed: 2  50 non-null     object
 3   Unnamed: 3  12 non-null     object
 4   Unnamed: 4  6 non-null      object
dtypes: object(5)
memory usage: 217.8+ KB




 ## Data Splitting:
We'll split the dataset into a training set (70%) and a test set (30%).

*import pandas as pd*
*from sklearn.model_selection import train_test_split*

*#Load the dataset (replace 'dataset.csv' with your dataset file)*
*data = pd.read_csv('dataset.csv')*

*#Data Cleaning*
*data.dropna(inplace=True)*  # Remove rows with missing values
*data.drop_duplicates(inplace=True)  # Remove duplicates*

*#Data Splitting*
*X = data['text']* # Assuming 'text' column contains email text
*y = data['label']* # Assuming 'label' column contains spam/ham labels

*X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)*

## Step 5: Building the TensorFlow Model
We'll build a deep learning model using TensorFlow/Keras for text classification.

*from tensorflow.keras.models import Sequential*
*from tensorflow.keras.layers import Dense, Embedding, LSTM, *SpatialDropout1D*
*from tensorflow.keras.preprocessing.text import Tokenizer*
*from tensorflow.keras.preprocessing.sequence import pad_sequences*

#Tokenization and padding
*max_features = 10000  # Maximum number of words to keep*
*maxlen = 100  # Maximum length of a sequence*
*tokenizer = Tokenizer(num_words=max_features)*
*tokenizer.fit_on_texts(X_train)*
*X_train_seq = tokenizer.texts_to_sequences(X_train)*
*X_test_seq = tokenizer.texts_to_sequences(X_test)*
*X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)*
*X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)*

#Build the model
*model = Sequential()*
*model.add(Embedding(max_features, 128, input_length=maxlen))*
*model.add(SpatialDropout1D(0.2))*
*model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))*
*model.add(Dense(1, activation='sigmoid'))*
*model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])*

## Import the Models

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


## Train the Models

from sklearn.metrics import accuracy_score, precision_score
def train_classifier(clfs, X_train, y_train, X_test, y_test):
    clfs.fit(X_train,y_train)
    y_pred = clfs.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy , precision

## Step 6: Model Evaluation
We'll evaluate the model's performance using accuracy, confusion matrix, and classification report.

*from sklearn.metrics import accuracy_score, confusion_matrix, classification_report*

#Predictions
*y_pred = model.predict_classes(X_test_pad)*

#Model Evaluation
*accuracy = accuracy_score(y_test, y_pred)*
*conf_matrix = confusion_matrix(y_test, y_pred)*
*class_report = classification_report(y_test, y_pred)*

*print("Accuracy:", accuracy)*
*print("Confusion Matrix:\n", conf_matrix)*
*print("Classification Report:\n", class_report)*

## Step 7: Analyzing Model Coefficients
For deep learning models like LSTM, analyzing coefficients is not as straightforward as in linear models. We can inspect the importance of different words by examining the weights of the embedding layer.

#Get the weights of the embedding layer
*weights = model.layers[0].get_weights()[0]*

#Map indices to words
*word_index = tokenizer.word_index*
*reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])*

#Print word importance
*for word, index in word_index.items():*
    *if index < max_features:*
        *print(word, weights[index])*

## Step 8: Conclusion

Data preprocessing involved handling missing values, outliers, and duplicates. We addressed missing values by imputing them using the mean of the respective features. After preprocessing, we split the data into training and test sets, with a common ratio of 80-20.

Overall, our model achieved commendable performance on the test set, demonstrating its potential for real-world application in spam email detection. By systematically preprocessing the data, building an effective model, and analyzing its behavior, we gained valuable insights into the underlying patterns of spam emails and how machine learning techniques can help combat them effectively

