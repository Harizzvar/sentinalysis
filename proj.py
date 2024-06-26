import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


np.random.seed(500)

data = pd.read_csv('training.csv',encoding='latin1')
#print(data['Sentence'])
data.dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
data['Sentence'] = [entry.lower() for entry in data['Sentence']]
data['Sentiment'] = np.where(data['Sentiment'].str.contains('positive'), 1, 0)

#print(len(data['Sentence']))
Train_X, Test_X, Train_Y, Test_Y = train_test_split(data['Sentence'],data['Sentiment'],test_size=0.3)
#70-30

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


d = pd.read_csv("stopwords.csv")
my_stopword=d.values.tolist()
#vectorizer = TfidfVectorizer(stop_words=[word[0] for word in my_stopword])
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit_transform(data['Sentence'])
feature_names = vectorizer.get_feature_names_out()
#print(response)
Train_X_Tfidf = vectorizer.transform(Train_X)
Test_X_Tfidf = vectorizer.transform(Test_X)
print(Train_X_Tfidf.shape,Train_Y.shape)

#svm
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
#print(Test_Y,predictions_SVM)

# Naive Bayes Model
NB = MultinomialNB()
NB.fit(Train_X_Tfidf, Train_Y)
predictions_NB = NB.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)

print("Enter sentences: ")
sentences = []
for i in range(2):
    sentence = input()
    sentences.append(sentence)

test_data = vectorizer.transform(sentences)

predictions_SVM = SVM.predict(test_data)
predictions_NB = NB.predict(test_data)

for prediction in predictions_SVM:
    if prediction == 1:
        print("---- SVM: positive")
    else:
        print("---- SVM: negative")

for prediction in predictions_NB:
    if prediction == 1:
        print("---- Naive Bayes: positive")
    else:
        print("---- Naive Bayes: negative")
'''lst = [ ] 
print("Enter sentences: ")
  
for i in range(0, 2): 
    ele = input()
    lst.append(ele) 
      
#print(lst) 
tes=vectorizer.transform(lst)
#print(tes)
predictions= SVM.predict(tes)
#print(predictions)
for i in predictions:
        if i == 1 :
            print("---- positive")
        elif i == 0:
            print("---- negative")'''