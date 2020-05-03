import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import Image
import warnings

warnings.filterwarnings("ignore")

#loading the dataset using pandas
data = pd.read_csv('spam.csv', encoding='latin-1')
print(data.head(n=10))

#plotting the dataset into a bar graph to see the skewness
count_Class=pd.value_counts(data["v1"], sort= True)
count_Class.plot(kind = 'bar',color = ["green","red"])
plt.title('Bar Plot')
plt.show();

#feature engineering without TfidVectorizer
# count1 = Counter(" ".join(data[data['v1']=='ham']["v2"]).split()).most_common(20)
# df1 = pd.DataFrame.from_dict(count1)
# df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
# count2 = Counter(" ".join(data[data['v1']=='spam']["v2"]).split()).most_common(20)
# df2 = pd.DataFrame.from_dict(count2)
# df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})


#TfidVectorizer combines Tfidtransformer and CountVectorizer
#creating an instance of TfidVectorizer
f = feature_extraction.text.TfidfVectorizer(stop_words = 'english')

#fitting the dataset into the vectorizer instance
X = f.fit_transform(data["v2"])

#shaping the data obtained from vectorizer into an array using numpy
np.shape(X)
print(X)

#labling the result column ,i.e., spam and ham, into a boolean value
data["v1"]=data["v1"].map({'spam':1,'ham':0})

#splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)

#creating a list of alpha values to check which is the best fit
list_alpha = np.arange(1/100000, 20, 0.11)

#creating vectors for storing the scores obtained by using various values of alpha
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))

#we use the recall and precision tests because our data is skewed and not equally distributed
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))

#for various values of aplha we will fit the training dataset and predict the values for test set
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    #how many values were predicted correctly for the training and the testing data
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1

#we will convert the calculated data into a matrix with headings and print the values
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns =
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print(models.head(n=10))

#we will store the value for which the precision test has best value
best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]
models[models['Test Precision']==1].head(n=5)
best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()

#we will train our model using the training data again for the best value of alpha
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
# models.iloc[best_index, :]

#we will plot a confusion matrix after predicting the output on out test set
#confusion matrix consists of True Positive, True Negative, False Positive and False Negative predicted values
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1']))

#now we can use our own sample dataset and run on our trained model to check for the output
str=input("Enter your mail to check whether it is spam or not")
data={'Spam':str}
df=pd.DataFrame([data], index=['v1']).T
df.to_csv('sample_input.csv')
data1 = pd.read_csv('sample_input.csv', encoding='latin-1')
X = f.transform(data1["v1"])
np.shape(X)
print(bayes.predict(X))
