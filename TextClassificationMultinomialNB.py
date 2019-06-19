import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv('training.1600000.processed.noemoticon.csv', sep=',', encoding='latin-1')
df = df.rename(columns={df.columns[0]: 'target',
                        df.columns[1]: 'ids',
                        df.columns[2]: 'date',
                        df.columns[3]: 'flag',
                        df.columns[4]: 'user',
                        df.columns[5]: 'text'})
df.head()

# Check for the existence of NaN values in a cell:
df.isnull().sum()

df.dropna(inplace=True)

print(df['target'].value_counts())

len(df)

blanks = []  # start with an empty list

for index,target,ids,date,flag,user,text  in df.itertuples():  # iterate over the DataFrame
    if type(text) == str:  # avoid NaN values
        if text.isspace():  # test 'review' for whitespace
            blanks.append(index)  # add matching index numbers to the list

df.drop(blanks, inplace=True)

len(df)

df['target'].value_counts()

from sklearn.model_selection import train_test_split

X = df['text']
y = df['target']


chart_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
chart_y_train = []
chart_y_test = []
for test_size in chart_x:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    # Naïve Bayes:
    text_clf_nb = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3))), ('clf', MultinomialNB()),
    ])


    text_clf_nb.fit(X_train, y_train)

    # Form a prediction set
    predictions = text_clf_nb.predict(X_test)
    predictionsTrain = text_clf_nb.predict(X_train)

    # Report the confusion matrix
    from sklearn import metrics
    print(metrics.confusion_matrix(y_test,predictions))

    # Print a classification report
    print(metrics.classification_report(y_test,predictions))

    # Print the overall accuracy
    # print('Acuracia do teste:')
    # print(metrics.accuracy_score(y_test,predictions))
    # print('Acuracia do treino:')
    # print(metrics.accuracy_score(y_train,predictionsTrain))
    chart_y_train.append(metrics.accuracy_score(y_train,predictionsTrain))
    chart_y_test.append(metrics.accuracy_score(y_test,predictions))

plt.plot(chart_x, chart_y_train, color='g')
plt.plot(chart_x, chart_y_test, color='orange')
plt.xlabel('test size')
plt.ylabel('accuracy')
plt.title('Multinomial Naive Bayes perfomance - ngram_range=(1, 3)')
# plt.show()
plt.savefig('multinomialNB13.jpg')