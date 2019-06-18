import numpy as np
import pandas as pd


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

for index, target, ids, date, flag,user, text  in df.itertuples():  # iterate over the DataFrame
    if type(text) == str:  # avoid NaN values
        if text.isspace():  # test 'review' for whitespace
            blanks.append(index)  # add matching index numbers to the list
df.drop(blanks, inplace=True)

len(df)

df['target'].value_counts()

from sklearn.model_selection import train_test_split

X = df['text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Linear SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                     ('clf', LinearSVC()),
])

# Next we'll run Linear SVC
text_clf_lsvc.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf_lsvc.predict(X_test)

# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))