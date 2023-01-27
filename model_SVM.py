import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


df = pd.read_csv('diabetes.csv')

# separate the attributes in the dataset and stores them in a variable
X = df[df.columns[:8]]

# separate the labels in the dataset and stores them in a variable
y = df['Outcome']

# standardize the values of the dataset
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# create an SVC object and train the model
clf = SVC()
clf.fit(X_train, y_train)

# calculate the accuracy score of the data train
clf.score(X_train, y_train)

# calculate the accuracy score of prediction
clf.score(X_test, y_test)
