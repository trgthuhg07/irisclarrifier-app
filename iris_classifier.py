from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle as pickle

iris = load_iris()
X, y =iris.data, iris.target

print(X.shape)
#print(y.shape)

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)

clf=RandomForestClassifier()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

print('Saving model to pickle file')
pickle.dump(clf, open("iris_model.pkl", 'wb'))

#new_input=[5, 3.1,6,2].reshape(-1,1)
#print(clf.predict(new_input))
