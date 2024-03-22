import streamlit as st
import pickle
#from sklearn.datasets import load_iris
import pandas as pd

# Đường link đến dữ liệu trên GitHub
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
# Đọc dữ liệu vào DataFrame
iris = pd.read_csv(url)

#iris = load_iris()
#####
# Load the trained model

clf = pickle.load(open('iris_model.pkl', 'rb'))

# Sidebar for user input
st.sidebar.title('Iris Classifier')
sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)

# Make predictions
prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
# Display prediction
st.write('## Prediction:') 
st.write(iris.target_names[prediction[0]])
