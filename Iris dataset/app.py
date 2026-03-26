import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


iris = load_iris()

X = iris.data
y = iris.target


model = LogisticRegression(max_iter=200)
model.fit(X, y)


st.title("Iris Flower Prediction")


sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.number_input("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.number_input("Petal Width", 0.1, 2.5, 0.2)


if st.button("Predict"):

    sample = np.array([[sepal_length,
                        sepal_width,
                        petal_length,
                        petal_width]])

    prediction = model.predict(sample)

    flower = iris.target_names[prediction][0]

    st.success("Prediction: " + flower)
