#iris flower predictor
#Keep it simple and straightforward
#works perfectly

#import statements
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image,ImageFilter,ImageEnhance
import os




#Usual opening
st.markdown('''
# Iris Flower Species Predictor 
This app detects the type of Iris flower based on Machine Learning!
- App built by Pranav Sawant and Anshuman Shukla of Team Skillocity.
- Dataset credits: R.A. Fisher "The Use of Multiple Measurements in Taxonomic Problems" 
- This dataset is also available on the UC Irvine Machine Learning Repository
- Dataset License: Open Data Commons Public Domain Dedication and License (PDDL)
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.  
''')
st.write('---')

st.title('Iris Classifier and Predictor')
st.sidebar.header('Input Features')

#Inputs
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.2, 8.0, 4.4)
    sepal_width = st.sidebar.slider('Sepal width', 1.8, 4.2, 3.2)
    petal_length = st.sidebar.slider('Petal length', 0.8, 7.0, 1.6)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.4, 0.8)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
#start
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Iris Image Manipulation
@st.cache_resource
def load_image(img):
    im =Image.open(os.path.join(img))
    return im


#Prediction  with images
if iris.target_names[prediction] == 'setosa':
    st.text("Showing Setosa Species")
    st.image(load_image('iris_setosa.jpg'))
elif iris.target_names[prediction] == 'versicolor':
    st.text("Showing Versicolor Species")
    st.image(load_image('IRIS_VERSICOLOR.jpg'))
elif iris.target_names[prediction] == 'virginica':
    st.text("Showing Virginica Species")
    st.image(load_image('virginca.jpg'))

#article

st.sidebar.subheader("An article about this app: https://proskillocity.blogspot.com/2021/05/iris-classification-and-prediction.html")
image = Image.open('killocity (3).png')
st.image(image, use_column_width=True)
