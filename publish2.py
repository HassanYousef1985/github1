import pandas as pd
from sklearn import model_selection 
import streamlit as st 
from collections import Counter
from tensorflow.keras.preprocessing import text, sequence
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import re
from sklearn.metrics import precision_score, recall_score ,accuracy_score,f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dropout, Dense,Input,Embedding,Flatten, MaxPooling1D, Conv1D
from tensorflow.keras.models import Sequential,Model
from sklearn import metrics
from tensorflow.python.keras.layers.merge import Concatenate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def plot_metrics(metrics_list, model, X_test, y_test, class_names):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels = class_names)
            st.pyplot()
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()

def load_data():
        #Load the dataset
        # data = pd.read_csv('tweetsofmyself1.csv', encoding='cp1252').fillna(' ')
        # preprocesing
        # rename the labels
        # data['check-worthy'] = data['check-worthy'].replace(['yes'],1)
        # data['check-worthy'] = data['check-worthy'].replace(['no'],0)
        data = pd.read_csv('train3.csv', encoding="utf8").fillna(' ')
        # remove hyberlinks
        data['tweet'] = data['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        # data['tweet'] = data['tweet'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)
        # remove the word <link>
        data['tweet'] = data['tweet'].replace(to_replace=r'<link>',value='',regex=True)
        # remove emogis
        data['tweet'] = data['tweet'].apply(lambda x: re.split('[^\w\s#@/:%.,_-]', str(x))[0])
        # data['tweet'] = data['tweet'].replace(to_replace=r'[^\w\s#@/:%.,_-]',value='',regex=True)
        # more cleaning (remove usernames-hashtags)
        data['tweet'] = data['tweet'].replace(to_replace=r'(@){1}.+?( ){1}',value='',regex=True)
        data['tweet'] = data['tweet'].replace(to_replace=r'(#){1}.+?( ){1}',value='',regex=True)
        # convert to lowercase
        data['tweet'] = data['tweet'].str.lower() 
        return data



@st.cache(persist = True)
def split(df, test_size_value):
        X = df['tweet']
        y = df['check-worthy']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_value, random_state = 0)
        return X_train, X_test, y_train, y_test


def main():
    st.title("Predicting check worthy tweet")
    st.markdown("This app is created to predict if a tweet is check worthy or not")
    st.markdown("Build your model first then predict your tweet!")

    classifier = ["Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"]
    df=load_data()
    class_names = ['worthy', 'not worthy']

    choice = st.sidebar.selectbox("Choose Classifier",classifier)
    test_size_value = st.sidebar.radio("Test size", (0.1, 0.2, 0.3), key = 'test_size')
    X_train, X_test, y_train, y_test = split(df,test_size_value)
    if choice == "Support Vector Machine (SVM)":
        text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])
      
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        # num_of_epochs = st.sidebar.number_input("Number of epochs",1 ,30 , 10, step = 1, key = 'num_of_epochs')
        if st.sidebar.button("Build", key = 'classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model=text_clf.fit(X_train, y_train)
            y_pred = text_clf.predict(X_test)
            accuracy = text_clf.score(X_test, y_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics,model,X_test,y_test,class_names)        

    if choice == "Logistic Regression":
    
        text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression()),
                     ])
      
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        # num_of_epochs = st.sidebar.number_input("Number of epochs",1 ,30 , 10, step = 1, key = 'num_of_epochs')
        if st.sidebar.button("Build", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model=text_clf.fit(X_train, y_train)
            y_pred = text_clf.predict(X_test)
            accuracy = text_clf.score(X_test, y_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics,model,X_test,y_test,class_names)       

    if choice == "Random Forest":
    
        text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier()),
                     ])
      
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        # num_of_epochs = st.sidebar.number_input("Number of epochs",1 ,30 , 10, step = 1, key = 'num_of_epochs')
        if st.sidebar.button("Build", key = 'classify'):
            st.subheader("Random Forest Results")
            model=text_clf.fit(X_train, y_train)
            y_pred = text_clf.predict(X_test)
            accuracy = text_clf.score(X_test, y_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics,model,X_test,y_test,class_names)    

    
        # Evaluate Model
    with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Tweet:")        
            submitted = st.form_submit_button("Predict")     
            if submitted:

                # remove hyberlinks
                test_tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', test_tweet, flags=re.MULTILINE)
                # remove the word <link>
                test_tweet = re.sub(r'<link>', '', test_tweet, flags=re.MULTILINE)
                # remove emogis
                test_tweet = re.sub(r'[^\w\s#@/:%.,_-]', '', test_tweet, flags=re.MULTILINE)
                # more cleaning (usernames-hashtags)
                test_tweet = re.sub(r'(@){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                test_tweet = re.sub(r'(#){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                # convert to lowercase
               

    if st.sidebar.checkbox("Show Training Dataset", False):
        st.subheader("Tweets Dataset - After Preprocessing -")
        # st.dataframe(df.style.highlight_max(axis=0),width=3000, height=400)
        st.table(df.style.highlight_max(axis=0))

    

if __name__ == '__main__':
    main()