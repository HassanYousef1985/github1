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
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import layers
from sklearn.metrics import precision_score, recall_score ,accuracy_score,f1_score

import matplotlib.pyplot as plt



def load_data():
        data = pd.read_csv('train3.csv', encoding="utf8").fillna(' ')
        # remove hyberlinks
        data['tweet'] = data['tweet'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)
        # remove the word <link>
        data['tweet'] = data['tweet'].replace(to_replace=r'<link>',value='',regex=True)
        # remove emogis
        data['tweet'] = data['tweet'].replace(to_replace=r'[^\w\s#@/:%.,_-]',value='',regex=True)
        # more cleaning (remove usernames-hashtags)
        data['tweet'] = data['tweet'].replace(to_replace=r'(@){1}.+?( ){1}',value='',regex=True)
        data['tweet'] = data['tweet'].replace(to_replace=r'(#){1}.+?( ){1}',value='',regex=True)
        # convert to lowercase
        data['tweet'] = data['tweet'].str.lower() 
        return data

def main():
    st.title("Detecting Check Worthy Tweets WebApp")
    menu = ["Convolutions Neural Network (CNN)", "Word2Vec and LSTM", "Random Forest"]
    df=load_data()
    X = df['tweet'].values
    y = df['check-worthy'].values
    choice = st.sidebar.selectbox("Choose Classifier",menu)
    if choice == "Convolutions Neural Network (CNN)":
        RANDOM_STATE = 42
        # Split train & test
        test_size_value = st.sidebar.radio("Test size", (0.1, 0.2, 0.3), key = 'test_size')
        text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, random_state=RANDOM_STATE)
        # Tokenize and transform to integer index
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_train)
        X_train = tokenizer.texts_to_sequences(text_train)
        X_test = tokenizer.texts_to_sequences(text_test)

        vocab_size =len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        maxlen = max(len(x) for x in X_train) # longest text in train set

        col1,col2 = st.beta_columns([2,1])
        with col1:
            st.subheader("Number of all unique words in train set:")
        with col2:
            st.subheader(vocab_size)


        # Add pading to ensure all vectors have same dimensionality
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

        embedding_dim = 100

        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        num_of_filters = st.sidebar.number_input("Number of filters",32 ,512 , 128, step = 1, key = 'num_of_filters')
        model.add(layers.Conv1D(num_of_filters, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        print(model.summary())

        num_of_epochs = st.sidebar.number_input("Number of epochs",1 ,30 , 10, step = 1, key = 'num_of_epochs')      

        if st.sidebar.button("Classify", key = 'classify'):
            # Fit model
            history = model.fit(X_train, y_train,
                                epochs=num_of_epochs,
                                verbose=True,
                                validation_data=(X_test, y_test),
                                batch_size=10)
            # loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
            st.subheader("Resaults of Classification using CNN")
            # st.write("Training Accuracy: {:.4f}".format(accuracy))
            loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
            st.write("Accuracy: ",accuracy)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred<0.5,0,1)
            st.write("Precision: ",precision_score(y_test,y_pred))
          

            # fig, ax = plt.subplots()
            
            # ax.scatter([1, 2, 3], [1, 2, 3])
            # st.pyplot(fig)

        # Evaluate Model
        with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Tweet to Predict:")        
            submitted = st.form_submit_button("Predict")     
            if submitted:
                # tweet = [test_tweet]
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
                test_tweet = test_tweet.lower() 
                test_tweet_df = [test_tweet]
                test_tweet = tokenizer.texts_to_sequences(test_tweet_df)
                x_testing = sequence.pad_sequences(test_tweet, maxlen=maxlen)
                y_testing = model.predict(x_testing)
                prediction = 'not worthy' if y_testing < 0.5 else 'worthy'
                col1,col2 = st.beta_columns([2,2])
                with col1:
                    st.info("Prediction")
                    st.write(prediction)
                with col2:
                    st.info("Prediction Probability")
                    st.write(y_testing[0,0])


    if st.sidebar.checkbox("Show Tweets Dataset", False):
        st.subheader("Tweets Dataset - After Preprocessing -")
        st.write("(1=check-worthy, 0=not-check-worthy)")
        # st.dataframe(df.style.highlight_max(axis=0),width=3000, height=400)
        st.table(df.style.highlight_max(axis=0))

if __name__ == '__main__':
    main()