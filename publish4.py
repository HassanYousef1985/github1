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
from flair.embeddings import WordEmbeddings, FlairEmbeddings
import gensim.downloader as api




def load_data():
        #Load the dataset
        data = pd.read_csv('tweetsofmyself1.csv', encoding='cp1252').fillna(' ')
        # preprocesing
        # rename the labels
        data['check-worthy'] = data['check-worthy'].replace(['yes'],1)
        data['check-worthy'] = data['check-worthy'].replace(['no'],0)
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


def main():
    st.title("Detecting Check Worthy Tweets WebApp")
    menu = ["Convolutions Neural Network (CNN)", "choice 2", "choice 3"]
    df=load_data()
    X = df['tweet'].values
    y = df['check-worthy'].values
    choice = st.sidebar.selectbox("Choose Classifier",menu)

    if choice == "Convolutions Neural Network (CNN)":
        GloVe = 50
        # Twitter has a relatively new 280-character limit.
        max_tweet_length = 280  
        max_features=20000
        col1,col2,col3 = st.beta_columns([8,2,2])
        with col1:
            st.subheader("The maximum length of a tweet is:")
        with col2:
            st.subheader(max_tweet_length)
        with col3:
            st.subheader("characters")
        col1,col2 = st.beta_columns([2,1])
        with col1:
            st.subheader("Max of features  that we are going to use :")
        with col2:
            st.subheader(max_features)
        
        # Tokenize and Pad Text Data
        x_tokenizer=text.Tokenizer(max_features)
        x_tokenizer.fit_on_texts(list(X))
        x_tokenized=x_tokenizer.texts_to_sequences(X)
        x_train_val=sequence.pad_sequences(x_tokenized,maxlen=max_tweet_length)

        # Prepare Embedding Matrix with Pre-trained GloVe Embeddings
        # Let's make a dict mapping words (strings) to their NumPy vector representation
        # if GloVe == '100':
            # embedding_dims = 100
        # else:
        embedding_dims = 50
        hits = 0
        misses = 0
        embeddings_index = dict()
        # if GloVe == '100':
            # f = open('glove.6B.100d.txt', encoding='utf8')
        # else:
        f = open('glove.6B.50d.txt', encoding='utf8')

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        # Now, let's prepare a corresponding embedding matrix that we can use in a Keras Embedding layer.
        # It's a simple NumPy matrix where entry at index i is the pre-trained vector for the word of index i in our vectorizer's vocabulary.
        embedding_matrix = np.zeros((max_features, embedding_dims))
        for word, index in x_tokenizer.word_index.items():
            if index > max_features -1:
                break
            else:
               embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
                hits += 1
            else:
                misses += 1
        col1,col2 = st.beta_columns([1,9])
        # with col1:
            # st.subheader("")
        # with col2:
            # st.write("Converted %d words (%d misses)" % (hits, misses))


        # Create the Embedding Layer
        model = Sequential()
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        # (we don't want to update them during training).
        model.add(Embedding(max_features,
                    embedding_dims,
                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                    trainable=False))
        model.add(Dropout(0.2))
        #  Build the model
        #  there are hyper parameters
        # filter is how many output channels this convolution layer has
        num_of_filters = st.sidebar.number_input("Number of filters",32 ,512 , 250, step = 1, key = 'num_of_filters')
        filters = num_of_filters
        kernel_Size = st.sidebar.radio("Kernel size", (3, 5), key = 'kernel_Size')
        kernel_size = kernel_Size
        hidden_dims = 250
        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu'))
        model.add(MaxPooling1D())
        model.add(Conv1D(filters,
                        5,
                        padding='valid',
                        activation='relu'))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())
        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims, activation='relu'))
        model.add(Dropout(0.2))
        # We project has only one output so we set the activation to sigmoid:
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        num_of_epochs = st.sidebar.number_input("Number of epochs",1 ,30 , 10, step = 1, key = 'num_of_epochs')
        if st.sidebar.button("Classify", key = 'classify'):
            batch_size = 32
            epochs = num_of_epochs
            # Train the model
            x_train, x_val, y_train, y_val = train_test_split(x_train_val, y, test_size=0.15, random_state=1)
            model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val))    
            loss,acc = model.evaluate(x_train, y_train, verbose = 2, batch_size = batch_size)       
            # val_loss,val_acc = model.evaluate(x_val, y_val, verbose = 2, batch_size = batch_size)       

            test_df = pd.read_csv('test.csv')
            x_test = test_df['tweet'].values
            # remove hyberlinks
            test_df['tweet'] = test_df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
            # test_df['tweet'] = test_df['tweet'].str.replace('https:\/\/.*', '', flags=re.UNICODE)
            # remove the word <link>
            test_df['tweet'] = test_df['tweet'].str.replace('<link>', '', flags=re.UNICODE)
            # remove emogis
            test_df['tweet'] = test_df['tweet'].apply(lambda x: re.split('[^\w\s#@/:%.,_-]', str(x))[0])
            # test_df['tweet'] = test_df['tweet'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
             # more cleaning (remove usernames-hashtags)
            test_df['tweet'] = test_df['tweet'].replace(to_replace=r'(@){1}.+?( ){1}',value='',regex=False)
            test_df['tweet'] = test_df['tweet'].replace(to_replace=r'(#){1}.+?( ){1}',value='',regex=False)
            # convert to lowercase
            test_df['tweet'] = test_df['tweet'].str.lower() 
            x_test_tokenized1 = x_tokenizer.texts_to_sequences(x_test)
            x_testing1 = sequence.pad_sequences(x_test_tokenized1, maxlen=max_tweet_length)

            y_testing1 = model.predict(x_testing1, verbose = 1, batch_size=32)
            test_loss,test_acc = model.evaluate(x_testing1, y_testing1, verbose = 2, batch_size = batch_size)       

            st.subheader("Resaults of Classification using CNN")
            # acc = accuracy_score(x_val, y_val)
            st.write("Accuracy:", acc)
            st.write("Loss:", loss)
            st.write("Validation Accuracy:", test_acc)
            st.write("Validation Loss:", test_loss)

            # Precision = precision_score(y_testing, y_val)
            # st.write("Accuracy:", Precision)
            st.subheader("Evaluate Model using Testing Set")
            test_df['check-worthy'] = ['not worthy' if x < .5 else 'worthy' for x in y_testing1]
            # st.write(test_df[['tweet', 'check-worthy']])
            st.table(test_df[['tweet', 'check-worthy']].style.highlight_max(axis=0))

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
                test_tweet = test_tweet.lower() 
                test_tweet_df = [test_tweet]
                x_test_tokenized2 = x_tokenizer.texts_to_sequences(test_tweet_df)
                x_testing2 = sequence.pad_sequences(x_test_tokenized2, maxlen=max_tweet_length)
                y_testing2 = model.predict(x_testing2)                 
                prediction = 'not worthy' if y_testing2 < 0.5 else 'worthy'
                col1,col2 = st.beta_columns([2,2])
                with col1:
                    st.info("Prediction")
                    st.write(prediction)
                with col2:
                    st.info("Probability")
                    st.write(y_testing2[0,0])

    if st.sidebar.checkbox("Show Training Dataset", False):
        st.subheader("Tweets Dataset - After Preprocessing -")
        # st.dataframe(df.style.highlight_max(axis=0),width=3000, height=400)
        st.table(df.style.highlight_max(axis=0))

    if st.sidebar.checkbox("Show Testing Dataset", False):
        st.subheader("Tweets Dataset - After Preprocessing -")
        # st.dataframe(df.style.highlight_max(axis=0),width=3000, height=400)
        st.table(test_df.style.highlight_max(axis=0))

if __name__ == '__main__':
    main()