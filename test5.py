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
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import matplotlib.pyplot as plt

def load_data():
        data = pd.read_csv('Tweetsofmyself1.csv', encoding='cp1252').fillna(' ')
        # rename the labels
        data['check-worthy'] = data['check-worthy'].replace(['yes'],1)
        data['check-worthy'] = data['check-worthy'].replace(['no'],0)
        # remove unused colomn 
        data = data.drop('claim',axis=1)
        # remove hyberlinks
        data['tweet'] = data['tweet'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)
        # remove the word <link>
        data['tweet'] = data['tweet'].replace(to_replace=r'<link>',value='',regex=True)
        # remove emogis
        data['tweet'] = data['tweet'].replace(to_replace=r'[^\w\s#@/:%.,_-]',value='',regex=True)
        # more cleaning (usernames-hashtags)
        data['tweet'] = data['tweet'].replace(to_replace=r'(@){1}.+?( ){1}',value='',regex=True)
        data['tweet'] = data['tweet'].replace(to_replace=r'(#){1}.+?( ){1}',value='',regex=True)
        # convert to lowercase
        data['tweet'] = data['tweet'].str.lower() 
        return data

# def plot_metrics(metrics_list, model):
#         if 'loss' in metrics_list:
#             plt.plot(model["loss"])

        # if 'ROC Curve' in metrics_list:
        #     st.subheader("ROC Curve")
        #     plot_roc_curve(model, x_val, y_val)
        #     st.pyplot()

        # if 'Precision-Recall Curve' in metrics_list:
        #     st.subheader("Precision-Recall Curve")
        #     plot_precision_recall_curve(model, x_val, y_val)
        #     st.pyplot()

def main():
    st.title("Detecting Check Worthy Tweets WebApp")
    menu = ["Convolutions Neural Network (CNN)", "Word2Vec and LSTM", "Random Forest"]
    df=load_data()
    X = df['tweet'].values
    y = df['check-worthy'].values
    choice = st.sidebar.selectbox("Choose Classifier",menu)

    if choice == "Convolutions Neural Network (CNN)":
        GloVe = st.sidebar.radio("GloVe (Global Vectors for Word Representation)", ("100", "50"), key = 'GloVe')
        max_text_length = df['tweet'].str.len().max()
        max_features=len(Counter(" ".join(df['tweet'].str.lower().values.tolist()).split(" ")).items())
        col1,col2,col3 = st.beta_columns([8,2,2])
        with col1:
            st.subheader("Length of longest tweet in the dataframe:")
        with col2:
            st.subheader(max_text_length)
        with col3:
            st.subheader("characters")
        col1,col2 = st.beta_columns([2,1])
        with col1:
            st.subheader("Number of all unique words in the dataframe:")
        with col2:
            st.subheader(max_features)
        
        # Tokenize and Pad Text Data
        x_tokenizer=text.Tokenizer(max_features)
        x_tokenizer.fit_on_texts(list(X))
        x_tokenized=x_tokenizer.texts_to_sequences(X)
        x_train_val=sequence.pad_sequences(x_tokenized,maxlen=max_text_length)

        # Prepare Embedding Matrix with Pre-trained GloVe Embeddings
        # Let's make a dict mapping words (strings) to their NumPy vector representation
        if GloVe == '100':
            embedding_dims = 100
        else:
            embedding_dims = 50
        hits = 0
        misses = 0
        embeddings_index = dict()
        if GloVe == '100':
            f = open('glove.6B.100d.txt', encoding='utf8')
        else:
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
                # st.write(word)
                misses += 1
        col1,col2 = st.beta_columns([1,9])
        with col1:
            st.subheader("")
        with col2:
            st.write("Converted %d words (%d misses)" % (hits, misses))


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

        # metrics = st.sidebar.multiselect("What metrics to plot?", ('loss', 'ROC Curve', 'Precision-Recall Curve'))


        if st.sidebar.button("Classify", key = 'classify'):
            batch_size = 32
            epochs = num_of_epochs
            # Train the model
            x_train, x_val, y_train, y_val = train_test_split(x_train_val, y, test_size=0.15, random_state=1)
            history=model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val))   
            st.subheader("Resaults of Classification using CNN")

            y_pred = model.predict(x_val)

            
            
            # evaluate the model
            # st.write("Accuracy: ", accuracy_score(x_val, y_val))
            st.write("Precision: ", precision_score(y_val, y_pred))


            # plot loss during training
            fig, ax = plt.subplots()
            
            ax.scatter([1, 2, 3], [1, 2, 3])
            st.pyplot(fig)
            # pyplot.plot(history.history['loss'], label='train')
            # pyplot.plot(history.history['val_loss'], label='test')
            # pyplot.legend()
            # # plot accuracy during training
            # pyplot.subplot(212)
            # pyplot.title('Accuracy')
            # pyplot.plot(history.history['accuracy'], label='train')
            # pyplot.plot(history.history['val_accuracy'], label='test')
            # pyplot.legend()
            # plt.show()

            # accuracy = model.score(x_val, y_val)
            # LR = linear_model.LinearRegression()
            # y_pred=LR.predict(x_val)
            # print(y_pred)
            # y_pred = model.predict(x_val)
            # cm_dtc=confusion_matrix(y_val,y_pred)
            # st.write(cm_dtc)

            # st.write("Accuracy: ", accuracy.round(2))
            # st.write("Precision: ", precision_score(y_val, y_pred, labels = [1,0]).round(2))
            # st.write("Recall: ", recall_score(y_val, y_pred, labels = [1,0]).round(2))
            # plot_metrics(metrics,history)
            # st.plotly_chart(confusion_matrix(y_val, y_pred))


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
                x_test_tokenized = x_tokenizer.texts_to_sequences(test_tweet_df)
                x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=max_text_length)
                y_testing = model.predict(x_testing)                 
                prediction = 'not worthy' if y_testing < 0.5 else 'worthy'
                col1,col2 = st.beta_columns([2,2])
                with col1:
                    st.info("Prediction")
                    st.write(prediction)
                with col2:
                    st.info("Probability")
                    st.write(y_testing[0,0])

    if st.sidebar.checkbox("Show Tweets Dataset", False):
        st.subheader("Tweets Dataset - After Preprocessing -")
        # st.dataframe(df.style.highlight_max(axis=0),width=3000, height=400)
        st.table(df.style.highlight_max(axis=0))

if __name__ == '__main__':
    main()