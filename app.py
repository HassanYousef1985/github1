import pandas as pd
import streamlit as st 
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import layers
import re                             
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import RandomizedSearchCV

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

@st.cache(persist = True)
def load_data():
    #Load the dataset
    data = pd.read_csv('data.csv')
    # rename the labels
    data['check-worthy'] = data['check-worthy'].replace(['yes'],1)
    data['check-worthy'] = data['check-worthy'].replace(['no'],0)
    # delete unused column "claim"    
    data = data.drop('claim', 1)

    # remove hyberlinks
    data['tweet'] = data['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    # remove the word <link>
    data['tweet'] = data['tweet'].replace(to_replace=r'<link>',value='',regex=True)
    # remove emogis
    data['tweet'] = data['tweet'].apply(lambda x: re.split('[^\w\s#@/:%.,_-]', str(x))[0])
    # more cleaning (remove usernames-hashtags)
    data['tweet'] = data['tweet'].replace(to_replace=r'(@){1}.+?( ){1}',value='',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r'(#){1}.+?( ){1}',value='',regex=True)
    # convert to lowercase
    data['tweet'] = data['tweet'].str.lower() 
    return data

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

def load_data1():
        #Load the dataset
    data = pd.read_csv('data1.csv')
    
    # remove hyberlinks
    data['tweet'] = data['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    # remove the word <link>
    data['tweet'] = data['tweet'].replace(to_replace=r'<link>',value='',regex=True)
    # remove emogis
    data['tweet'] = data['tweet'].apply(lambda x: re.split('[^\w\s#@/:%.,_-]', str(x))[0])
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


# @st.cache(allow_output_mutation=True)
# def load_model(): 
#     model_weights = "model.h5"
#     model_json = "model.json"
#     with open(model_json) as json_file:
#         loaded_model = model_from_json(json_file.read())
#     loaded_model.load_weights(model_weights)
#     loaded_model.summary()  # included to make it visible when model is reloaded
#     return loaded_model


def main():

    st.title("Predicting Check Worthy Tweet")
    st.markdown("This app is created to predict if a tweet is check worthy or not in the domain of Corona Virus")
    st.markdown("Please choose your classifier model first then classify your tweet!")

    classifier = [ "Choose Classifier","Logistic Regression", "Word Embeddings", "Pretrained Word Embeddings", "CNN - Type1", "CNN - Type1 (optimization)"]
    choice = st.sidebar.selectbox("Choose Classifier",classifier)
   
    # df =load_data()
    df=load_data()
    # df = pd.concat([df, df1])

    test_size_value = st.sidebar.radio("Test size: (Default is 30%)", (0.1, 0.2, 0.3), 2, key = 'test_size')
    
    class_names = ['worthy', 'not worthy']


  

    if choice == "Logistic Regression":
        X_train, X_test, y_train, y_test = split(df,test_size_value)
        vectorizer = CountVectorizer()
        vectorizer.fit(X_train)

        X_train = vectorizer.transform(X_train)
        X_test  = vectorizer.transform(X_test)

        lr_clf = LogisticRegression()            
        st.subheader("Classifier Metrics - Logistic Regression:")
        lr_clf.fit(X_train, y_train)

        y_pred = lr_clf.predict(X_test)
        st.write("Training Accuracy: {:.2f}".format(lr_clf.score(X_train, y_train)))
        st.write("Testing Accuracy: {:.2f}".format(lr_clf.score(X_test, y_test)))
        # st.write("Precision: {:.4f}".format(precision_score(y_test, y_pred, labels = class_names)))
        # st.write("Recall: {:.4f}".format(recall_score(y_test, y_pred, labels = class_names)))

        

                                        # st.subheader("Confusion Matrix")
                                        # plot_confusion_matrix(lr_clf, X_test, y_test, display_labels = class_names)
                                        # st.pyplot()

                                        # st.subheader("ROC Curve")
                                        # plot_roc_curve(lr_clf, X_test, y_test)
                                        # st.pyplot()

                                        # st.subheader("Precision-Recall Curve")
                                        # plot_precision_recall_curve(lr_clf, X_test, y_test)
                                        # st.pyplot()

        with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Own Tweet:")        
            submitted = st.form_submit_button("Classify")     
            if submitted:
                if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                else:
                    if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        # remove hyberlinks
                        test_tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', test_tweet, flags=re.MULTILINE)
                        # remove the word <link>
                        test_tweet = re.sub(r'<link>', '', test_tweet, flags=re.MULTILINE)
                        # remove emogis
                        test_tweet = re.sub(r'[^\w\s#@/:%.,_-]', '', test_tweet, flags=re.MULTILINE)
                        # more cleaning (usernames-hashtags)
                        test_tweet = re.sub(r'(@){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                        test_tweet = re.sub(r'(#){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                        # # convert to lowercase
                        test_tweet = test_tweet.lower() 

                        test_tweet_df = [test_tweet]
                        X_test_sample  = vectorizer.transform(test_tweet_df)
                        y_test = lr_clf.predict(X_test_sample)              

                        prediction = 'Not check-worthy' if y_test[0] == 0 else 'Check-worthy'
                        col1,col2 = st.columns([2,2])
                        with col1:    
                            st.info("Prediction")
                            st.write(prediction)
                        with col2:
                            st.info("% Confidence")
                            if prediction == 'Not check-worthy' : st.write("≈ {:.0f}".format(lr_clf.predict_proba(X_test_sample)[0,0]*100))
                            else                                : st.write("≈ {:.0f}".format(lr_clf.predict_proba(X_test_sample)[0,1]*100))  

  



                           # @st.cache
                                # def __calculate_score(y_pred_class, y_pred_prob):
                                # if y_pred_class == 0:
                                #     MAX = 0.5
                                #     scaled_percentage = (y_pred_prob * MAX) / 100
                                #     return MAX - scaled_percentage
                                # else:
                                #     MAX = 1
                                #     scaled_percentage = (y_pred_prob * MAX) / 100
                                #     return scaled_percentage   



    if choice == "Word Embeddings":
        X_train, X_test, y_train, y_test = split(df,test_size_value)
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)


        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        maxlen = 280

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        embedding_dim = 50

        word_embeddings_clf = Sequential()
        word_embeddings_clf.add(layers.Embedding(input_dim=vocab_size, 
                                output_dim=embedding_dim, 
                                input_length=maxlen))
        word_embeddings_clf.add(layers.GlobalMaxPool1D())
        # seq_clf.add(layers.Flatten())
        word_embeddings_clf.add(layers.Dense(10, activation='relu'))
        word_embeddings_clf.add(layers.Dense(1, activation='sigmoid'))
        word_embeddings_clf.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        # model.summary()

        history = word_embeddings_clf.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
        st.subheader("Classifier Metrics - Sequential Model with Word Embeddings:")
        y_pred = word_embeddings_clf.predict(X_test)

        loss, accuracy = word_embeddings_clf.evaluate(X_train, y_train, verbose=False)
        st.write("Training Accuracy: {:.2f}".format(accuracy))
        loss, accuracy = word_embeddings_clf.evaluate(X_test, y_test, verbose=False)
        st.write("Testing Accuracy:  {:.2f}".format(accuracy))
        # st.write("Precision: {:.4f}".format(precision_score(y_test, y_pred, labels = class_names)))


        with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Own Tweet:")        
            submitted = st.form_submit_button("Classify")     
            if submitted:
                if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                else:
                    if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        # remove hyberlinks
                        test_tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', test_tweet, flags=re.MULTILINE)
                        # remove the word <link>
                        test_tweet = re.sub(r'<link>', '', test_tweet, flags=re.MULTILINE)
                        # remove emogis
                        test_tweet = re.sub(r'[^\w\s#@/:%.,_-]', '', test_tweet, flags=re.MULTILINE)
                        # more cleaning (usernames-hashtags)
                        test_tweet = re.sub(r'(@){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                        test_tweet = re.sub(r'(#){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                        # # convert to lowercase
                        test_tweet = test_tweet.lower() 


                        test_tweet_df = [test_tweet]
                        X_test_sample = tokenizer.texts_to_sequences(test_tweet_df)
                        X_test_sample = pad_sequences(X_test_sample, padding='post', maxlen=maxlen)
                        # sample_to_predict = np.array(X_test_sample)

                        y_test = word_embeddings_clf.predict(X_test_sample)                 
                        # y_test1 = seq_clf.predict(sample_to_predict)                 

                        # st.write(y_test)
                        # st.write(y_test1)

                        # st.write(y_test[0]*100)


                        prediction = 'Not check-worthy' if y_test[0]*100 < 50 else 'Check-worthy'
                        col1,col2 = st.columns([2,2])
                        with col1:    
                            st.info("Prediction")
                            st.write(prediction)
                        with col2:
                            st.info("% Confidence")
                            st.write("≈ {:.0f}".format(    (y_test[0]*100)[0]   ) )
                            
                            # st.write("≈ {:.0f}".format(lr_clf.predict_proba(X_test_sample)[0,0]*100))
                            # if prediction == 'not worthy' : st.write((y_test[0]*100)[0])
                            # else                          : st.write((y_test[0]*100)[0])


    if choice == "Pretrained Word Embeddings":
        X_train, X_test, y_train, y_test = split(df,test_size_value)
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        maxlen = 280

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        embedding_dim = 50
        embedding_matrix = create_embedding_matrix(
             'glove.6B.50d.txt',
             tokenizer.word_index, embedding_dim)

        pretrained_embeddings_clf = Sequential()
        pretrained_embeddings_clf.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
        pretrained_embeddings_clf.add(layers.GlobalMaxPool1D())
        pretrained_embeddings_clf.add(layers.Dense(10, activation='relu'))
        pretrained_embeddings_clf.add(layers.Dense(1, activation='sigmoid'))
        pretrained_embeddings_clf.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        pretrained_embeddings_clf.summary()

        history = pretrained_embeddings_clf.fit(X_train, y_train,
                            epochs=10,
                            verbose=False,
                            validation_data=(X_test, y_test),
                            batch_size=10)
        
        
        st.subheader("Classifier Metrics - Sequential model with Pretrained Word Embeddings:")
        y_pred = pretrained_embeddings_clf.predict(X_test)

        loss, accuracy = pretrained_embeddings_clf.evaluate(X_train, y_train, verbose=False)
        st.write("Training Accuracy: {:.2f}".format(accuracy))
        loss, accuracy = pretrained_embeddings_clf.evaluate(X_test, y_test, verbose=False)
        st.write("Testing Accuracy:  {:.2f}".format(accuracy))      

        with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Own Tweet:")        
            submitted = st.form_submit_button("Classify")     
            if submitted:
                if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                else:
                    if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        # remove hyberlinks
                        test_tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', test_tweet, flags=re.MULTILINE)
                        # remove the word <link>
                        test_tweet = re.sub(r'<link>', '', test_tweet, flags=re.MULTILINE)
                        # remove emogis
                        test_tweet = re.sub(r'[^\w\s#@/:%.,_-]', '', test_tweet, flags=re.MULTILINE)
                        # more cleaning (usernames-hashtags)
                        test_tweet = re.sub(r'(@){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                        test_tweet = re.sub(r'(#){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                        # # convert to lowercase
                        test_tweet = test_tweet.lower() 


                        test_tweet_df = [test_tweet]
                        X_test_sample = tokenizer.texts_to_sequences(test_tweet_df)
                        X_test_sample = pad_sequences(X_test_sample, padding='post', maxlen=maxlen)

                        y_test = pretrained_embeddings_clf.predict(X_test_sample)                 
                        # y_test1 = seq_clf.predict(sample_to_predict)                 

                        # st.write(y_test)
                        # st.write(y_test1)

                        # st.write(y_test[0]*100)


                        prediction = 'Not check-worthy' if y_test[0]*100 < 50 else 'Check-worthy'
                        col1,col2 = st.columns([2,2])
                        with col1:    
                            st.info("Prediction")
                            st.write(prediction)
                        with col2:
                            st.info("% Confidence")
                            st.write("≈ {:.0f}".format(    (y_test[0]*100)[0]   ) )
                            # if prediction == 'not worthy' : st.write((y_test[0]*100)[0])
                            # else                          : st.write((y_test[0]*100)[0])
                            

   


    if choice == "CNN - Type1":
            X_train, X_test, y_train, y_test = split(df,test_size_value)
            # Tokenize and transform to integer index
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(X_train)

            X_train = tokenizer.texts_to_sequences(X_train)
            X_test = tokenizer.texts_to_sequences(X_test)

            vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
            maxlen = max(len(x) for x in X_train) # longest text in train set

            # # Add pading to ensure all vectors have same dimensionality
            X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
            X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
            # # Define CNN architecture

            embedding_dim = 100

            # cnn_clf = Sequential()
            # cnn_clf.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
            # cnn_clf.add(layers.Conv1D(128, 5, activation='relu'))
            # cnn_clf.add(layers.GlobalMaxPooling1D())
            # cnn_clf.add(layers.Dense(10, activation='relu'))
            # cnn_clf.add(layers.Dense(1, activation='sigmoid'))
            # cnn_clf.compile(optimizer='adam',
            #             loss='binary_crossentropy',
            #             metrics=['accuracy'])
            # print(cnn_clf.summary())

            # # Fit model
            # history = cnn_clf.fit(X_train, y_train,
            #                     epochs=5,
            #                     verbose=True,
            #                     validation_data=(X_test, y_test),
            #                     batch_size=10)

            # model_json = cnn_clf.to_json()
            # with open("model.json", "w") as json_file:
            #     json_file.write(model_json)
            # cnn_clf.save_weights("model.h5")

            # cnn_clf.save('my_model.h5')
            st.subheader("Classifier Metrics - Convolutions Neural Network (CNN) (Type1):")
        
            # y_pred = cnn_clf.predict(X_test)


            # loss, accuracy = cnn_clf.evaluate(X_train, y_train, verbose=True)
            # st.write("Training Accuracy: {:.2f}".format(accuracy))
            # loss, accuracy = cnn_clf.evaluate(X_test, y_test, verbose=False)
            # st.write("Testing Accuracy:  {:.2f}".format(accuracy))
        
           
            with st.form("my_form"):
                test_tweet = st.text_area("Enter Your Own Tweet:")        
                submitted = st.form_submit_button("Classify")     
                if submitted:
                    if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                        else:
                            # remove hyberlinks
                            test_tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', test_tweet, flags=re.MULTILINE)
                            # remove the word <link>
                            test_tweet = re.sub(r'<link>', '', test_tweet, flags=re.MULTILINE)
                            # remove emogis
                            test_tweet = re.sub(r'[^\w\s#@/:%.,_-]', '', test_tweet, flags=re.MULTILINE)
                            # more cleaning (usernames-hashtags)
                            test_tweet = re.sub(r'(@){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                            test_tweet = re.sub(r'(#){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
                            # # convert to lowercase
                            test_tweet = test_tweet.lower() 


                            test_tweet_df = [test_tweet]
                            X_test_sample = tokenizer.texts_to_sequences(test_tweet_df)
                            X_test_sample = pad_sequences(X_test_sample, padding='post', maxlen=maxlen)
                            model = load_model('my_model.h5')
                            # cnn_clf.make_predict_function()

                            # model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                            # y_test = model.predict(X_test_sample)
                            # loaded_model=tf.saved_model.load("my_model")
                            # y_test=loaded_model.predict(X_test_sample)
                    
                            y_test = model.predict(X_test_sample)




                            # Read and load the JSON file
                            # json_file = open('model.json', 'r')
                            # loaded_model_json = json_file.read()
                            # json_file.close()

                            # # Use Keras's built in model_from_json function to convert the JSON file to a model
                            # loaded_model = model_from_json(loaded_model_json)

                            # model = load_model()
                            # y_test = loaded_model.predict(X_test_sample)

                            prediction = 'Not check-worthy' if y_test[0] <0.5 else 'Check-worthy'
                            col1,col2 = st.columns([2,2])
                            with col1:    
                                st.info("Prediction")
                                st.write(prediction)
                            with col2:
                                st.info("% Confidence")
                                st.write("≈ {:.0f}".format(    (y_test[0]*100)[0]   ) )
                                # if prediction == 'not worthy' : st.write(y_test[0,0]*100)
                                # else                          : st.write(y_test[0,0]*100)   



                              


    # if choice == "CNN - Type1 (optimization)":

    #     param_grid = dict(num_filters=[32, 64, 128],
    #               kernel_size=[3, 5, 7],
    #               vocab_size=[5000], 
    #               embedding_dim=[50],
    #               maxlen=[100])
    #     # Main settings
    #     epochs = 10
    #     embedding_dim = 50
    #     maxlen = 280
    #     output_file = 'output.txt'  
    #     X = df['tweet']
    #     y = df['check-worthy']
    #     # Train-test split
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_value, random_state = 1000)
        
    #     # Tokenize words
    #     tokenizer = Tokenizer(num_words=5000)
    #     tokenizer.fit_on_texts(X_train)
    #     X_train = tokenizer.texts_to_sequences(X_train)
    #     X_test = tokenizer.texts_to_sequences(X_test)

    #     # Adding 1 because of reserved 0 index
    #     vocab_size = len(tokenizer.word_index) + 1

    #     # Pad sequences with zeros
    #     X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    #     X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    #     # Parameter grid for grid search
    #     param_grid = dict(num_filters=[32, 64, 128],
    #                     kernel_size=[3, 5, 7],
    #                     vocab_size=[vocab_size],
    #                     embedding_dim=[embedding_dim],
    #                     maxlen=[maxlen])
    #     model = KerasClassifier(build_fn=create_model,
    #                             epochs=epochs, batch_size=10,
    #                             verbose=False)
    #     grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
    #                             cv=4, verbose=1, n_iter=5)
    #     grid_result = grid.fit(X_train, y_train)
    #     # Evaluate testing set
    #     st.subheader("Classifier Metrics - Convolutions Neural Network (CNN) (Type1) - Hyperparameters Optimization:")


    #     st.write("Training Accuracy: {:.4f}".format(grid.score(X_train, y_train)))


    #     test_accuracy = grid.score(X_test, y_test)
    #     st.write ("Testing Accuracy: {:.4f}".format(test_accuracy))
        
       

      













    if choice == "Convolutions Neural Network (CNN) - Type2":
            
            # Tokenize and transform to integer index
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(X_train)

            X_train = tokenizer.texts_to_sequences(X_train)
            X_test = tokenizer.texts_to_sequences(X_test)

            vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
            maxlen = max(len(x) for x in X_train) # longest text in train set

            # Add pading to ensure all vectors have same dimensionality
            X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
            X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
            # Define CNN architecture

            embedding_dim = 100

            cnn_clf = Sequential()
            cnn_clf.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

            cnn_clf.add( layers.Conv1D(filters=50,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu"))
            cnn_clf.add(layers.Conv1D(filters=50,
                                            kernel_size=3,
                                            padding="valid",
                                            activation="relu"))
            cnn_clf.add(layers.Conv1D(filters=50,
                                            kernel_size=4,
                                            padding="valid",
                                            activation="relu"))
            cnn_clf.add(layers.GlobalMaxPool1D())
            
            cnn_clf.add(layers.Dense(units=512, activation="relu"))
            cnn_clf.add(layers.Dropout(rate=0.1))
            cnn_clf.add(layers.Dense(units=1,  activation="sigmoid"))
            cnn_clf.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            # print(cnn_clf.summary())
            # Fit model
            history = cnn_clf.fit(X_train, y_train,
                                epochs=5,
                                verbose=True,
                                validation_data=(X_test, y_test),
                                batch_size=10)
            cnn_clf.save('cnn_clf2')
                    
            y_pred = cnn_clf.predict(X_test)


            loss, accuracy = cnn_clf.evaluate(X_train, y_train, verbose=True)
            # st.write("Training Accuracy: {:.4f}".format(accuracy))
            st.write("Training Accuracy: {:.4f}".format(accuracy))

            loss, accuracy = cnn_clf.evaluate(X_test, y_test, verbose=False)

            st.write("Testing Accuracy:  {:.4f}".format(accuracy))
          
            with st.form("my_form"):
                                    test_tweet = st.text_area("Enter Your Own Tweet:")        
                                    submitted = st.form_submit_button("Classify")     
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
                                        # # convert to lowercase
                                        test_tweet = test_tweet.lower() 


                                        test_tweet_df = [test_tweet]
                                        X_test_sample = tokenizer.texts_to_sequences(test_tweet_df)
                                        X_test_sample = pad_sequences(X_test_sample, padding='post', maxlen=maxlen)

                                        loaded_model = tf.keras.models.load_model('cnn_clf2')
                                        y_test=loaded_model.predict(X_test_sample)
                                        # y_test = cnn_clf.predict(X_test1)
                            

                                        prediction = 'not worthy' if y_test[0] <0.5 else 'worthy'
                                        col1,col2 = st.columns([2,2])
                                        with col1:    
                                            st.info("Prediction")
                                            st.write(prediction)
                                        with col2:
                                            st.info("% Confidence")
                                            if prediction == 'not worthy' : st.write(y_test[0,0]*100)
                                            else                          : st.write(y_test[0,0]*100)   


    






    if choice == "Decision Tree":
            vectorizer = CountVectorizer()
            vectorizer.fit(X_train)

            X_train = vectorizer.transform(X_train)
            X_test  = vectorizer.transform(X_test)
            dt_clf = DecisionTreeClassifier()
            dt_clf.fit(X_train, y_train)
            acc = dt_clf.score(X_test, y_test)
            y_pred = dt_clf.predict(X_test)
            st.write("Accuracy: {:.4f}".format(dt_clf.score(X_test, y_test)))
            st.write("Precision: {:.4f}".format(precision_score(y_test, y_pred, labels = class_names)))

            with st.form("my_form"):
                        test_tweet = st.text_area("Enter Your Own Tweet:")        
                        submitted = st.form_submit_button("Classify")     
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
                            # # convert to lowercase
                            test_tweet = test_tweet.lower() 


                            test_tweet_df = [test_tweet]
                            X_test_sample  = vectorizer.transform(test_tweet_df)
                            y_test = dt_clf.predict(X_test_sample)                 

                            prediction = 'not worthy' if y_test[0] == 0 else 'worthy'
                            col1,col2 = st.columns([2,2])
                            with col1:    
                                st.info("Prediction")
                                st.write(prediction)
                            with col2:
                                st.info("% Confidence")
                                if prediction == 'not worthy' : st.write(dt_clf.predict_proba(X_test_sample)[0,0]*100)
                                else                          : st.write(dt_clf.predict_proba(X_test_sample)[0,1]*100)   


    if choice == "Support Vector Machine (SVM)":
                vectorizer = CountVectorizer()
                vectorizer.fit(X_train)

                X_train = vectorizer.transform(X_train)
                X_test  = vectorizer.transform(X_test)
                svm_clf=SVC()
                svm_clf.fit(X_train, y_train)
                acc = svm_clf.score(X_test, y_test)
                # st.write('Accuracy: ', acc)
                y_pred = svm_clf.predict(X_test)
              

                # dt_clf = DecisionTreeClassifier()
                # dt_clf.fit(X_train, y_train)
                # acc = dt_clf.score(X_test, y_test)
                # y_pred = dt_clf.predict(X_test)
                st.write("Accuracy: {:.4f}".format(svm_clf.score(X_test, y_test)))
                st.write("Precision: {:.4f}".format(precision_score(y_test, y_pred, labels = class_names)))
                cm=confusion_matrix(y_test,y_pred)
                st.write('Confusion matrix: ', cm)
                with st.form("my_form"):
                            test_tweet = st.text_area("Enter Your Own Tweet:")        
                            submitted = st.form_submit_button("Classify")     
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
                                # # convert to lowercase
                                test_tweet = test_tweet.lower() 


                                test_tweet_df = [test_tweet]
                                X_test_sample  = vectorizer.transform(test_tweet_df)
                                y_test = dt_clf.predict(X_test_sample)                 

                                prediction = 'not worthy' if y_test[0] == 0 else 'worthy'
                                col1,col2 = st.columns([2,2])
                                with col1:    
                                    st.info("Prediction")
                                    st.write(prediction)
                                with col2:
                                    st.info("% Confidence")
                                    if prediction == 'not worthy' : st.write(dt_clf.predict_proba(X_test_sample)[0,0]*100)
                                    else                          : st.write(dt_clf.predict_proba(X_test_sample)[0,1]*100)   

  



   
    if st.sidebar.checkbox("Show Tweets Data-set", False):
        st.subheader("Tweets Data-set")
        # st.info("1 : Worthy    -----    0: Not Worthy")


        # st.subheader("Tweets Dataset - After Preprocessing -")
        # st.dataframe(df.style.highlight_max(axis=0),width=3000, height=400)
        st.table(df.style.highlight_max(axis=0))  

    




if __name__ == '__main__':
    main()