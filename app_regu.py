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
from tensorflow.keras.models import load_model
from joblib import dump, load
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2


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
    data['tweet'] = data['tweet'].replace(to_replace=r'https:\/\/.*',value='',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r'\<a href',value='',regex=True)

    # data['tweet'] = data['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

    # remove the word <link>
    data['tweet'] = data['tweet'].replace(to_replace=r'<link>',value='',regex=True)
    # remove emogis
    # data['tweet'] = data['tweet'].apply(lambda x: re.split('[^\w\s#@/:%.,_-]', str(x))[0])
    data['tweet'] = data['tweet'].replace(to_replace=r'[^\w\s#@/:%.,_-]',value='',regex=True)
    # # more cleaning (remove usernames-hashtags)
    # data['tweet'] = data['tweet'].replace(to_replace=r'(@){1}.+?( ){1}',value='',regex=True)
    # data['tweet'] = data['tweet'].replace(to_replace=r'(#){1}.+?( ){1}',value='',regex=True)
    # more replacing 
    data['tweet'] = data['tweet'].replace(to_replace=r'WH',value='world health organization',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r"shouldn't",value="should not",regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r'doesnt',value='does not',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r"don't",value="do not",regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r'dont',value='do not',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r'didnt',value='did not',regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r"didn't",value='did not',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r"isn't",value='is not',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r'isnt',value='is not',regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r"it's",value='it is',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r"couldn't",value='could not',regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r"aren't",value='are not',regex=True)                  
    data['tweet'] = data['tweet'].replace(to_replace=r"won't",value='will not',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r"wont",value="will not",regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r"hasn't",value='has not',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r"wasn't",value="was not",regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r'thats',value='that is',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r'lets',value='let us',regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r"hes",value='he is',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r"theyre",value='they are',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r'whats',value='what is',regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r"can't",value='can not',regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=r"cant",value='can not',regex=True)  
    data['tweet'] = data['tweet'].replace(to_replace=r"im ",value='i am',regex=True)                  

    # remove punctaution
    data['tweet'] = data['tweet'].str.replace('[^\w\s]','')
    # convert to lowercase
    data['tweet'] = data['tweet'].str.lower() 
    return data



@st.cache(persist = True)
def single_tweet_preprocess(test_tweet):
    # remove hyberlinks
    test_tweet = re.sub(r'https:\/\/.*', '', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'\<a href', '', test_tweet, flags=re.MULTILINE)    
    # remove the word <link>
    test_tweet = re.sub(r'<link>', '', test_tweet, flags=re.MULTILINE)
    # remove emogis
    test_tweet = re.sub(r'[^\w\s#@/:%.,_-]', '', test_tweet, flags=re.MULTILINE)
    # more cleaning (usernames-hashtags)
    # test_tweet = re.sub(r'(@){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
    # test_tweet = re.sub(r'(#){1}.+?( ){1}', ' ', test_tweet, flags=re.MULTILINE)
    # more replacing 
    test_tweet = re.sub(r'WH','world health organization', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"shouldn't",'should not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'doesnt','does not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"don't",'do not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'dont','do not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'didnt','did not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"didn't",'did not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"isn't",'is not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'isnt','is not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"it's",'it is', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"couldn't",'could not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"aren't",'are not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"won't",'will not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'wont','will not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"hasn't",'has not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"wasn't",'was not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'thats','that is', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'lets','let us', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'hes','he is', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'theyre','they are', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'whats','what is', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r"can't",'can not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'cant','can not', test_tweet, flags=re.MULTILINE)
    test_tweet = re.sub(r'im ','i am', test_tweet, flags=re.MULTILINE)
    # remove punctaution
    test_tweet =re.sub(r'[^\w\s]','', test_tweet, flags=re.MULTILINE)
    # convert to lowercase
    test_tweet = test_tweet.lower() 
    return test_tweet







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

# def load_data1():
#         #Load the dataset
#     data = pd.read_csv('data1.csv')
    
#     # remove hyberlinks
#     data['tweet'] = data['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
#     # remove the word <link>
#     data['tweet'] = data['tweet'].replace(to_replace=r'<link>',value='',regex=True)
#     # remove emogis
#     data['tweet'] = data['tweet'].apply(lambda x: re.split('[^\w\s#@/:%.,_-]', str(x))[0])
#     # more cleaning (remove usernames-hashtags)
#     data['tweet'] = data['tweet'].replace(to_replace=r'(@){1}.+?( ){1}',value='',regex=True)
#     data['tweet'] = data['tweet'].replace(to_replace=r'(#){1}.+?( ){1}',value='',regex=True)
#     # convert to lowercase
#     data['tweet'] = data['tweet'].str.lower() 
#     return data

@st.cache(persist = True)
def split(df, test_size_value):
    X = df['tweet']
    y = df['check-worthy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_value, random_state = 0)
    return X_train, X_test, y_train, y_test













def main():
    st.title("Detecting check worthy Tweets")
    st.markdown("This app is created to detect if a tweet is check-worthy or not in the domain of Corona Virus")
    st.markdown("Please choose your classifier model first then classify your tweet!")
    classifier = [ "Choose Classifier","Logistic Regression", "Word Embeddings", "Pretrained Word Embeddings", "CNN - Type 1",  "CNN - Type 2"]
    choice = st.sidebar.selectbox("Choose Classifier",classifier)
    # df =load_data()
    df=load_data()
    # df = pd.concat([df, df1])
    # test_size_value = st.sidebar.radio("Test size: (Default is 30%)", (0.1, 0.2, 0.3), 0, key = 'test_size')
    class_names = ['worthy', 'not worthy']


    if choice == "Logistic Regression":
        X_train, X_test, y_train, y_test = split(df,0.20)
        vectorizer = CountVectorizer()
        vectorizer.fit(X_train)
        X_train = vectorizer.transform(X_train)
        X_test  = vectorizer.transform(X_test)

        # lr_clf = LogisticRegression()            
        # lr_clf.fit(X_train, y_train)
        # dump(lr_clf, 'lr_clf.joblib') 

        lr_clf_model = load('lr_clf.joblib')
        st.subheader("Classifier Metrics - Logistic Regression:")
        st.write("Training Accuracy: {:.2f}".format(lr_clf_model.score(X_train, y_train)))
        st.write("Testing Accuracy: {:.2f}".format(lr_clf_model.score(X_test, y_test)))
        with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Own Tweet:")        
            submitted = st.form_submit_button("Classify")     
            if submitted:
                if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                else:
                    if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        test_tweet=single_tweet_preprocess(test_tweet)
                        test_tweet_df = [test_tweet]
                        X_test_sample  = vectorizer.transform(test_tweet_df)

                        y_pred = lr_clf_model.predict(X_test_sample)              
                        prediction = 'Not check-worthy' if y_pred[0] == 0 else 'Check-worthy'
                        col1,col2 = st.columns([2,2])
                        with col1:    
                            st.info("Prediction")
                            st.write(prediction)
                        with col2:
                            st.info("% Confidence")
                            if prediction == 'Not check-worthy' : 
                                st.write("??? {:.0f}".format(lr_clf_model.predict_proba(X_test_sample)[0,0]*100))
                            else : 
                                st.write("??? {:.0f}".format(lr_clf_model.predict_proba(X_test_sample)[0,1]*100)) 


    if choice == "Word Embeddings":
        X_train, X_test, y_train, y_test = split(df,0.20)
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        maxlen = 280
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen) 
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        # embedding_dim = 50
        # word_embeddings_clf = Sequential()
        # word_embeddings_clf.add(layers.Embedding(input_dim=vocab_size, 
        #                         output_dim=embedding_dim, 
        #                         input_length=maxlen))
        # word_embeddings_clf.add(layers.GlobalMaxPool1D())
        # # word_embeddings_clf.add(layers.Flatten())
        # # word_embeddings_clf.add(layers.Dense(10, activation='relu', kernel_regularizer=l2(0.6)))
        # word_embeddings_clf.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.07)))
        # word_embeddings_clf.compile(optimizer='adam',
        #             loss='binary_crossentropy',
        #             metrics=['accuracy'])
        # word_embeddings_clf.fit(X_train, y_train,
        #             epochs=15,
        #             verbose=False,
        #             validation_data=(X_test, y_test),
        #             batch_size=10)
        # word_embeddings_clf.save('word_embeddings_clf.h5')

        word_embeddings_clf_model = load_model('word_embeddings_clf.h5')
        st.subheader("Classifier Metrics - Sequential Model with Word Embeddings:")
        loss, accuracy = word_embeddings_clf_model.evaluate(X_train, y_train, verbose=False)
        st.write("Training Accuracy: {:.2f}".format(accuracy))
        loss, accuracy = word_embeddings_clf_model.evaluate(X_test, y_test, verbose=False)
        st.write("Testing Accuracy:  {:.2f}".format(accuracy))
        with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Own Tweet:")        
            submitted = st.form_submit_button("Classify")     
            if submitted:
                if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                else:
                    if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        test_tweet=single_tweet_preprocess(test_tweet)
                        test_tweet_df = [test_tweet]
                        X_test_sample = tokenizer.texts_to_sequences(test_tweet_df)
                        X_test_sample = pad_sequences(X_test_sample, padding='post', maxlen=maxlen)
                        y_pred = word_embeddings_clf_model.predict(X_test_sample)                 

                        prediction = 'Not check-worthy' if y_pred[0]*100 < 50 else 'Check-worthy'
                        col1,col2 = st.columns([2,2])
                        with col1:    
                            st.info("Prediction")
                            st.write(prediction)
                        with col2:
                            st.info("% Confidence")
                            st.write("??? {:.0f}".format(    (100-y_pred[0]*100)[0]   ) )


    if choice == "Pretrained Word Embeddings":
        X_train, X_test, y_train, y_test = split(df,0.20)
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        maxlen = 280

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        # embedding_dim = 50
        # embedding_matrix = create_embedding_matrix(
        #      'glove.6B.50d.txt',
        #      tokenizer.word_index, embedding_dim)

        # pretrained_embeddings_clf = Sequential()
        # pretrained_embeddings_clf.add(layers.Embedding(vocab_size, embedding_dim, 
        #                    weights=[embedding_matrix], 
        #                    input_length=maxlen, 
        #                    trainable=True))
        # pretrained_embeddings_clf.add(layers.GlobalMaxPool1D())
        # pretrained_embeddings_clf.add(layers.Dense(10, activation='relu'))
        # pretrained_embeddings_clf.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.7)))
        # pretrained_embeddings_clf.compile(optimizer='adam',
        #             loss='binary_crossentropy',
        #             metrics=['accuracy'])
        # pretrained_embeddings_clf.fit(X_train, y_train,
        #                     epochs=20   ,
        #                     verbose=False,
        #                     validation_data=(X_test, y_test),
        #                     batch_size=10)    
        # pretrained_embeddings_clf.save('pretrained_embeddings_clf.h5')

        pretrained_embeddings_clf_model = load_model('pretrained_embeddings_clf.h5')
        st.subheader("Classifier Metrics - Sequential model with Pretrained Word Embeddings:")
        loss, accuracy = pretrained_embeddings_clf_model.evaluate(X_train, y_train, verbose=False)
        st.write("Training Accuracy: {:.2f}".format(accuracy))
        loss, accuracy = pretrained_embeddings_clf_model.evaluate(X_test, y_test, verbose=False)
        st.write("Testing Accuracy:  {:.2f}".format(accuracy))      

        with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Own Tweet:")        
            submitted = st.form_submit_button("Classify")     
            if submitted:
                if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                else:
                    if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        test_tweet=single_tweet_preprocess(test_tweet)
                        test_tweet_df = [test_tweet]
                        X_test_sample = tokenizer.texts_to_sequences(test_tweet_df)
                        X_test_sample = pad_sequences(X_test_sample, padding='post', maxlen=maxlen)
                        y_pred = pretrained_embeddings_clf_model.predict(X_test_sample)                 
                                  
                        prediction = 'Not check-worthy' if y_pred[0]*100 < 50 else 'Check-worthy'
                        col1,col2 = st.columns([2,2])
                        with col1:    
                            st.info("Prediction")
                            st.write(prediction)
                        with col2:
                            st.info("% Confidence")
                            st.write("??? {:.0f}".format(    (100-y_pred[0]*100)[0]   ) )
    

    # if choice == "Decision Tree":
    #         X_train, X_test, y_train, y_test = split(df,0.20)
    #         vectorizer = CountVectorizer()
    #         vectorizer.fit(X_train)

    #         X_train = vectorizer.transform(X_train)
    #         X_test  = vectorizer.transform(X_test)

    #         # dt_clf = DecisionTreeClassifier()
    #         # dt_clf.fit(X_train, y_train)
    #         # dump(dt_clf, 'dt_clf.joblib') 

    #         dt_clf_model = load('dt_clf.joblib')
    #         st.subheader("Classifier Metrics - Decision Tree:")
    #         st.write("Training Accuracy: {:.2f}".format(dt_clf_model.score(X_train, y_train)))
    #         st.write("Testing Accuracy: {:.2f}".format(dt_clf_model.score(X_test, y_test)))

    #         with st.form("my_form"):
    #             test_tweet = st.text_area("Enter Your Own Tweet:")        
    #             submitted = st.form_submit_button("Classify")     
    #             if submitted:
    #                 if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
    #                 else:
    #                     if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
    #                     else:                         
    #                         test_tweet=single_tweet_preprocess(test_tweet)
    #                         test_tweet_df = [test_tweet]
    #                         X_test_sample  = vectorizer.transform(test_tweet_df)
    #                         y_pred = dt_clf_model.predict(X_test_sample)                 
    #                         prediction = 'Not check-worthy' if y_pred[0] == 0 else 'Check-worthy'
    #                         col1,col2 = st.columns([2,2])
    #                         with col1:    
    #                             st.info("Prediction")
    #                             st.write(prediction)
    #                         with col2:
    #                             st.info("% Confidence")
    #                             if prediction == 'Not check-worthy' : 
    #                                 st.write("??? {:.0f}".format(100))
    #                             else : 
    #                                 st.write("??? {:.0f}".format(100)) 


   
    # if choice == "SVC":
    #     X_train, X_test, y_train, y_test = split(df,0.20)
    #     vectorizer = CountVectorizer()
    #     vectorizer.fit(X_train)

    #     X_train = vectorizer.transform(X_train)
    #     X_test  = vectorizer.transform(X_test)

    #     # svc_clf=SVC()
    #     # svc_clf.fit(X_train, y_train)
    #     # dump(svc_clf, 'svc_clf.joblib') 

    #     svc_clf_model = load('svc_clf.joblib')
    #     st.subheader("Classifier Metrics - Support Vector Classifier (SVC):")
    #     st.write("Training Accuracy: {:.2f}".format(svc_clf_model.score(X_train, y_train)))
    #     st.write("Testing Accuracy: {:.2f}".format(svc_clf_model.score(X_test, y_test)))
    #     with st.form("my_form"):
    #         test_tweet = st.text_area("Enter Your Own Tweet:")        
    #         submitted = st.form_submit_button("Classify")     
    #         if submitted:
    #             if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
    #             else:
    #                 if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
    #                 else:
    #                     test_tweet=single_tweet_preprocess(test_tweet)
    #                     test_tweet_df = [test_tweet]
    #                     X_test_sample  = vectorizer.transform(test_tweet_df)

    #                     y_pred = svc_clf_model.predict(X_test_sample)      
    #                     prediction = 'Not check-worthy' if y_pred[0] == 0 else 'Check-worthy'
    #                     col1,col2 = st.columns([2,2])
    #                     with col1:    
    #                         st.info("Prediction")
    #                         st.write(prediction)
    #                     with col2:
    #                         st.info("% Confidence")
    #                         if prediction == 'Not check-worthy' : 
    #                             st.write("??? {:.0f}".format(100))
    #                         else : 
    #                             st.write("??? {:.0f}".format(100)) 


    


    if choice == "CNN - Type 1":
            X_train, X_test, y_train, y_test = split(df,0.20)
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

            # embedding_dim = 100
            # cnn_clf2 = Sequential()
            # cnn_clf2.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
            # cnn_clf2.add( layers.Conv1D(filters=50,
            #                             kernel_size=2,
            #                             padding="valid",
            #                             activation="relu"))
            # cnn_clf2.add(layers.Conv1D(filters=50,
            #                                 kernel_size=3,
            #                                 padding="valid",
            #                                 activation="relu"))
            # cnn_clf2.add(layers.Conv1D(filters=50,
            #                                 kernel_size=4,
            #                                 padding="valid",
            #                                 activation="relu"))
            # cnn_clf2.add(layers.GlobalMaxPool1D())
            # cnn_clf2.add(layers.Dense(units=512, activation="relu"))
            # cnn_clf2.add(layers.Dropout(rate=0.1))
            # cnn_clf2.add(layers.Dense(units=1,  activation="sigmoid"))
            # cnn_clf2.compile(optimizer='adam',
            #             loss='binary_crossentropy',
            #             metrics=['accuracy'])
            # cnn_clf2.fit(X_train, y_train,
            #                     epochs=5,
            #                     verbose=True,
            #                     validation_data=(X_test, y_test),
            #                     batch_size=10)
                    
            # cnn_clf2.save('cnn_clf2.h5')

            cnn_clf2_model = load_model('cnn_clf2.h5')
            st.subheader("Classifier Metrics - Convolutions Neural Network (CNN) - Type 1:")
            loss, accuracy = cnn_clf2_model.evaluate(X_train, y_train, verbose=True)
            st.write("Training Accuracy: {:.2f}".format(accuracy))
            loss, accuracy = cnn_clf2_model.evaluate(X_test, y_test, verbose=False)
            st.write("Testing Accuracy:  {:.2f}".format(accuracy))

            with st.form("my_form"):
                test_tweet = st.text_area("Enter Your Own Tweet:")        
                submitted = st.form_submit_button("Classify")     
                if submitted:
                    if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                        else:
                            test_tweet=single_tweet_preprocess(test_tweet)
                            test_tweet_df = [test_tweet]
                            X_test_sample = tokenizer.texts_to_sequences(test_tweet_df)
                            X_test_sample = pad_sequences(X_test_sample, padding='post', maxlen=maxlen)

                            y_pred = cnn_clf2_model.predict(X_test_sample)
                            prediction = 'Not check-worthy' if y_pred[0] <0.5 else 'Check-worthy'
                            col1,col2 = st.columns([2,2])
                            with col1:    
                                st.info("Prediction")
                                st.write(prediction)
                            with col2:
                                st.info("% Confidence")
                                if prediction == 'Not check-worthy' : st.write('??? {:.0f}'.format(y_pred[0][0]*2*100))
                                else                                : st.write('??? {:.0f}'.format( (100 - (   y_pred[0][0]*100   )    )/2 )) 

                               
        
          
         
    if choice == "CNN - Type 2":
        X_train, X_test, y_train, y_test = split(df,0.20)
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
        # # Define CNN architecture

        # embedding_dim = 100
        # cnn_clf = Sequential()
        # cnn_clf.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        # cnn_clf.add(layers.Conv1D(128, 5, activation='relu'))
        # cnn_clf.add(layers.GlobalMaxPooling1D())
        # cnn_clf.add(layers.Dense(10, activation='relu'))
        # cnn_clf.add(layers.Dense(1, activation='sigmoid'))
        # cnn_clf.compile(optimizer='adam',
        #             loss='binary_crossentropy',
        #             metrics=['accuracy'])
        # cnn_clf.fit(X_train, y_train,
        #                     epochs=5,
        #                     verbose=True,
        #                     validation_data=(X_test, y_test),
        #                     batch_size=10)
        # cnn_clf.save('cnn_clf.h5')

        cnn_clf_model = load_model('cnn_clf.h5')
        st.subheader("Classifier Metrics - Convolutions Neural Network (CNN)- Type 2:")
        loss, accuracy = cnn_clf_model.evaluate(X_train, y_train, verbose=True)
        st.write("Training Accuracy: {:.2f}".format(accuracy))
        loss, accuracy = cnn_clf_model.evaluate(X_test, y_test, verbose=False)
        st.write("Testing Accuracy:  {:.2f}".format(accuracy))

        with st.form("my_form"):
            test_tweet = st.text_area("Enter Your Own Tweet:")        
            submitted = st.form_submit_button("Classify")     
            if submitted:
                if test_tweet =="" : st.error("Tweet should not be empty or less than 5 words!")
                else:
                    if len(test_tweet.split()) < 5 : st.error("Tweet should not be empty or less than 5 words!")
                    else:
                        test_tweet=single_tweet_preprocess(test_tweet)
                        test_tweet_df = [test_tweet]
                        X_test_sample = tokenizer.texts_to_sequences(test_tweet_df)
                        X_test_sample = pad_sequences(X_test_sample, padding='post', maxlen=maxlen)

                        y_pred = cnn_clf_model.predict(X_test_sample)
                        prediction = 'Not check-worthy' if y_pred[0] <0.5 else 'Check-worthy'
                        col1,col2 = st.columns([2,2])
                        with col1:    
                            st.info("Prediction")
                            st.write(prediction)
                        with col2:
                            st.info("% Confidence")
                            if prediction == 'Not check-worthy' : st.write('??? {:.0f}'.format(y_pred[0][0]*2*100))
                            else                                : st.write('??? {:.0f}'.format( (100 - (   y_pred[0][0]*100   )    )/2 )) 

                               
   

                              

              

               

             

              



    


    

    # if choice == "CNN - Hyperparameters optimization":

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
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1000)
        
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
        
       



   
    if st.sidebar.checkbox("Show Tweets Data-set", False):
        st.subheader("Tweets Data-set:")
        # st.info("1 : Worthy    -----    0: Not Worthy")


        # st.subheader("Tweets Dataset - After Preprocessing -")
        # st.dataframe(df.style.highlight_max(axis=0),width=3000, height=400)
        st.table(df.style.highlight_max(axis=0))  

    if st.sidebar.button("About Us!"):
        st.sidebar.info("This App is done for a master's thesis at the university of Duisburg-Essen under the supervisement of Prof. Torsten Zesch and Dr. Ahmet Aker. The motivation of this thesis is detecting check-worthy tweets in the domain of corona virus. Hassan??Yousef")




if __name__ == '__main__':
    main()