#!/usr/bin/env python3
# importing relevant libraries
import os
import pickle
from PIL import Image
import pandas as pd
from collections import Counter
import math
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import textstat

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

# downloading relevant nltk packages
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('stopwords')


stop_words = stopwords.words('english')


# initializes the ImageOCRModel class with the base directory
class ImageOCRModel:
    def __init__(self, base_dir):
        # base_dir: The base directory where the unstructured data (OCR text files and images) is located
        self.base_dir = base_dir
        self.model = None  # placeholder for the trained machine learning model
        self.scaler = None  # placeholder for the scaler used to normalize the feature data
        self.label_encoder = None  # placeholder for the label encoder used to encode the labels

    # turns unstructured text and image data into a pandas dataframe
    def load_data(self):
        # initializes an empty list to store the data
        raw_image_data = []
        # iterates through the subdirectories in the 'ocr' folder
        for subdir_name in os.listdir(os.path.join(base_dir, 'ocr')):
            # defines the path to the subdirectory for the ocr data
            subdir_path_ocr = os.path.join(base_dir, 'ocr', subdir_name)
            # defines the path to the subdirectory for the images data
            subdir_path_images = os.path.join(base_dir, 'images', subdir_name)
            # checks if the ocr subdirectory exists
            if os.path.isdir(subdir_path_ocr):
                # Iterates through the files in the current ocr subdirectory
                for file_name in os.listdir(subdir_path_ocr):
                    # checks if the file name is a.TIF.txt file, like all the ocr data
                    if file_name.endswith('.TIF.txt'):
                        # appends the ocr file name to the ocr file path                
                        ocr_file_path = os.path.join(subdir_path_ocr, file_name)
                        # drops .txt from the file name to get the image file name
                        image_file_name = file_name.replace('.TIF.txt', '.TIF')
                        # appends the image file name to the image file path
                        image_file_path = os.path.join(subdir_path_images, image_file_name)
                        # opens the files and reads its contents, utf-8 is default encoding system for python
                        with open(ocr_file_path, 'r', encoding='utf-8') as f:
                            # reads the text from the ocr file
                            text = f.read()
                            # opens the image file
                            image = Image.open(image_file_path)
                            # calculates the width and height of the image in pixels and saves these values as variables
                            image_width, image_height = image.width, image.height
                            # calculates the total pixels in the image
                            total_pixels = image_width * image_height
                            # calculates the number of black pixels in the image
                            black_pixels = sum(1 for pixel in image.getdata() if pixel == 0)
                            # calculates the percentage of black pixels in the image
                            percentage_black_pixels = (black_pixels / total_pixels) * 100
                            # appends the features extracted to a new row in the raw_image_data list
                            raw_image_data.append({
                                'label': subdir_name,
                                'image_file_name': image_file_name,
                                'text': text,
                                'image_width': image_width,
                                # 'image_height': image_height,
                                'total_pixels': total_pixels,
                                'black_pixels': black_pixels,
                                'percentage_black_pixels': percentage_black_pixels
                            })
        return pd.DataFrame(raw_image_data)

    # preprocesses the data (e.g., extract features, clean text)
    def preprocess_data(self, df):
        # raw_text- subtract all of the \n from text
        # creates a new column called 'raw_text' that removes newlines, 
        # any whitespaces associated with new lines, and any leading or trailing whitespaces
        df['raw_text'] = df['text'].str.replace('\n', ' ').str.replace('\s+', ' ', regex=True).str.strip()

        # num lines- extracted from text
        # This line counts the number of newline characters (\n) in the text column 
        # and adds 1 to account for the last line, which doesn't have a newline character.
        df['num_lines'] = df['text'].str.count('\n') + 1

        # num words- extracted from raw_text
        # This line first splits the raw_text column by whitespace using str.split(), 
        # which creates a list of words for each row. 
        # Then, it counts the length of this list using str.len(), 
        # giving us the number of words in each row.
        df['num_words'] = df['raw_text'].str.split().str.len()

        # num characters- extracted from raw_text
        # This line uses the str.len() method to count the number of characters in the raw_text column.
        df['num_chars'] = df['raw_text'].str.len()


        # defines refined regular expression patterns to match different date formats
        patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',             # Matches dates like 2-8-69, 2/8/1969
            r'\b\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+,? \d{2,4}\b',  # Matches dates like 2nd August 1969, 2 August 1969
            r'\b[A-Za-z]+ \d{1,2}, \d{2,4}\b',                 # Matches dates like August 2, 1969
            r'\b[A-Za-z]+ \d{1,2}\b',                          # Matches dates like August 2
            r'\b[A-Za-z]+ \d{4}\b'                             # Matches dates like August 1969
        ]

        # combines all patterns into a single pattern
        combined_pattern = '|'.join(patterns)

        # defines a function to count the number of dates in a text
        def count_dates(text):
            # finds *all* the matches and returns them as a list of strings, with each string representing one match
            matches = re.findall(combined_pattern, text, flags=re.IGNORECASE)
            return len(matches)


        def unique_word_count(text):
            words = word_tokenize(text)
            unique_words = set(words)
            return len(unique_words)

        def average_word_length(text):
            words = word_tokenize(text)
            len_words = len(words)
            if len_words == 0:
                return 0  # returns 0 if there are no words in the text to avoid div/0 error
            return sum(len(word) for word in words) / len_words

        def stopword_ratio(text):
            words = word_tokenize(text)
            len_words = len(words)
            if len_words == 0:
                return 0  # returns 0 if there are no words in the text to avoid div/0 error
            stopword_count = sum(1 for word in words if word in stop_words)
            return stopword_count / len_words


        # applies the function to the dataframe to create the 'num_dates_present' column
        df['num_dates_present'] = df['raw_text'].apply(count_dates)
        # applies all feature functions
        df['unique_word_count'] = df['raw_text'].apply(unique_word_count)
        df['average_word_length'] = df['raw_text'].apply(average_word_length)
        df['stopword_ratio'] = df['raw_text'].apply(stopword_ratio)


        # function to calculate sentiment using TextBlob library
        def calculate_sentiment(text):
            blob = TextBlob(text)
            sentiment = blob.sentiment
            return pd.Series([sentiment.polarity, sentiment.subjectivity], index=['sentiment_polarity', 'sentiment_subjectivity'])

        # function to calculate readability scores using textstat library
        def calculate_readability(text):
            # different methods of readability scores
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
            return pd.Series([flesch_reading_ease, flesch_kincaid_grade], index=['flesch_reading_ease', 'flesch_kincaid_grade'])

        # applies the functions to the text column
        df[['sentiment_polarity', 'sentiment_subjectivity']] = df['raw_text'].apply(calculate_sentiment)
        df[['flesch_reading_ease', 'flesch_kincaid_grade']] = df['raw_text'].apply(calculate_readability)

        # function to calculate parts of speech frequency using NLTK
        def pos_frequency_nltk(text):
            words = word_tokenize(text)
            # removing punctuation and stopwords
            words = [word for word in words if word.isalpha() and word not in stop_words]
            pos_tags = pos_tag(words, tagset='universal')
            pos_counts = Counter(tag for word, tag in pos_tags)
            num_words = len(words)
            pos_freq = {tag: count / num_words for tag, count in pos_counts.items()}
            return pd.Series(pos_freq)

        # creates a new DataFrame with the POS frequency columns
        pos_df = df['raw_text'].apply(pos_frequency_nltk).fillna(0)

        # concatenates POS frequency columns with the original DataFrame
        df = pd.concat([df, pos_df], axis=1)

        return df

    # splits the data into training and test sits and then performs tfidf vector calculations
    def split_and_scale_data(self, df):
        y = df['label']
        # using a test_size = 0.2 which means 20% of the data will be used for testing and 80% for training (4:1 ratio)
        # stratify=y means that the data will be split in a stratified manner (equal representation in each fold) based on the labels in the y column
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)


        # # function to calculate sentiment using TextBlob library
        # def calculate_sentiment(text):
        #     blob = TextBlob(text)
        #     sentiment = blob.sentiment
        #     return pd.Series([sentiment.polarity, sentiment.subjectivity], index=['sentiment_polarity', 'sentiment_subjectivity'])

        # # function to calculate readability scores using textstat library
        # def calculate_readability(text):
        #     # different methods of readability scores
        #     flesch_reading_ease = textstat.flesch_reading_ease(text)
        #     flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        #     return pd.Series([flesch_reading_ease, flesch_kincaid_grade], index=['flesch_reading_ease', 'flesch_kincaid_grade'])

        # # function to calculate parts of speech frequency using NLTK
        # def pos_frequency_nltk(text):
        #     words = word_tokenize(text)
        #     # removing punctuation and stopwords
        #     words = [word for word in words if word.isalpha() and word not in stop_words]
        #     pos_tags = pos_tag(words, tagset='universal')
        #     pos_counts = Counter(tag for word, tag in pos_tags)
        #     num_words = len(words)
        #     pos_freq = {tag: count / num_words for tag, count in pos_counts.items()}
        #     return pd.Series(pos_freq)

        # # applies the functions to the text column
        # X_train[['sentiment_polarity', 'sentiment_subjectivity']] = X_train['raw_text'].apply(calculate_sentiment)
        # X_train[['flesch_reading_ease', 'flesch_kincaid_grade']] = X_train['raw_text'].apply(calculate_readability)
        # X_test[['sentiment_polarity', 'sentiment_subjectivity']] = X_test['raw_text'].apply(calculate_sentiment)
        # X_test[['flesch_reading_ease', 'flesch_kincaid_grade']] = X_test['raw_text'].apply(calculate_readability)

        # # creates a new DataFrame with the POS frequency columns
        # train_pos_df = X_train['raw_text'].apply(pos_frequency_nltk).fillna(0)
        # test_pos_df = X_test['raw_text'].apply(pos_frequency_nltk).fillna(0)

        # # concatenates POS frequency columns with the original DataFrame
        # X_train = pd.concat([X_train, train_pos_df], axis=1)
        # X_test = pd.concat([X_test, test_pos_df], axis=1)



        # indicates how many to tfidf vectors to keep
        how_many_tfidf_vectors = 10000

        # applying after train-test split
        training_doc_corpus = X_train['raw_text']
        test_doc_corpus = X_test['raw_text']

        vectorizer = TfidfVectorizer(stop_words=stop_words)
        # using fit_transform for training data but just transform for test data to prevent data leakage
        training_matrix = vectorizer.fit_transform(training_doc_corpus)
        test_matrix = vectorizer.transform(test_doc_corpus)

        # convert sparse matrix to dataframe
        training_sparse_matrix = pd.DataFrame(training_matrix.toarray())
        test_sparse_matrix = pd.DataFrame(test_matrix.toarray())

        # take only the most common words so we don't overfit
        tfidf_feature_columns = list(training_sparse_matrix.mean().sort_values()[-how_many_tfidf_vectors:].index)

        # redefine the dataframe's to only include these columns
        training_sparse_matrix_features = training_sparse_matrix[tfidf_feature_columns]
        test_sparse_matrix_features = test_sparse_matrix[tfidf_feature_columns]

        # convert column names to strings
        training_sparse_matrix_features.columns = training_sparse_matrix_features.columns.astype(str)
        test_sparse_matrix_features.columns = test_sparse_matrix_features.columns.astype(str)

        # redefines the training and test data to include the tfidf vectors
        X_train = pd.concat([X_train.reset_index(drop=True), training_sparse_matrix_features], axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True), test_sparse_matrix_features], axis=1)

        # drops text data columns that aren't needed anymore
        X_train_with_label = X_train.drop(['image_file_name', 'text', 'raw_text'], axis=1)
        X_test_with_label = X_test.drop(['image_file_name', 'text', 'raw_text'], axis=1)

        # creates feature table from the training data by dropping the label / target column
        X_train = X_train_with_label.drop(['label'], axis=1)
        # creates feature table from the test data by dropping the label / target column
        X_test = X_test_with_label.drop(['label'], axis=1)

        # converts labels to numerical format (dummy variables)
        self.label_encoder = LabelEncoder()
        # one-hot encodes the target column in the training data using fit_transform
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        # one-hot encodes the target column in the training data just using transform to prevent data leakage
        y_test_encoded = self.label_encoder.transform(y_test)

        # initializes the Standard Scaler which uses the Standard Normal Gaussian distribution
        self.scaler = StandardScaler()
        # scales the training data using fit_transform
        X_train_scaled = self.scaler.fit_transform(X_train)
        # scales the test data just using transform to prevent data leakage
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, X_test_with_label

    # trains the model using best performing algorithm
    def train_model(self, X_train, y_train):
        # max iterations set to 1000 to prevent warnings from sklearn
        # lbfgs is the default solver for multinomial multiclass classification
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        self.model.fit(X_train, y_train)

    # generates the multi-class confusion matrix and classification report
    def evaluate_model(self, X_test, y_test, X_test_with_label):
        # generates prediction on the test data
        y_pred = self.model.predict(X_test)

        # evaluates the model on the scaled validation set by generating probabilities for each class
        y_pred_proba = self.model.predict_proba(X_test)
        # turns y_pred_proba into a pandas dataframe with each column labeled with proba_ prepended to the class name for easy viewing
        y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=[f'proba_{cls}' for cls in self.label_encoder.classes_])
        # adds the predicted labels as a column, which will also represent the label with the maximum predicted probability        
        y_pred_proba_df['y_pred'] = self.label_encoder.inverse_transform(y_pred)
        # creates a new dataframe called test_with_proba that combines everything in X_test_with_label with y_pred_proba_df
        test_with_proba = pd.concat([X_test_with_label.reset_index(drop=True), y_pred_proba_df], axis=1)        

        # adds a boolean column to indicate whether the prediction was correct or not
        test_with_proba['correct'] = test_with_proba['label'] == test_with_proba['y_pred']

        # generates the true and predicted values on the test data for the confusion matrix and classification report
        y_true = np.array(test_with_proba['label'])
        y_pred = np.array(test_with_proba['y_pred'])

        # generates the multi-class confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        # generates the  classification report
        class_report = classification_report(y_true, y_pred, digits=4)

        return conf_matrix, class_report

    # saves the model, scaler, and label encoder to a file
    def save_model(self, file_path):
        # 'wb' stands for "white binary", which means the file is opened for writing in binary mode, 
        # which is necessary when dealing with non-text data like a pickled Python object.
        # 'f' is the file object, or the opened file, which in this case is used to write to the file
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,  # Save the trained model
                'scaler': self.scaler,  # Save the scaler used for normalizing the data
                'label_encoder': self.label_encoder  # Save the label encoder used for encoding the labels
              }, f)

    # loads a trained model, scaler, and label encoder from a file
    def load_model(self, file_path):
        # 'rb' stands for "read binary", which means the file is opened for reading in binary mode
        # 'f' is the file object, or the opened file, which in this case is used to read from the file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']  # Load the trained model
            self.scaler = data['scaler']  # Load the scaler used for normalizing the data
            self.label_encoder = data['label_encoder']  # Load the label encoder used for encoding the labels

if __name__ == "__main__":
    # gets the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # defines the base directory relative to the script's directory
    base_dir = os.path.join(script_dir, '..', 'data')
    # creates an instance of the ImageOCRModel class with the base directory
    model = ImageOCRModel(base_dir)
    # loads the data from the base directory
    df = model.load_data()
    # preprocesses the data (e.g., extract features, clean text)
    df = model.preprocess_data(df)
    # splits the data into training and test sets and scales the feature data
    X_train, X_test, y_train, y_test, X_test_with_label = model.split_and_scale_data(df)
    # trains the model using scaled training data and best performing algorithm
    model.train_model(X_train, y_train)
    # generates the multi-class confusion matrix and classification report
    conf_matrix, class_report = model.evaluate_model(X_test, y_test, X_test_with_label)

    # prints the Confusion Matrix
    print("Confusion Matrix:\n", conf_matrix)
    # prints the Classification Report containing Accuracy, Precision, Recall, and F-1 score
    print("Classification Report:\n", class_report)

    # saves the trained model as a pickle file
    model.save_model(os.path.join(script_dir, 'trained_model.pkl'))

    # saves the classification report to a .txt file in the ML/src folder
    with open(os.path.join(script_dir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

    # saves the confusion matrix to a .txt file in the ML/src folder after converting it to string format
    with open(os.path.join(script_dir, 'confusion_matrix.txt'), 'w') as f:
        f.write(str(conf_matrix))