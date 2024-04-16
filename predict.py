import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import pandas as pd

dataset = pd.read_csv('a2_RestaurantReviews_FreshDump.tsv',
                    delimiter='\t', quoting=3)
dataset.head()


nltk.download('stopwords')

ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus = []

for i in range(0, 100):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
            for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Loading BoW dictionary
cvFile = 'c1_BoW_Sentiment_Model.pkl'
# cv = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('./drive/MyDrive/Colab Notebooks/2 Sentiment Analysis (Basic)/3.1 BoW_Sentiment Model.pkl', "rb")))
cv = pickle.load(open(cvFile, "rb"))


X_fresh = cv.transform(corpus).toarray()
X_fresh.shape

classifier = joblib.load('c2_Classifier_Sentiment_Model')

y_pred = classifier.predict(X_fresh)
print(y_pred)

dataset['predicted_label'] = y_pred.tolist()
dataset.head()

dataset.to_csv("c3_Predicted_Sentiments_Fresh_Dump.tsv",
            sep='\t', encoding='UTF-8', index=False)