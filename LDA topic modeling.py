
import pandas as pd
import numpy as np
from PIL import Image
from pylab import *
import glob
import nltk
import matplotlib.pyplot as plt

import pandas as pd
import re
import gensim
df=pd.read_excel('/Users/manojpadmaraju/Downloads/Reviews and Ratings.xlsx')
corpus = df.review.values.tolist()
my_list=[re.sub(r'[.!@#39$%^&*~฿€‹â:;><?|/Ã©™+=œ]', '', i.lower().replace('quot','')) for i in corpus]
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(my_list))
lemma=[]
lemmatizer = nltk.stem.WordNetLemmatizer()
for token in data_words:
    lemmatized_token = [lemmatizer.lemmatize(item.lower()) for item in token if item.isalpha()]
    lemma.append(lemmatized_token)
removeHTML = []
for i in lemma:
    nonHTML = [word for word in i if word not in ('lt', 'gt', 'quot', 'wa', 'ha')]
    removeHTML.append(nonHTML)
count_list=[]
for i in removeHTML:
    aud=" ".join(i)
    count_list.append(aud)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english',min_df=5,ngram_range=(1,2),strip_accents='unicode')
X2 = vectorizer.fit_transform(count_list)
terms = vectorizer.get_feature_names()
print(X2.shape)
print(X2.toarray())

from sklearn.decomposition import LatentDirichletAllocation
num_topics=6
lda = LatentDirichletAllocation(n_components=num_topics).fit(X2)
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[::-1]]))
print(lda.components_)

# Create Document - Topic Matrix
lda_output = lda.transform(X2)
# column names
topicnames = ["Topic" + str(i) for i in range(lda.n_components)]
# index names
docnames = ["Doc" + str(i) for i in range(len(df))]
# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
dominant_topic2 = df_document_topic.values.argsort()[:, -2]
df_document_topic['dominant_topic'] = dominant_topic
df_document_topic['dominant_topic2'] = dominant_topic2
df_document_topic.head(10)
restaurant= df_document_topic.iloc[0:10].reset_index()
movie= df_document_topic.iloc[500:510].reset_index()
print(df_document_topic)
print(movie)
print(restaurant)
df_document_topic.to_csv('document_topic1.csv')
movie.to_csv('movie1.csv')
restaurant.to_csv('restaurant1.csv')

# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda, n_words=5):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda, n_words=5)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
print(df_topic_keywords)

df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
print(df_topic_distribution.sort_values('Topic Num', ascending=False))



