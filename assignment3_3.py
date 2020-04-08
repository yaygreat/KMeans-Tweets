import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemming = PorterStemmer()
stops = set(stopwords.words("english"))
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

def preprocess(df):
    df['tweet'] = df['tweet'].str.lower()
    df = df.replace({'‘':'\''}, regex=True)
    df = df.replace({'’':'\''}, regex=True)
    df = df.replace({'#':''}, regex=True)
    df = df.replace({'@\S+':''}, regex=True)
    df = df.replace({'http\S+\s*':''}, regex=True)
    df['words'] = df.apply(identify_tokens, axis = 1)
    df['stemmed_words'] = df.apply(stem_list, axis = 1)
    df['stem_meaningful'] = df.apply(remove_stops, axis=1)
    df['processed'] = df.apply(rejoin_words, axis=1)
    cols_to_drop = ['id','date','tweet','words','stemmed_words','stem_meaningful']
    df.drop(cols_to_drop, inplace=True,axis=1)
    return df

def identify_tokens(row):
    tweet = row['tweet']
    tokens = nltk.word_tokenize(tweet)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

def stem_list(row):
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

def remove_stops(row):
    my_list = row['stemmed_words']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

def rejoin_words(row):
    my_list = row['stem_meaningful']
    joined_words = (" ".join(my_list))
    return joined_words

def vectorize(df):
    v = CountVectorizer()
    x = v.fit_transform(df['processed'])
    # x[x == 0] = -1
    # v = TfidfVectorizer()
    # x = v.fit_transform(df['processed'])
    # x[x == 0] = -1
    x = x.toarray()
    # print(x)
    # print(sum(x))
    return x

def jaccard(a, b):
    I = inter(a,b)
    U = union(a,b)
    return round(1 - (float(I) / U), 4)

def union(listA,listB):
    count = 0
    for (a,b) in zip(listA,listB):
        if (a == 1 or b == 1):
            count += 1
        # if (a != 0):
        #     count += 1
        # if (b != 0):
        #     count += 1
    return count

def inter(listA,listB):
    count = 0
    for (a,b) in zip(listA,listB):
        if (a == 1 and b == 1):
            count += 1
        # if (a == b):
        #     if (a != 0):
        #         if (b != 0):
        #             count += 1
    return count

##############################################################################################
class K_Means:
    def __init__(self, k=2, tol=0.000001, max_iter=3000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}
        # print('data: ', data)

        for i in range(self.k):
            rand = random.randint(0,(len(data)-1))
            print('picking tweet #', rand, ': ', data[rand])
            self.centroids[i] = data[rand]
            # print('data: ', data[random.randint(0,(len(data)-1))])
            # print('sum= ', sum(data[random.randint(0,(len(data)-1))]))

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                # distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                distances = [jaccard(featureset, (self.centroids[centroid])) for centroid in self.centroids]
                # print('distances= ', distances)
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                if abs(np.sum((current_centroid-original_centroid)))/len(data[0]) > self.tol:
                    print('tol < ', abs(np.sum((current_centroid-original_centroid)))/len(data[0]))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
###############################################################################################################

def scatPlot(x, clf):
    pca = PCA(n_components=2).fit(x)
    data2D = pca.transform(x)
    # y_kmeans = c.predict(x)
    y_kmeans = []
    for i in range(len(x)):
        predict_me = np.array(x[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        y_kmeans.append(clf.predict(predict_me))

    plt.scatter(data2D[:,0], data2D[:,1], c=y_kmeans, s=50, cmap='viridis')

    final_centroids = []
    for centroid in clf.centroids:
        # print('clf.centroids= ', clf.centroids[centroid])
        final_centroids.append(clf.centroids[centroid])
    print('final_centroids: ', final_centroids)
    centers2D = pca.transform(final_centroids)

    plt.scatter(centers2D[:,0], centers2D[:,1], 
                marker='x', s=200, linewidths=3, c='r')
    plt.show()

def results(x, clf):
    count0=0
    count1=0
    count2=0
    count3=0
    count4=0
    predictions = []
    for i in range(len(x)):
        predict_me = np.array(x[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = clf.predict(predict_me)
        if(prediction == 0):
            count0 += 1
        if(prediction == 1):
            count1 += 1
        if(prediction == 2):
            count2 += 1
        if(prediction == 3):
            count3 += 1
        if(prediction == 4):
            count4 += 1
        predictions.append(prediction)

    print('predictions')
    print('0: ', count0, 'tweets')
    print('1: ', count1, 'tweets')
    print('2: ', count2, 'tweets')
    print('3: ', count3, 'tweets')
    print('4: ', count4, 'tweets')

    scatPlot(x, clf)

    return predictions
