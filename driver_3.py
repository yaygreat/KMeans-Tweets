from assignment3_3 import *
from sklearn_Kmeans_3 import *

header = ['id','date','tweet']
df=pd.read_csv(open('foxnewshealth.txt'), sep = '|', header=None, names=header, engine='c')
skdf = preprocess(df)
df = preprocess(df)

# sklearnKMeans(skdf)

x = vectorize(df)
clf = K_Means()
clf.fit(x)
predictions = results(x, clf)
df['labels'] = predictions

df.to_csv('scratchKMeans.csv', sep=',', index = None, header=None)

# print(df)
