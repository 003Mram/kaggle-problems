import pandas as pd
from sklearn import model_selection, preprocessing, naive_bayes


ghosts_train = pd.read_csv('D:\\datascience\\projects\\ghosts\\train\\train.csv')
ghosts_train.shape
ghosts_train.info()
y_train = ghosts_train[['type']]
ghosts_train.drop(['id','type','color'], axis =1, inplace = True) #color removed from list

X_train = ghosts_train
classifier = naive_bayes.GaussianNB()
classifier.fit(X_train,y_train)

ghosts_test = pd.read_csv('D:\\datascience\\projects\\ghosts\\test\\test.csv')
ghosts_test.drop(['id','color'], axis =1, inplace = True)
ghosts_test1 = ghosts_test

ghosts_test = pd.read_csv('D:\\datascience\\projects\\ghosts\\test\\test.csv')
ghosts_test['type'] = classifier.predict(ghosts_test1)

ghosts_test.to_csv('D:\\datascience\\projects\\ghosts\\submit\\submit1.csv', index = False, columns =['id', 'type'])




