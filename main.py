from scipy.io import arff
import urllib.request
import io
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


url = "http://www.ece.uah.edu/~thm0009/icsdatasets/gas_final.arff"
ftpstream = urllib.request.urlopen(url)
data, meta = arff.loadarff(io.StringIO(ftpstream.read().decode('utf-8')))


from dbn.tensorflow import SupervisedDBNClassification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from common.models.boltzmann import dbn

df = pd.DataFrame(data)
df['result'] = df['result'].apply(lambda x: int(x.decode('utf-8')))

df.columns = df.columns.str.replace(' ', '')
X = df.drop(columns=['result'], axis=1)
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# # XgBoost
# model = xgb.XGBClassifier(objective='multi:softmax',
#                           num_class=8, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("started")
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")


#dbn
# print("success")

# classifier = SupervisedDBNClassification(hidden_layers_structure=[
#                                         256, 256], learning_rate_rbm=0.05, learning_rate=0.1, n_epochs_rbm=10, n_iter_backprop=100, batch_size=32, activation_function='relu', dropout_p=0.2)
# print(x_train, y_train)

# classifier.fit(x_train, y_train)
# y_pred = classifier.predict(x_test)


# print('nAccuracy of Prediction: %f' % accuracy_score(x_test, y_pred))
