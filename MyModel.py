import urllib.request
import pickle
from sklearn.externals import joblib
import pandas as pd


class MyModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """
    fix = 1
    model = None
    def __init__(self, fix = 2, url = 'https://shield.mlamp.cn/task/api/file/space/download/bb000281867ad0a18fb40aaa2012d7b1/55766/sklearn_save.m'):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        self.fix = fix
        urllib.request.urlretrieve(url, "model.m")
        self.model = joblib.load('model.m')

    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        print("Predict called - will run identity function")
        if self.model:
            return self.model.predict(X)
        else:
            return "less is more more more more %d" % self.fix


#aa = MyModel()
#test = pd.read_csv("/tmp/test_X.csv")
#test=test.drop(['Unnamed: 0'], axis=1)
#print(aa.predict(test.values))
