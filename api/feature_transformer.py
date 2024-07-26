from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd


# Define the FeatureEngin
class FeatureEngineeringTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self
        
    
    def transform(self, X):
        X = X.copy()
       
        X["BP_Category"] = pd.cut(X["PR"],bins=[0,120,129,139,float("inf")],
                            labels=["Normal","Prehypertension","Hypertension Stage 1","Hypertension Stage 2",]).astype("object")
        
        X["Age_Group"] = pd.cut(X["Age"],bins=[0,18,30,65,float("inf")],
                            labels=["children","Young adults","Middle Aged","Aged"]).astype("object")
        return X