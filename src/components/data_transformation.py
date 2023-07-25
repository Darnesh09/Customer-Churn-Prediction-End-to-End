import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path : str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj = DataTransformationConfig()

    def get_preprocessor_object(self,df):
        try:
            categorical_features = list(df.select_dtypes(include=[object]).columns)
            categorical_features.append('SeniorCitizen')
            numerical_features = list(df.select_dtypes(include=[int,float]).columns)
            numerical_features.remove('Churn')
            numerical_features.remove('SeniorCitizen')
            
            cat_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoding',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            num_pipeline = Pipeline(
                steps = [
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('catpipeline',cat_pipeline,categorical_features),
                    ['num_pipeline',num_pipeline,numerical_features]
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            target_feature = 'Churn'
            train[target_feature].replace({'No':0,'Yes':1},inplace=True)
            test[target_feature].replace({'No':0,'Yes':1},inplace=True)

            preprocessor = self.get_preprocessor_object(train)
            logging.info('Obtained Preprocessing object')

            input_train = train.drop(target_feature,axis=1)
            input_test = test.drop(target_feature,axis=1)

            input_train_processed = preprocessor.fit_transform(input_train)
            input_test_processed = preprocessor.transform(input_test)
            logging.info('Applied preprocessor object on input_train and input_test')

            train_arr = np.c_[input_train_processed,np.array(train[target_feature])]
            test_arr = np.c_[input_test_processed,np.array(test[target_feature])]

            save_object(self.preprocessor_obj.preprocessor_obj_path,preprocessor)

            logging.info('preprocessor object saved to "preprocessor.pkl"')

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e,sys)