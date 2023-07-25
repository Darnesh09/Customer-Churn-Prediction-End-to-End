import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainingConfig:
    model_trainer_obj_path : str = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_trainer_obj = ModelTrainingConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            x_train,x_test,y_train,y_test = (
            train_arr[:,0:-1],
            test_arr[:,0:-1],
            train_arr[:,-1],
            test_arr[:,-1]
            )
            logging.info("train input, test input, train output and test output are assigned")
            
            models = {
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier(),
                'SVC':SVC(),
                'CatBoostClassifier':CatBoostClassifier(),
                'XGBClassifier':XGBClassifier()
            }

            params = {
                'DecisionTreeClassifier':{
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5]
                },
                'AdaBoostClassifier':{
                    'n_estimators': [100,200,300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'algorithm': ['SAMME', 'SAMME.R']
                },
                'SVC':{
                    'kernel': ['linear', 'rbf'],
                    'C': [1, 10, 100],
                    'gamma': [0.01, 0.001, 0.0001]
                },
                'CatBoostClassifier':{
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [3, 4, 5, 6]
                },
                'XGBClassifier':{
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 4, 5, 6],
                    'booster':['gbtree','gblinear']
                }
            }

            model_dict = evaluate_model(x_train,x_test,y_train,y_test,models,params)
            print('model_dict')
            print(model_dict)

            best_accuracy = max(model_dict.values())
            best_model_name = list(model_dict.keys())[list(model_dict.values()).index(best_accuracy)]
            best_model = models[best_model_name]
            print(f"accu : {best_accuracy} | name : {best_model_name} | model : {best_model}")

            if best_accuracy<0.6:
                raise CustomException('No best model found!')

            logging.info('best model with best accuracy is found')
            
            save_object(self.model_trainer_obj.model_trainer_obj_path,best_model)

            logging.info('Best Model object saved in "model.pkl"')

            return (best_model_name,best_accuracy)

        except Exception as e:
            raise CustomException(e,sys)




