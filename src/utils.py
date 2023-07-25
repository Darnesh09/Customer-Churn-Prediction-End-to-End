import os
import sys
import pickle
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
        
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,x_test,y_train,y_test,models:dict,params:dict):
    try:
        result = {}
        for model_name,model in models.items():
            gs = GridSearchCV(estimator=model,param_grid=params[model_name],cv=5)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            pred = model.predict(x_test)
            accuracy = accuracy_score(y_test,pred)
            result[model_name] = accuracy
        return result

    except Exception as e:
        raise CustomException(e,sys)

    
def load_obj(file_path):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
            
    except Exception as e:
        raise CustomException(e,sys)
