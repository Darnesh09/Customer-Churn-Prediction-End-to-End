import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj


class Prediction():
    def predict_class(self,df):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_obj(model_path)
            preprocessor = load_obj(preprocessor_path)
            processed_df = preprocessor.transform(df)
            pred = model.predict(processed_df)
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        gender:str,
        seniorcitizen:int,
        partner:str,
        dependents:str,
        tenure:int,
        phoneservice:str,
        multiplelines:str,
        internetservice:str,
        onlinesecurity:str,
        onlinebackup:str,
        deviceprotection:str,
        techsupport:str,
        streamingtv:str,
        streamingmovies:str,
        contract:str,
        paperlessbilling:str,
        paymentmethod:str,
        monthlycharges:float,
        totalcharges:float
    ):

        self.gender = gender
        self.seniorcitizen = seniorcitizen
        self.partner = partner
        self.dependents = dependents
        self.tenure = tenure
        self.phoneservice = phoneservice
        self.multiplelines = multiplelines
        self.internetservice = internetservice
        self.onlinesecurity = onlinesecurity
        self.onlinebackup = onlinebackup
        self.deviceprotection = deviceprotection
        self.techsupport = techsupport
        self.streamingtv = streamingtv
        self.streamingmovies = streamingmovies
        self.contract = contract
        self.paperlessbilling = paperlessbilling
        self.paymentmethod = paymentmethod
        self.monthlycharges = monthlycharges
        self.totalcharges = totalcharges

    def get_as_dataframe(self):
        try:
            customdata_dict = {
                    'gender':[self.gender],
                    'SeniorCitizen':[self.seniorcitizen],
                    'Partner':[self.partner],
                    'Dependents':[self.dependents],
                    'tenure':[self.tenure],
                    'PhoneService':[self.phoneservice],
                    'MultipleLines':[self.multiplelines],
                    'InternetService':[self.internetservice],
                    'OnlineSecurity':[self.onlinesecurity],
                    'OnlineBackup':[self.onlinebackup],
                    'DeviceProtection':[self.deviceprotection],
                    'TechSupport':[self.techsupport],
                    'StreamingTV':[self.streamingtv],
                    'StreamingMovies':[self.streamingmovies],
                    'Contract':[self.contract],
                    'PaperlessBilling':[self.paperlessbilling],
                    'PaymentMethod':[self.paymentmethod],
                    'MonthlyCharges':[self.monthlycharges],
                    'TotalCharges':[self.totalcharges]
            }
        
            return pd.DataFrame(customdata_dict)

        except Exception as e:
            raise CustomException(e,sys)
