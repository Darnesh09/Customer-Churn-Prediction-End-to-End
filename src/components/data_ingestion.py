import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Initiating started for data ingestion')
        try:
            df = pd.read_csv("notebook/data/Telco-Customer-Churn.csv")
            logging.info("reading the data as DataFrame")
            df.drop('customerID',axis=1,inplace=True)
            df = df[df['TotalCharges']!=' ']
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
            df.replace(['No phone service','No internet service'],'No',inplace=True)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw data created")

            train,test = train_test_split(df,test_size=0.3,random_state=42)

            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info('Train data created')

            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Test data created')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    transformer = DataTransformation()
    train_arr,test_arr,preprocessor_path = transformer.initiate_data_transformation(train_path,test_path)
    trainer = ModelTraining()
    model_name,accuracy = trainer.initiate_model_training(train_arr,test_arr)
    print(model_name,accuracy)