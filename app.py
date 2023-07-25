from flask import Flask,redirect,url_for,render_template,request
import pandas as pd
from src.pipelines.predict_pipeline import CustomData,Prediction

app = Flask(__name__)

@app.route('/churnpredict',methods=['GET','POST'])
def predict_class():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            seniorcitizen=int(request.form.get('citizen')),
            partner=request.form.get('partner'),
            dependents=request.form.get('dependents'),
            tenure=int(request.form.get('tenure')),
            phoneservice=request.form.get('phoneservice'),
            multiplelines=request.form.get('multiplelines'),
            internetservice=request.form.get('internetservice'),
            onlinesecurity=request.form.get('onlinesecurity'),
            onlinebackup=request.form.get('onlinebackup'),
            deviceprotection=request.form.get('deviceprotection'),
            techsupport=request.form.get('techsupport'),
            streamingtv=request.form.get('streamingtv'),
            streamingmovies=request.form.get('streamingmovies'),
            contract=request.form.get('contract'),
            paperlessbilling=request.form.get('paperless'),
            paymentmethod=request.form.get('paymentmethod'),
            monthlycharges=float(request.form.get('monthly')),
            totalcharges=float(request.form.get('total'))
        )

        df = data.get_as_dataframe()
        pred_obj = Prediction()
        pred = pred_obj.predict_class(df)[0]
        print(f"printing {pred}")
        result = 'is Not' if pred==0 else 'is'

        return render_template(
            'index.html',
            output=f"Based on the analysis, the model predicts that the customer {result} Likely to Churn",
            **request.form,form_data=request.form
            ) 

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
