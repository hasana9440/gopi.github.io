from flask import Flask,render_template,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

import pickle

vectorizer = TfidfVectorizer()
model = pickle.load(open('Quora_Logistic_Exe_new.pkl','rb'))
dataset = pd.read_csv('quora_data1.csv')
x = dataset.question_text
x = vectorizer.fit_transform(x.astype('U'))
answer_dict = {0:'sincere',1:'insincere'}

print(x.shape)

app = Flask(__name__)
    
@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if isinstance(request.form['question'], str):
            question = str(request.form['question'])
            ques = question
            question = [question]
            question = vectorizer.transform(question)
            answer = model.predict(question)
            ans = " The message is " + answer_dict[answer[0]]
            return render_template('index.html',prediction_result=ans,plc=ques)
        else:
            return render_template('index.html',prediction_result="please enter valid input")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
