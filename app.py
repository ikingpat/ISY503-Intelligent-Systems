from flask import Flask, jsonify, json, render_template, request
import pandas as pd
from utils import build_vocab,load_model,pred


df = pd.read_csv('datasets/cleaned.csv')
vocab = build_vocab(df['Words'])
model = load_model(vocab=vocab, path='models/sentiment_model.pt')


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predict", methods=["post"])
def predict():
    data = request.json
    sentiment = ""
    print(data)
    
    text=data['data']
    probability , sentiment= pred(model=model,vocab=vocab, text=text)
    data = {'sentiment' : sentiment, 'probability' :round(probability*100, 2)}
    print(data)
    return jsonify(data)

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)