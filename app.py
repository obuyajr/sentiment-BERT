from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# load the model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# define the home page
@app.route('/')
def home():
    return render_template('home.html')



# define the predict page
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_csv(file,usecols=['product_name','Rate','Summary'])
    texts = df['Summary'].tolist()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    _, predicted = torch.max(outputs.logits, dim=1)
    df['sentiment'] = predicted.tolist()
    df['sentiment'] = df['sentiment'].replace({-1: 'neutral', 0: 'negative', 1: 'positive'})  
   
    
      # draw a bar chart of the sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/sentiment_distribution.png')
    
    return render_template('result.html', tables=[df.to_html(classes='data', header="true")])

if __name__ == '__main__':
    app.run(debug=True)
