from flask import Flask, render_template,request

import torch
import spacy
import pandas as pd
import evaluate
import numpy as np
from nltk.tokenize import RegexpTokenizer

import nltk
nltk.download('punkt')

# set cuda and device
if torch.cuda.is_available():
    print("GPU is enabled.")
    print("device count: {}, current device: {}".format(torch.cuda.device_count(), torch.cuda.current_device()))
else:
    print("GPU is not enabled.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.blank('id')
app = Flask(__name__)

def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# set tokenizer
from transformers import AutoTokenizer
t5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# define tokenizer function
def tokenize_sample_data(data):
    # Max token size is 14536 and 215 for inputs and labels, respectively.
    # Here I restrict these token size.
    input_feature = t5_tokenizer(data["article"], truncation=True, max_length=1024)
    label = t5_tokenizer(data["summary"], truncation=True, max_length=128)
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }

def tokenize_sentence(arg):
    encoded_arg = t5_tokenizer(arg)
    return t5_tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

# use model
from transformers import AutoModelForSeq2SeqLM
model = (AutoModelForSeq2SeqLM
         .from_pretrained("D:\\NLP APP\\maroon_model")
         .to(device))

from newspaper import Article

@app.route("/")
def msg():
    return render_template('index.html')

@app.route("/summarize-link", methods=['POST', 'GET'])
def getSummaryLink():
    url = request.form['article-link']
    a = Article(url, language='id')
    a.download()
    a.parse()
    body = a.text
    input_ids = t5_tokenizer.encode(body, return_tensors='pt', truncation=True, max_length=1024)
    with torch.no_grad():
        preds = model.generate(
            input_ids.to(device),
            num_beams=15,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
            max_length=256,
        )
    result = t5_tokenizer.batch_decode(preds, skip_special_tokens=True)
    return render_template('summary.html', result=result[0])

@app.route("/summarize",methods=['POST','GET'])
def getSummary():
    body=request.form['article-text']
    # result = model(body, num_sentences=5)
    input_ids = t5_tokenizer.encode(body, return_tensors='pt', truncation=True, max_length=1024)
    with torch.no_grad():
        preds = model.generate(
                input_ids.to(device),
                num_beams=15,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
                max_length=256,
            )
    result = t5_tokenizer.batch_decode(preds, skip_special_tokens=True)
    return render_template('summary.html',result=result[0])

if __name__ =="__main__":
    app.run(debug=True,port=8000)