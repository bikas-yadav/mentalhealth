from flask import Flask, render_template, request, redirect, url_for, session
import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertModel
import pickle
import numpy as np
import os

# Define Flask app
app = Flask(__name__)
app.secret_key = 'e8b7c2d0f47a89e9123c5a8f1b6a8b42'

# Static login credentials
USERNAME = 'admin'
PASSWORD = 'pass123'

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
clf = pickle.load(open("model.pkl", "rb"))

# Function to generate BERT embedding
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Extract text from uploaded PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# Analyze multiple texts and return one overall result
def summarize_overall_result(texts):
    label_map = {
        0: "✅ Normal",
        1: "⚠️ Stress",
        2: "🆘 Depression"
    }
    label_scores = {0: [], 1: [], 2: []}

    for text in texts:
        emb = get_bert_embedding(text).reshape(1, -1)
        proba = clf.predict_proba(emb)[0]
        for label in [0, 1, 2]:
            label_scores[label].append(proba[label])

    avg_scores = {label: sum(scores)/len(scores) for label, scores in label_scores.items()}
    predicted_label = max(avg_scores, key=avg_scores.get)
    return label_map[predicted_label], avg_scores[predicted_label] * 100

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname == USERNAME and pwd == PASSWORD:
            session['user'] = uname
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    result_summary = None
    if request.method == 'POST':
        pdf = request.files['pdf_file']
        if pdf:
            path = os.path.join("uploads", pdf.filename)
            pdf.save(path)
            text = extract_text_from_pdf(path)
            paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]
            result_summary = summarize_overall_result(paras)

    return render_template("index.html", result=result_summary)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
