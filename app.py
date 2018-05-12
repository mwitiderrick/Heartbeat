from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np
import re 
import nltk
from flask import jsonify
from sklearn.naive_bayes import GaussianNB
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def main():
	if request.method == 'GET':
		return render_template('index.html')

	if request.method == 'POST':
		review = request.form['review']	
		corpus = []
		review = re.sub('[^a-zA-Z]', ' ', review)
		review = review.lower()
		review = review.split()
		ps = PorterStemmer()
		review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
		review = ' '.join(review)
		corpus.append(review)
		classifier = joblib.load('classifier.pkl')
		tfidfVectorizer = joblib.load('tfidfVectorizer.pkl')
		x_tfid = tfidfVectorizer.transform(corpus).toarray()
		answer = classifier.predict(x_tfid)
		answer = str(answer[0])
		if answer == '1':
			return "That looks like a positive review"
		else:
			return "You dont seem to have liked that movie."


if __name__ == "__main__":
    app.run()