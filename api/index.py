import os
from flask import Flask, render_template, request
import nltk
from nltk.corpus import PlaintextCorpusReader, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from corpora.lib import *

__date__ = "Oct. 27, 2023"

app = Flask(__name__)

# def process_article(article):
#     num_coca_exclude = 3000  # this will exclude the top 3000 coca words
#     corpora_path = r'corpora'
#
#     # load corpus
#     corpus_coca20000 = PlaintextCorpusReader(corpora_path, 'coca-20000.txt')
#     corpus_ielts_zhenjing = PlaintextCorpusReader(corpora_path, 'ielts_zhenjing.txt')
#     corpus_longman3000 = PlaintextCorpusReader(corpora_path, 'Longman Communication 3000.txt')
#
#     # check nltk_data
#     check_nltk_data()
#
#     coca_words, longman_words_hard, ielts_words_hard = extract_word_list(article, corpus_coca20000, corpus_longman3000, corpus_ielts_zhenjing, num_coca_exclude)
#     processed_article = get_html(article, coca_words, longman_words_hard, ielts_words_hard)
#     return processed_article

def process_article(raw, num_coca_exclude=3000):
    # num_coca_exclude = 3000  # this will exclude the top 3000 coca words
    corpora_path = r'corpora'

    # load corpus
    corpus_coca20000 = PlaintextCorpusReader(corpora_path, 'coca-20000-lemma.txt')
    corpus_ielts = PlaintextCorpusReader(corpora_path, 'ielts_zhenjing.txt')
    corpus_longman3000 = PlaintextCorpusReader(corpora_path, 'Longman Communication 3000.txt')

    # check nltk_data
    check_nltk_data()

    article = Article(corpus_coca20000, corpus_longman3000, corpus_ielts)
    article.load_raw(raw)
    article.extract_word_list_coca20000(ncut=num_coca_exclude)
    article.extract_word_list_ielts()
    article.extract_word_list_longman3000()
    processed_article = article.get_html()
    return processed_article

@app.route('/', methods=['GET', 'POST'])
def main():
    nltk.data.path = ['nltk_data']
    if request.method == 'POST':
        article = request.form['article']
        num_coca_exclude = int(request.form['num_coca_exclude'])
        html_processed = process_article(article, num_coca_exclude)
        return render_template('result.html', html_processed=html_processed)
    return render_template('index.html')

if __name__ == '__main__':
    """
      note: 
      to run in vercel
      1. set corpora_path = r'corpora'
      2. set nltk.data.path = ['nltk_data']
    """
    app.run()