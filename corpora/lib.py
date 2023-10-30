import os
import re
import nltk
# from bs4 import BeautifulSoup
# from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import PlaintextCorpusReader, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

__date__ = "Oct. 27, 2023"

__all__ = [
    "check_nltk_data",
    "extract_word_list",
    "get_html",
    "write_html",
]

class Article(object):

    def __init__(self, corpora_coca20000, corpora_longman3000, corpora_ielts):
        self.corpora_coca20000 = corpora_coca20000
        self.corpora_longman3000 = corpora_longman3000
        self.corpora_ielts = corpora_ielts
        self.raw = None

    def load_raw(self, article):
        pass

    def represent_raw(self):
        pass

    def format_in_html(self):
        pass

    def extract_word_list_coca20000(self):
        pass

    def extract_word_list_longman3000(self):
        pass

    def extract_word_list_ielts(self):
        pass

def check_nltk_data():
    try:
        nltk.word_tokenize('')
    except LookupError:
        nltk.download('punkt')
    try:
        test_stemmer = WordNetLemmatizer()
        test_stemmer.lemmatize('')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.pos_tag([''])
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

def extract_word_list(article, corpus_coca20000, corpus_longman3000, corpus_ielts_zhenjing, num_coca_exclude=3000):
    """
    Reference:

    """
    # Define a custom tokenizer that considers hyphenated words and phrases (seperated by space) as single tokens
    tokenizer = RegexpTokenizer(r'\w+(?:-\w+)+|\w+|\w+\s\w+')

    # Tokenize the corpus using the custom tokenizer
    tokenized_corpus_coca20000 = tokenizer.tokenize(corpus_coca20000.raw())
    tokenized_corpus_longman3000 = tokenizer.tokenize(corpus_longman3000.raw())
    tokenized_corpus_ielts_zhenjing = tokenizer.tokenize(corpus_ielts_zhenjing.raw())

    # setup stemmer
    stemmer_verb = WordNetLemmatizer()
    stemmer_snowball = SnowballStemmer("english")

    # Tokenize the article into words
    # tokenized_article = nltk.word_tokenize(article, language='english')
    tokenized_article = lemmatization_of_article(article)
    num_words_total = len(tokenized_article)
    tokenized_article = set(tokenized_article)
    num_diff_words_total = len(tokenized_article)

    # tokenized_article = [stemmer_verb.lemmatize(i, pos='v') for i in tokenized_article]
    # tokenized_article = [stemmer_snowball.stem(i) for i in tokenized_article]
    # tokenized_article = set(tokenized_article)

    # extract hard words
    excluded_words = set(tokenized_corpus_coca20000[:num_coca_exclude])
    excluded_words = excluded_words | {'an'}

    # extract words beyond top 3000 in coca
    coca_words = (tokenized_article & set(tokenized_corpus_coca20000)) - excluded_words
    coca_words = list(coca_words)
    coca_words.sort(key=tokenized_corpus_coca20000.index)

    # extract hard words wrt longman 3000
    longman_words = set(tokenized_corpus_longman3000) & tokenized_article
    longman_words_hard = list(longman_words - excluded_words)  # the word not in top 3000 coca list
    longman_words_easy = list(longman_words - set(longman_words_hard))  # the word in top 3000 coca list
    longman_words_hard.sort(key=tokenized_corpus_longman3000.index)
    longman_words_easy.sort(key=tokenized_corpus_longman3000.index)

    # extract hard words wrt ielts_zhenjing
    ielts_words = set(tokenized_corpus_ielts_zhenjing) & tokenized_article
    ielts_words_hard = list(ielts_words - excluded_words)  # the word not in top 3000 coca list
    ielts_words_easy = list(ielts_words - set(ielts_words_hard))  # the word in top 3000 coca list
    ielts_words_hard.sort(key=tokenized_corpus_ielts_zhenjing.index)
    ielts_words_easy.sort(key=tokenized_corpus_ielts_zhenjing.index)

    return coca_words, longman_words_hard, ielts_words_hard

def get_html(article, coca_words, longman_words_hard, ielts_words_hard):
    c_green = r'009688'
    c_red = r'e53935'
    c_blue = r'#0d47a1'
    html = """"""

    # Write the HTML content
    # html += '<html>\n'
    # html += '<head>\n'
    # html += '<title>extracted word list</title>\n'
    # html += '<\head>\n'
    # html += '<style>\n'
    # html += 'body {\n'
    # html += '  max-width: 800px;\n'
    # html += '  margin: 50 auto;\n'
    # html += '  background-color: #FAF9F1; /* Change the color code to the desired background color */\n'
    # html += '}\n'
    # html += 'p {\n'
    # html += '  font-size: 20px;\n'
    # html += '  color: black;\n'
    # html += '  line-height: 1.3; /* Adjust the value to set the desired line spacing */\n'
    # html += '}\n'
    # html += '</style>\n'

    # html += '<body>\n'
    html += '<h1>TOC</h1>\n'
    html += '<ul>\n'
    html += '<li><a href="#article">Article</a></li>\n'
    html += '<li><a href="#wl_coca">COCA</a></li>\n'
    html += '<li><a href="#wl_longman">Longman Communication 3000</a></li>\n'
    html += '<li><a href="#wl_ielts">雅思词汇真经</a></li>\n'
    html += '</ul>\n'

    html += '<h1 id="article">Part I: Article</h1>\n'
    html += f'<p>{len(nltk.word_tokenize(article))} words</p>\n'
    article_html = "<br>".join(article.split('\n'))
    article_html = highlight_word_in_article(list(set(longman_words_hard) | set(ielts_words_hard)), article_html,
                                             c=c_red, w='bold')
    article_html = highlight_word_in_article(list(set(coca_words) - (set(longman_words_hard) | set(ielts_words_hard))),
                                             article_html, c=c_blue, w='bold')
    html += '<p style="font-size: 20px; color: black;">{}</p>\n'.format(article_html)

    # file.write('<p style="font-weight: bold;">This paragraph has bold text.</p>\n')
    # file.write('<p style="font-style: italic;">This paragraph has italic text.</p>\n')

    html += '<h1>Part II: Extracted word list</h1>\n'
    html += f'<h2 id="wl_coca">A. COCA ({len(coca_words)} words)</h2>\n'
    html += generate_html_table(coca_words, 4, 180, 30, c=c_blue)
    html += '\n'

    html += f'<h2 id="wl_longman">B. Longman Communication 3000 ({len(longman_words_hard)} words)</h2>\n'
    # f.write('<h3>Hard words:</h3>\n')
    html += generate_html_table(longman_words_hard, 4, 180, 30, c=c_red)
    html += '\n'

    # f.write('<h3>Easy words:</h3>\n')
    # f.write(generate_html_table(ielts_words_easy, 4, 180, 30))

    html += f'<h2 id="wl_ielts">C. 雅思词汇真经 ({len(ielts_words_hard)} words)</h2>\n'
    # f.write('<h3>Hard words:</h3>\n')
    html += generate_html_table(ielts_words_hard, 4, 180, 30, c=c_red)
    html += '\n'

    # f.write('<h3>Easy words:</h3>\n')
    # f.write(generate_html_table(longman_words_easy, 4, 180, 30))

    # html += '</body>\n'
    # html += '</html>\n'
    return html

def write_html(article, coca_words, longman_words_hard, ielts_words_hard, fname=r'article.html'):
    c_green = r'009688'
    c_red = r'e53935'
    c_blue = r'#0d47a1'
    with open(fname, 'w') as f:
        # Write the HTML content
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('<title>extracted word list</title>\n')
        f.write('<\head>\n')
        f.write('<style>\n')
        f.write('body {\n')
        f.write('  max-width: 800px;\n')
        f.write('  margin: 50 auto;\n')
        f.write('  background-color: #FAF9F1; /* Change the color code to the desired background color */\n')
        f.write('}\n')
        f.write('p {\n')
        f.write('  font-size: 20px;\n')
        f.write('  color: black;\n')
        f.write('  line-height: 1.3; /* Adjust the value to set the desired line spacing */\n')
        f.write('}\n')
        f.write('</style>\n')

        f.write('<body>\n')
        f.write('<h1>TOC</h1>\n')
        f.write('<ul>\n')
        f.write('<li><a href="#article">Article</a></li>\n')
        f.write('<li><a href="#wl_coca">COCA</a></li>\n')
        f.write('<li><a href="#wl_longman">Longman Communication 3000</a></li>\n')
        f.write('<li><a href="#wl_ielts">雅思词汇真经</a></li>\n')
        f.write('</ul>\n')

        f.write('<h1 id="article">Part I: Article</h1>\n')
        f.write(f'<p>{len(nltk.word_tokenize(article))} words</p>\n')
        article_html = "<br>".join(article.split('\n'))
        article_html = highlight_word_in_article(list(set(longman_words_hard) | set(ielts_words_hard)), article_html, c=c_red, w='bold')
        article_html = highlight_word_in_article(list(set(coca_words)-(set(longman_words_hard)|set(ielts_words_hard))), article_html, c=c_blue, w='bold')
        f.write('<p style="font-size: 20px; color: black;">{}</p>\n'.format(article_html))

        # file.write('<p style="font-weight: bold;">This paragraph has bold text.</p>\n')
        # file.write('<p style="font-style: italic;">This paragraph has italic text.</p>\n')

        f.write('<h1>Part II: Extracted word list</h1>\n')
        f.write(f'<h2 id="wl_coca">A. COCA ({len(coca_words)} words)</h2>\n')
        f.write(generate_html_table(coca_words, 4, 180, 30, c=c_blue))

        f.write(f'<h2 id="wl_longman">B. Longman Communication 3000 ({len(longman_words_hard)} words)</h2>\n')
        # f.write('<h3>Hard words:</h3>\n')
        f.write(generate_html_table(longman_words_hard, 4, 180, 30, c=c_red))

        # f.write('<h3>Easy words:</h3>\n')
        # f.write(generate_html_table(ielts_words_easy, 4, 180, 30))

        f.write(f'<h2 id="wl_ielts">C. 雅思词汇真经 ({len(ielts_words_hard)} words)</h2>\n')
        # f.write('<h3>Hard words:</h3>\n')
        f.write(generate_html_table(ielts_words_hard, 4, 180, 30, c=c_red))

        # f.write('<h3>Easy words:</h3>\n')
        # f.write(generate_html_table(longman_words_easy, 4, 180, 30))

        f.write('</body>\n')
        f.write('</html>\n')

def lemmatization_of_article(article):
    """
    see the following for reference:
    https://www.cnblogs.com/jclian91/p/9898511.html
    https://stackoverflow.com/questions/1787110/what-is-the-difference-between-lemmatization-vs-stemming
    https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
    https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    """
    lemma = WordNetLemmatizer()
    lemma_article = []
    sentences = nltk.sent_tokenize(article)
    pos_dict = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV}
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged_sent = nltk.pos_tag(tokens)
        for tag in tagged_sent:
            # method 1
            pos = pos_dict.get(tag[1][0]) or wordnet.NOUN
            lemma_article.append(lemma.lemmatize(word=tag[0], pos=pos).lower())
            # method 2
            # lemma_article.append(lemma.lemmatize(word=tag[0], pos='n').lower())
            # lemma_article.append(lemma.lemmatize(word=tag[0], pos='v').lower())
            # lemma_article.append(lemma.lemmatize(word=tag[0], pos='a').lower())
            # lemma_article.append(lemma.lemmatize(word=tag[0], pos='r').lower())
    return lemma_article

def highlight_word_in_article(highlight_words, article_html, c='#0d47a1', w='normal'):
    """
    c:
    w: normal, bold
    """
    for word in highlight_words:
        article_html = re.sub(r'\b{}\b'.format(word), r'hhhhhh' + word, article_html)

    for word in highlight_words:
        word_formated = r"<a href='http://www.ldoceonline.com/dictionary/{}' target='_blank'><span style='color: {}; text-decoration: none; font-weight: {};'>{}</span></a>".format(
            word, c, w, word)
        article_html = re.sub(r'\b{}\b'.format(r'hhhhhh' + word), word_formated, article_html)
    return article_html

def generate_html_table(words, words_per_row, cell_width=100, cell_height=50, c='#0d47a1', w='bold'):
    table_html = "<table>\n"

    table_html += "<style>"
    table_html += "td {"
    table_html += f"width: {cell_width}px;"
    table_html += f"height: {cell_height}px;"
    table_html += "font-size: 20px;"
    # table_html += "font-weight: bold;"
    table_html += "}"
    table_html += "a {"
    table_html += "text-decoration: none;"
    # table_html += "color: inherit;"
    # table_html += r"color: {};".format(c)
    table_html += "}"
    table_html += "</style>\n"

    # Generate table headers
    # table_html += "<tr>"
    # table_html += "<th>Index</th>"
    # table_html += "<th>Word</th>"
    # table_html += "</tr>\n"

    # Generate table rows
    for index, word in enumerate(words):
        if index % words_per_row == 0:
            table_html += "<tr>"

        # table_html += f"<td>{index + 1}</td>"
        table_html += r"<td><a href='http://www.ldoceonline.com/dictionary/{}' target='_blank'><span style='color: {}; text-decoration: none; font-weight: {};'>{}</span></a></td>".format(
            word, c, w, word
        )
        # table_html += f"<td>{word}</td>"

        if (index + 1) % words_per_row == 0 or (index + 1) == len(words):
            table_html += "</tr>\n"

    table_html += "</table>"

    return table_html


if __name__ == "__main__":
    os.chdir('/Users/jincao/Downloads/temp')

    # parameters
    num_coca_exclude = 3000  # this will exclude the top 3000 coca words
    # corpora_path = r'./corpora'
    corpora_path = os.path.join(os.path.split(__file__)[0], r'../corpora')
    input_fname = r'wsj_whats_news_20231027.txt'
    # input_fname = sys.argv[1]
    save_fname = input_fname + '.html'

    # load corpus
    corpus_coca20000 = PlaintextCorpusReader(corpora_path, 'coca-20000.txt')
    corpus_ielts_zhenjing = PlaintextCorpusReader(corpora_path, 'ielts_zhenjing.txt')
    corpus_longman3000 = PlaintextCorpusReader(corpora_path, 'Longman Communication 3000.txt')

    # check nltk_data
    check_nltk_data()

    # load article
    with open(input_fname, 'r') as f:
        article = f.read()

    coca_words, longman_words_hard, ielts_words_hard = extract_word_list(article, corpus_coca20000, corpus_longman3000, corpus_ielts_zhenjing, num_coca_exclude)
    write_html(article, coca_words, longman_words_hard, ielts_words_hard, fname=save_fname)
    # html = get_html(article, coca_words, longman_words_hard, ielts_words_hard)
    # with open(save_fname, 'w') as f:
    #     f.write(html)

    """
      * debug
    """


