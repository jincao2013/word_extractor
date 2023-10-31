import os
import re
import time
import nltk
# from bs4 import BeautifulSoup
# from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import PlaintextCorpusReader, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

__date__ = "Oct. 27, 2023"

__all__ = [
    "Article",
    "check_nltk_data",
    "extract_word_list",
    "get_html",
    "write_html",
]

class Article(object):

    def __init__(self, corpus_coca20000, corpus_longman3000, corpus_ielts):
        # init corpora
        self.corpus_coca20000 = corpus_coca20000
        self.corpus_longman3000 = corpus_longman3000
        self.corpus_ielts = corpus_ielts
        self.pos_dict = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV}
        self.wnl = WordNetLemmatizer()
        self.detokenizer = TreebankWordDetokenizer()

        # Define a custom tokenizer that considers hyphenated words and phrases (seperated by space) as single tokens
        tokenizer = RegexpTokenizer(r'\w+(?:-\w+)+|\w+|\w+\s\w+')

        # Tokenize the corpus using the custom tokenizer
        self.tokenized_coca20000 = [i.lower() for i in tokenizer.tokenize(corpus_coca20000.raw())]
        self.tokenized_longman3000 = [i.lower() for i in tokenizer.tokenize(corpus_longman3000.raw())]
        self.tokenized_ielts = [i.lower() for i in tokenizer.tokenize(corpus_ielts.raw())]

        self.map_coca = {i: self.tokenized_coca20000[i].lower() for i in range(len(self.tokenized_coca20000))}
        self.invmap_coca = {self.tokenized_coca20000[i].lower(): i for i in range(len(self.tokenized_coca20000))}

        # init data
        self.raw = None                 # rank 0
        self.token_sent = []            # rank 2: [ipara, isent]
        self.token = None               # rank 3: [ipara, isent, iword]
        self.pos = None                 # rank 3: [ipara, isent, iword]
        self.pos_short = None           # rank 3: [ipara, isent, iword]
        self.lemma = None               # rank 3: [ipara, isent, iword]
        self.wl_ranks = None            # rank 3: [ipara, isent, iword]

        self.num_paragraphs = None      # rank 0
        self.num_sentences = []         # rank 1: [ipara]
        self.num_words = []             # rank 2: [ipara, isent]

        # extracted word list
        self.wl_coca, self.wl_longman3000, self.wl_ielts = None, None, None
        self.is_coca, self.is_longman3000, self.is_ielts = None, None, None

    def get_word_list_rank_in_coca(self):
        # return [self._get_rank_in_coca(word) for para in self.lemma for sent in para for word in sent]
        return [[[self._get_rank_in_coca(word) for word in sent] for sent in para] for para in self.lemma]

    def _get_rank_in_coca(self, word):
        ranks = set([self.invmap_coca.get(self.wnl.lemmatize(word, pos)) for pos in ['n', 'v', 'a', 'r']])
        try: ranks.remove(None)
        except KeyError: pass
        try: return min(ranks)
        except ValueError: return 30000

    def load_raw(self, raw):
        self.raw = raw

        paragraphs = raw.split('\n')
        self.num_paragraphs = len(paragraphs)
        self.token = [[] for i in range(self.num_paragraphs)]
        self.pos = [[] for i in range(self.num_paragraphs)]
        self.pos_short = [[] for i in range(self.num_paragraphs)]
        self.lemma = [[] for i in range(self.num_paragraphs)]

        self.num_sentences = [None for i in range(self.num_paragraphs)]
        self.num_words = [[] for i in range(self.num_paragraphs)]

        for ipara in range(self.num_paragraphs):
            sentences = nltk.sent_tokenize(paragraphs[ipara])
            self.num_sentences[ipara] = len(sentences)
            self.token_sent.append(sentences)

            self.token[ipara] = [[] for isent in range(self.num_sentences[ipara])]
            self.pos[ipara] = [[] for isent in range(self.num_sentences[ipara])]
            self.pos_short[ipara] = [[] for isent in range(self.num_sentences[ipara])]
            self.lemma[ipara] = [[] for isent in range(self.num_sentences[ipara])]

            self.num_words[ipara] = [[] for isent in range(self.num_sentences[ipara])]

            for isent in range(self.num_sentences[ipara]):
                sent_tokens = nltk.word_tokenize(sentences[isent])
                sent_tagged = nltk.pos_tag(sent_tokens)
                self.num_words[ipara][isent] = len(sent_tokens)
                for tag in sent_tagged:
                    pos = tag[1]
                    pos_short = self.pos_dict.get(pos[0]) or wordnet.NOUN
                    word_lemma = self.wnl.lemmatize(word=tag[0], pos=pos_short).lower()
                    self.pos[ipara][isent].append(pos)
                    self.pos_short[ipara][isent].append(pos_short)
                    self.lemma[ipara][isent].append(word_lemma)
                self.token[ipara][isent] = sent_tokens
        self.wl_ranks = self.get_word_list_rank_in_coca()

    def extract_word_list_coca20000(self, ncut=3000):
        wl_coca = set([self.map_coca.get(word) if word>ncut else None for para in self.wl_ranks for sent in para for word in sent])
        wl_coca.remove(None)
        wl_coca = list(wl_coca)
        wl_coca.sort(key=self.tokenized_coca20000.index)
        is_coca = [[[ncut < word < 21000 for word in sent] for sent in para] for para in self.wl_ranks]
        self.wl_coca, self.is_coca = wl_coca, is_coca
        return wl_coca

    def extract_word_list_longman3000(self):
        wl_longman3000 = list(set(self.tokenized_longman3000) & set(self.wl_coca))
        wl_longman3000.sort(key=self.tokenized_longman3000.index)
        is_longman3000 = [[[self.map_coca.get(word) in wl_longman3000 for word in sent] for sent in para] for para in self.wl_ranks]
        self.wl_longman3000, self.is_longman3000 = wl_longman3000, is_longman3000
        return wl_longman3000, is_longman3000

    def extract_word_list_ielts(self):
        wl_ielts = list(set(self.tokenized_ielts) & set(self.wl_coca))
        wl_ielts.sort(key=self.tokenized_ielts.index)
        is_ielts = [[[self.map_coca.get(word) in wl_ielts for word in sent] for sent in para] for para in self.wl_ranks]
        self.wl_ielts, self.is_ielts = wl_ielts, is_ielts
        return wl_ielts, is_ielts

    def detoken(self, sep=' \n', fmt='txt'):
        raw = ''
        for ipara in range(self.num_paragraphs):
            raw += self.detoken_para(ipara, fmt=fmt)
            raw += sep
        return raw

    def detoken_para(self, ipara, sep=' ', fmt='txt'):
        raw = ''
        for isent in range(self.num_sentences[ipara]):
            raw += self.detoken_sent(ipara, isent, fmt=fmt)
            raw += sep
        return raw

    def detoken_sent(self, ipara, isent, fmt='txt'):
        sent = self.detokenizer.detokenize(self.token[ipara][isent])
        if fmt == 'html':
            c_green = r'#009688'
            c_red = r'#e53935'
            c_blue = r'#0d47a1'
            nw = len(self.token[ipara][isent])
            _wl_coca, _wl_main = [], []
            for iw in range(nw):
                if self.is_longman3000[ipara][isent][iw] | self.is_ielts[ipara][isent][iw]:
                    _wl_main.append(self.token[ipara][isent][iw])
                else:
                    if self.is_coca[ipara][isent][iw]:
                        _wl_coca.append(self.token[ipara][isent][iw])
            sent = self.mark_word_in_sent(_wl_coca, sent, c=c_blue, w='bold')
            sent = self.mark_word_in_sent(_wl_main, sent, c=c_red, w='bold')
        return sent

    def mark_word_in_sent(self, words, sent, c='#0d47a1', w='normal'):
        """
        c:
        w: normal, bold
        """
        for word in words:
            sent = re.sub(r'\b{}\b'.format(word), r'hhhhhh' + word, sent)
        for word in words:
            word_formated = r"<a href='http://www.ldoceonline.com/dictionary/{}' target='_blank'><span style='color: {}; text-decoration: none; font-weight: {};'>{}</span></a>".format(word, c, w, word)
            sent = re.sub(r'\b{}\b'.format(r'hhhhhh' + word), word_formated, sent)
        return sent

    def get_html(self):
        c_green = r'#009688'
        c_red = r'#e53935'
        c_blue = r'#0d47a1'

        html_body = self.get_html_body()
        html = """"""

        # Write the HTML content
        html += '<html>\n'
        html += '<head>\n'
        html += '<title>extracted word list</title>\n'
        html += '</head>\n'
        html += '<style>\n'
        html += 'body {\n'
        html += '  max-width: 800px;\n'
        html += '  margin: 50 auto;\n'
        html += '  background-color: #FAF9F1; /* Change the color code to the desired background color */\n'
        html += '}\n'
        html += 'p {\n'
        html += '  font-size: 20px;\n'
        html += '  color: black;\n'
        html += '  line-height: 1.3; /* Adjust the value to set the desired line spacing */\n'
        html += '}\n'
        html += '</style>\n'

        html += '<body>\n'
        html += html_body
        html += '</body>\n'
        html += '</html>\n'
        return html

    def get_html_body(self):
        c_green = r'#009688'
        c_red = r'#e53935'
        c_blue = r'#0d47a1'

        html = """"""
        html += '<h1>TOC</h1>\n'
        html += '<ul>\n'
        html += '<li><a href="#article">Article</a></li>\n'
        html += '<li><a href="#wl_coca">COCA</a></li>\n'
        html += '<li><a href="#wl_longman">Longman Communication 3000</a></li>\n'
        html += '<li><a href="#wl_ielts">雅思词汇真经</a></li>\n'
        html += '</ul>\n'

        html += '<h1 id="article">Part I: Article</h1>\n'

        html += f'<p>{sum([sent for para in self.num_words for sent in para])} words</p>\n'
        article_html = self.detoken(sep='<br>', fmt='html')
        # article_html = "<br>".join(article.split('\n'))
        # article_html = highlight_word_in_article(list(set(longman_words_hard) | set(ielts_words_hard)), article_html, c=c_red, w='bold')
        # article_html = highlight_word_in_article(list(set(coca_words) - (set(longman_words_hard) | set(ielts_words_hard))), article_html, c=c_blue, w='bold')
        html += '<p style="font-size: 20px; color: black;">{}</p>\n'.format(article_html)

        html += '<h1>Part II: Extracted word list</h1>\n'
        html += f'<h2 id="wl_coca">A. COCA ({len(self.wl_coca)} words)</h2>\n'
        html += self._get_html_table(self.wl_coca, 4, 180, 30, c=c_blue)
        html += '\n'

        html += f'<h2 id="wl_longman">B. Longman Communication 3000 ({len(self.wl_longman3000)} words)</h2>\n'
        html += self._get_html_table(self.wl_longman3000, 4, 180, 30, c=c_red)
        html += '\n'

        html += f'<h2 id="wl_ielts">C. 雅思词汇真经 ({len(self.wl_ielts)} words)</h2>\n'
        html += self._get_html_table(self.wl_ielts, 4, 180, 30, c=c_red)
        html += '\n'
        return html

    def _get_html_table(self, words, words_per_row, cell_width=100, cell_height=50, c='#0d47a1', w='bold'):
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
    c_green = r'#009688'
    c_red = r'#e53935'
    c_blue = r'#0d47a1'
    with open(fname, 'w') as f:
        # Write the HTML content
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('<title>extracted word list</title>\n')
        f.write('</head>\n')
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
    corpus_coca20000 = PlaintextCorpusReader(corpora_path, 'coca-20000-lemma.txt')
    corpus_ielts = PlaintextCorpusReader(corpora_path, 'ielts_zhenjing.txt')
    corpus_longman3000 = PlaintextCorpusReader(corpora_path, 'Longman Communication 3000.txt')

    # check nltk_data
    check_nltk_data()

    # load article
    with open(input_fname, 'r') as f:
        raw = f.read()

    T0 = time.time()
    coca_words, longman_words_hard, ielts_words_hard = extract_word_list(raw, corpus_coca20000, corpus_longman3000, corpus_ielts, num_coca_exclude)
    # write_html(raw, coca_words, longman_words_hard, ielts_words_hard, fname=save_fname)
    html = get_html(raw, coca_words, longman_words_hard, ielts_words_hard)
    # with open(save_fname, 'w') as f:
    #     f.write(html)

    T1 = time.time()

    """
      * debug
    """
    article = Article(corpus_coca20000, corpus_longman3000, corpus_ielts)
    article.load_raw(raw)
    article.extract_word_list_coca20000(ncut=num_coca_exclude)
    article.extract_word_list_ielts()
    article.extract_word_list_longman3000()

    # self = article
    html = article.get_html()

    T2 = time.time()

    with open(save_fname, 'w') as f:
        f.write(html)

