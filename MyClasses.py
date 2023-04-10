from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import math
import nltk
from nltk.corpus import wordnet
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd
import re
import spacy
import unicodedata
import string

tokenizer = ToktokTokenizer()
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
class Normalize_Corpus:
       # def normalize(corpus,html_stripping=True,contraction_expansion=True,accented_char_removal=True,text_lower_case=True,text_lemmatization=True,special_char_removal=True,stopword_removal=True,remove_digits=True):
       # could not make it work when using the above function with the ones beneath inside of it
    def normalized_corpus(corpus,html_stripping=True,contraction_expansion=True,accented_char_removal=True,text_lower_case=True,text_lemmatization=True,special_char_removal=True,stopword_removal=True,remove_digits=True):
        # normalize each document in the corpus
        normalized_corpus = []
        # normalized_corpus = pd.Series(normaizled_corpus)
        # for doc in corpus:
        #     doc=pd.Series(doc)
        # stripping html
        def strip_html_tags(text):
            soup = BeautifulSoup(text, "html.parser")  # making the soup/
            [s.extract() for s in soup(['iframe', 'script'])]
            stripped_text = soup.get_text()
            stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
            return stripped_text
        # expansding contractions
        def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                              flags=re.IGNORECASE | re.DOTALL)
            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = contraction_mapping.get(match) \
                    if contraction_mapping.get(match) \
                    else contraction_mapping.get(match.lower())
                expanded_contraction = first_char + expanded_contraction[1:]
                return expanded_contraction
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
            return expanded_text
        # removing accent marks
        def remove_accented_chars(text):
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            return text
        # setting text to lower case
        def text_lower_case(text):
            return text.lower()  # returns the string in lowercase -- https://python-reference.readthedocs.io/en/latest/docs/str/lower.html
        # removes special characters
        def remove_special_characters(text):
            text = re.sub(r'[^a-zA-z0-9\s]', '',text)  # removes all non-alpha/numeric characters -- https://www.dataquest.io/wp-content/uploads/2019/03/python-regular-expressions-cheat-sheet.pdf
            text = text.replace('\n',' ').replace('&#;','').replace('  ',' ')
            text = text.translate(str.maketrans('', '',string.punctuation))  # removing punctuation using .translate -- https://python-reference.readthedocs.io/en/latest/docs/str/translate.html;  string.punctuation -- https://docs.python
            return text
        # lemmatizes text
        def lemmatize_text(text):
            text = nlp(text)
            text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
            return text
        # removes stop words
        def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
            tokens = tokenizer.tokenize(text)
            tokens = [token.strip() for token in tokens]
            if is_lower_case:
                filtered_tokens = [token for token in tokens if token not in stopwords]
            else:
                filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
            filtered_text = ' '.join(filtered_tokens)
            return filtered_text
        # removes numbers
        def remove_digits(text):
            text = re.sub(r'[^\D\s]', '',
                          text)  # removes all numbers -- https://www.dataquest.io/wp-content/uploads/2019/03/python-regular-expressions-cheat-sheet.pdf
            return text
        if html_stripping:
            print('Stripping HTML...')
            corpus = corpus.apply(lambda x: strip_html_tags(x))
        if contraction_expansion:
            print('Expanding Contratcions...')
            corpus = corpus.apply(lambda x: expand_contractions(x))
        if accented_char_removal:
            print('Removing Accent Markings...')
            corpus = corpus.apply(lambda x: remove_accented_chars(x))
        if text_lower_case:
            print('Changing Letter Case to Lower...')
            corpus = corpus.apply(lambda x: text_lower_case(x))
        if text_lemmatization:
            print('Text Lemmatization...')
            corpus = corpus.apply(lambda x: lemmatize_text(x))
        if special_char_removal:
            print('Removing Special Characters...')
            corpus = corpus.apply(lambda x: remove_special_characters(x))
        if stopword_removal:
            print('Removing Stopwords...')
            corpus = corpus.apply(lambda x: remove_stopwords(x))
        if remove_digits:
            print('Removing Numbers...')
            corpus = corpus.apply(lambda x: remove_digits(x))
        if special_char_removal: #have some special character lingering;second pass for straglers
            corpus = corpus.apply(lambda x: remove_special_characters(x))
        return corpus

