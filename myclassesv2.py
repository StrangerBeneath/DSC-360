
class Normalize_Corpus():

    def normalize(self, corpus, html_stripping=True, contraction_expansion=True,
                             accented_char_removal=True, text_lower_case=True,
                             text_lemmatization=True, special_char_removal=True,
                             stopword_removal=True, digits_removal=True):
        self.html_stripping=html_stripping
        self.contraction_expansion = contraction_expansion
        self.accented_char_removal = accented_char_removal
        self.text_lower_case = text_lower_case
        self.text_lemmatization = text_lemmatization
        self.special_char_removal = special_char_removal
        self.stopword_removal = stopword_removal
        self.digits_removal = digits_removal
        import re
        import string
        from bs4 import BeautifulSoup
        import unicodedata
        from contractions import CONTRACTION_MAP
        import spacy
        import nltk
        from nltk.tokenize.toktok import ToktokTokenizer
        tokenizer = ToktokTokenizer()
        stopword_list = nltk.corpus.stopwords.words('english')
        nlp = spacy.load('en_core_web_sm')

        # stripping html
        def strip_html_tags(ingredients):
            soup = BeautifulSoup(ingredients, "html.parser")  # making the soup/
            [s.decompose() for s in soup(['iframe', 'script'])]
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
        def lower_case(text):
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

        def normalize_corpus(corpus, html_stripping=False, contraction_expansion=False,
                             accented_char_removal=False, text_lower_case=False,
                             text_lemmatization=False, special_char_removal=False,
                             stopword_removal=False, digits_removal=False):
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
                corpus = corpus.apply(lambda x: lower_case(x))
            if text_lemmatization:
                print('Text Lemmatization...')
                corpus = corpus.apply(lambda x: lemmatize_text(x))
            if special_char_removal:
                print('Removing Special Characters...')
                corpus = corpus.apply(lambda x: remove_special_characters(x))
            if stopword_removal:
                print('Removing Stopwords...')
                corpus = corpus.apply(lambda x: remove_stopwords(x))
            if digits_removal:
                print('Removing Numbers...')
                corpus = corpus.apply(lambda x: remove_digits(x))
            if special_char_removal: #have some special character lingering;second pass for straglers
                corpus = corpus.apply(lambda x: remove_special_characters(x))
            return corpus
        return normalize_corpus(corpus, html_stripping, contraction_expansion,
                             accented_char_removal, text_lower_case,
                             text_lemmatization, special_char_removal,
                             stopword_removal, digits_removal)

    def BOW(self,norm_corpus):
        import pandas as pd

        print('Bag of Words Model:\n')
        # starting on page 208
        from sklearn.feature_extraction.text import CountVectorizer
        # get bag of words features in sparse format
        cv = CountVectorizer(min_df=0., max_df=1.)
        cv_matrix = cv.fit_transform(norm_corpus)
        # view non-zero feature positions in the sparse matrix
        # print(cv_matrix, '\n')

        # view dense representation
        # warning - might give a memory error if the data is too big
        cv_matrix = cv_matrix.toarray()
        # print(cv_matrix, '\n')

        # get all unique words in the corpus
        vocab = cv.get_feature_names_out()
        # show document feature vectors
        cv_df = pd.DataFrame(cv_matrix, columns=vocab)
        # print(cv_df, '\n')

        # you can set the n-gram range to 1,2 to get unigrams as well as bigrams
        bv = CountVectorizer(ngram_range=(2, 2))
        bv_matrix = bv.fit_transform(norm_corpus)
        bv_matrix = bv_matrix.toarray()
        vocab = bv.get_feature_names_out()
        bv_df = pd.DataFrame(bv_matrix, columns=vocab)
        # print(bv_df, '\n')
        return bv_df

    def TfIdf_Transformer(self,norm_corpus):
        # print('TF-IDF transformer:\n')
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer

        cv = CountVectorizer(min_df=0., max_df=1.)
        cv_matrix = cv.fit_transform(norm_corpus)

        tt = TfidfTransformer(norm='l2', use_idf=True)
        tt_matrix = tt.fit_transform(cv_matrix)
        tt_matrix = tt_matrix.toarray()
        vocab = cv.get_feature_names_out()
        # print(pd.DataFrame(np.round(tt_matrix, 2), columns=vocab), '\n')
        return pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)

    def TfIdf_Vectorizer(self,norm_corpus):
        print('TF-IDF Vectorizer:\n')
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer

        tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True)
        tv_matrix = tv.fit_transform(norm_corpus)
        tv_matrix = tv_matrix.toarray()
        vocab = tv.get_feature_names_out()
        # return tv_matrix, pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
        return pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)


    def Similarity_Matrix(self,tv_matrix):
        import pandas as pd
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(tv_matrix)
        similarity_df = pd.DataFrame(similarity_matrix)

        similarity_df[similarity_df > 0.999] = np.NaN # Sets instances of comparing document to its self NaN

        return similarity_df
