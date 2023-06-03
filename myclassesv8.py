# **Title: Week 10**
# **Author: Michael J. Montana**
# **Date: 21 May 2023**
# **Modified By: N/A**
# **Description: Classes containting fucntions to preclean data, return metrics, Summation, Parsing function for movie scrips
class Normalize_Corpus():

    def normalize(self, corpus,stopword_list, html_stripping=True, contraction_expansion=True,
                  accented_char_removal=True, text_lower_case=True,
                  text_lemmatization=True, special_char_removal=True,
                  stopword_removal=True, digits_removal=True):
        self.corpus=corpus
        self.html_stripping=html_stripping
        self.contraction_expansion = contraction_expansion
        self.accented_char_removal = accented_char_removal
        self.text_lower_case = text_lower_case
        self.text_lemmatization = text_lemmatization
        self.special_char_removal = special_char_removal
        self.stopword_removal = stopword_removal
        self.digits_removal = digits_removal
        self.stopwords = stopword_list
        import re
        import string
        from bs4 import BeautifulSoup
        import unicodedata
        from contractions import CONTRACTION_MAP
        import spacy
        import nltk
        from nltk.tokenize.toktok import ToktokTokenizer
        from tqdm import tqdm
        import numpy as np
        tokenizer = ToktokTokenizer()
        nlp = spacy.load('en_core_web_sm')

        # stripping html
        def strip_html_tags(ingredients):
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module='bs4')#stops warning display
            soup = BeautifulSoup(ingredients, "html.parser")  # making the soup/
            [s.decompose() for s in soup(['iframe', 'script'])]
            stripped_text = soup.get_text()
            stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
            stripped_text = re.sub(r'\S*https?:\S*', '',stripped_text)  # Removes entire link
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
            text = re.sub(r'\s{2,}', ' ', text)
            text= text.lstrip()
            text = text.replace('\n',' ').replace('&#;','')
            text = text.translate(str.maketrans('', '',string.punctuation))  # removing punctuation using .translate -- https://python-reference.readthedocs.io/en/latest/docs/str/translate.html;  string.punctuation -- https://docs.python
            return text
        # lemmatizes text
        def lemmatize_text(text):
            text = nlp(text)
            text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
            return text
        # removes stop words
        def remove_stopwords(text, is_lower_case=False):
            tokens = tokenizer.tokenize(text)
            tokens = [token.strip() for token in tokens]
            stopwords=self.stopwords
            if stopwords =="":
                stopwords=nltk.corpus.stopwords.words('english')
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
            from tqdm import tqdm
            import pandas as pd
            import numpy as np
            pbar = tqdm(total=9, desc='Cleaning', colour='green', position=0,leave=True)
            if html_stripping:
                # print('Stripping HTML...')
                corpus = corpus.apply(lambda x: strip_html_tags(x))
            pbar.update(1)
            if contraction_expansion:
                # print('Expanding Contratcions...')
                corpus = corpus.apply(lambda x: expand_contractions(x))
            pbar.update(1)
            if accented_char_removal:
                # print('Removing Accent Markings...')
                corpus = corpus.apply(lambda x: remove_accented_chars(x))
            pbar.update(1)
            if text_lower_case:
                # print('Changing Letter Case to Lower...')
                corpus = corpus.apply(lambda x: lower_case(x))
            pbar.update(1)
            if text_lemmatization:
                # print('Text Lemmatization...')
                corpus = corpus.apply(lambda x: lemmatize_text(x))
            pbar.update(1)
            if special_char_removal:
                # print('Removing Special Characters...')
                corpus = corpus.apply(lambda x: remove_special_characters(x))
            pbar.update(1)
            if stopword_removal:
                # print('Removing Stopwords...')
                corpus = corpus.apply(lambda x: remove_stopwords(x))
            pbar.update(1)
            if digits_removal:
                # print('Removing Numbers...')
                corpus = corpus.apply(lambda x: remove_digits(x))
            pbar.update(1)
            if special_char_removal: #have some special character lingering;second pass for straglers
                corpus = corpus.apply(lambda x: remove_special_characters(x))
            pbar.update(1)
            return corpus

        return normalize_corpus(corpus, html_stripping, contraction_expansion,
                             accented_char_removal, text_lower_case,
                             text_lemmatization, special_char_removal,
                             stopword_removal, digits_removal)

    def white_space_remover(self,df):
        import numpy as np
        from tqdm import tqdm
        df.replace('', np.nan, inplace=True)  # Replace empty str w/ NaN
        df.apply(lambda x: x.strip() if isinstance(x, str) else x)  # Strip whitespace
        df.dropna(inplace=True)  # Drop empties
        return df

    def tokenize_sentences(self,df,col):
        import pandas as pd
        import nltk
        def tokenize_sentences(text):
            return nltk.sent_tokenize(text)
        df[col] = df[col].apply(tokenize_sentences)#performing sentence tokenization
        df = df.explode(col)
        df = df.reset_index(drop=True)# Reset the index of the DataFrame
        return df

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
        print(bv_df, '\n')
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

class Model_Evaluation():
    def display_confusion_matrix(self,true_labels, predicted_labels, classes=[1, 0]):
        import pandas as pd
        from sklearn import metrics
        total_classes = len(classes)
        level_labels = [total_classes * [0], list(range(total_classes))]

        cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels,
                                      labels=classes)
        cm_frame = pd.DataFrame(data=cm,
                                columns=pd.MultiIndex(levels=[['Predicted:'], classes],
                                                      codes=level_labels),  # Updated labels -> codes
                                index=pd.MultiIndex(levels=[['Actual:'], classes],
                                                    codes=level_labels))  # Updated labels -> codes
        print(cm_frame)

    def display_classification_report(self,true_labels, predicted_labels, classes=[1, 0]):
        from sklearn import metrics
        report = metrics.classification_report(y_true=true_labels,
                                               y_pred=predicted_labels,
                                               labels=classes)
        print(report)

    def get_metrics(self,true_labels, predicted_labels):
        import numpy as np
        from sklearn import metrics
        print('Accuracy:', np.round(metrics.accuracy_score(true_labels,predicted_labels),4))
        print('Precision:', np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),4))
        print('Recall:', np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),4))
        print('F1 Score:', np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),4))

    def display_model_performance_metrics(self,true_labels, predicted_labels, classes=[1, 0]):
        print('Model Performance metrics:')
        print('-' * 30)
        self.get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
        print('\nModel Classification report:')
        print('-' * 30)
        self.display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, classes=classes)
        print('\nPrediction Confusion Matrix:')
        print('-' * 30)
        self.display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, classes=classes)

    def train_predict_model(self, classifier, train_features, train_labels, test_features, test_labels):
        # build model
        classifier.fit(train_features, train_labels)
        # predict using model
        predictions = classifier.predict(test_features)
        return predictions

class Summarizer():
    def flatten_corpus(self, corpus):
        return ' '.join([document.strip()
                         for document in corpus])

    def compute_ngrams(self, sequence, n):
        return zip(*[sequence[index:]
                     for index in range(n)])

    def get_top_ngrams(self, corpus, ngram_val=1, limit=5):
        import nltk
        from operator import itemgetter
        tokens = nltk.word_tokenize(corpus)

        ngrams = self.compute_ngrams(tokens, ngram_val)
        ngrams_freq_dist = nltk.FreqDist(ngrams)
        sorted_ngrams_fd = sorted(ngrams_freq_dist.items(),
                                  key=itemgetter(1), reverse=True)
        sorted_ngrams = sorted_ngrams_fd[0:limit]
        sorted_ngrams = [(' '.join(text), freq)
                         for text, freq in sorted_ngrams]
        return sorted_ngrams

    def collocation_finder(self,corpus):
        from nltk.collocations import BigramCollocationFinder
        from nltk.metrics import BigramAssocMeasures
        from nltk.collocations import TrigramCollocationFinder
        from nltk.metrics import TrigramAssocMeasures
        print('Collocation Finder:')
        finder = BigramCollocationFinder.from_documents([item.split()
                                                         for item
                                                         in corpus])
        bigram_measures = BigramAssocMeasures()
        print('\tBigram Association Measures:')
        print('\t\t',finder.nbest(bigram_measures.raw_freq, 10))
        print('\t\t',finder.nbest(bigram_measures.pmi, 10))
        finder = TrigramCollocationFinder.from_documents([item.split()
                                                          for item
                                                          in corpus])
        trigram_measures = TrigramAssocMeasures()
        print('\tTrigram Association Measures:')
        print('\t\t',finder.nbest(trigram_measures.raw_freq, 10))
        print('\t\t',finder.nbest(trigram_measures.pmi, 10))

    def get_chunks(self,corpus, grammar=r'NP: {<DT>? <JJ>* <NN.*>+}'):
        import itertools
        import nltk
        stopword_list = nltk.corpus.stopwords.words('english')
        all_chunks = []
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        for sentence in corpus:
            tagged_sents = nltk.pos_tag_sents(
                [nltk.word_tokenize(sentence)])
            chunks = [chunker.parse(tagged_sent)
                      for tagged_sent in tagged_sents]
            wtc_sents = [nltk.chunk.tree2conlltags(chunk)
                         for chunk in chunks]
            flattened_chunks = list(itertools.chain.from_iterable(wtc_sent
                                                                  for wtc_sent in wtc_sents))
            valid_chunks_tagged = [(status, [wtc for wtc in chunk]) for status, chunk in
                                   itertools.groupby(flattened_chunks, lambda word_pos_chunk:
                                    word_pos_chunk[2] != 'O')]
            valid_chunks = [' '.join(word.lower()
                                     for word, tag, chunk in wtc_group
                                        if word.lower() not in stopword_list)
                                            for status, wtc_group in valid_chunks_tagged
                                                if status]
            all_chunks.append(valid_chunks)
        return all_chunks

    def get_tfidf_weighted_keyphrases(self, text, grammar=r'NP: {<DT>? <JJ>* <NN.*>+}',top_n=10):
        import nltk
        from gensim.models import TfidfModel
        from gensim.corpora import Dictionary
        from gensim.summarization import keywords
        from operator import itemgetter
        valid_chunks = self.get_chunks(text, grammar=grammar)
        dictionary = Dictionary(valid_chunks)
        corpus = [dictionary.doc2bow(chunk) for chunk in valid_chunks]
        tfidf = TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        weighted_phrases = {dictionary.get(idx): value
                            for doc in corpus_tfidf for idx, value in doc}
        weighted_phrases = sorted(weighted_phrases.items(), key=itemgetter(1), reverse=True)
        weighted_phrases = [(term, round(wt, 3))
                            for term, wt in weighted_phrases]
        return weighted_phrases[:top_n]

    def keyword(self, text, ratio=1.0, scores=True, lemmatize=False):
        from gensim.summarization import keywords
        key_words = keywords(text, ratio=1.0, scores=True, lemmatize=False)
        print('Keywords:\n', [(item, round(score, 3)) for item, score in key_words][:25])

class Text_Import():
    def __init__(self, file_path):
        self.file_path = file_path

    def read_txt(self, file_path):
        with open(file_path, 'r') as file:
            text = file.read()
        return text

class Parse_Dialogue():
    import pandas as pd
    def __init__(self,movie_name:str,characters:list,script:str):
        self.movie_name=movie_name
        self.characters=characters
        self.script=script

    def character_dialogue(self)-> pd.DataFrame:
        import pandas as pd
        import re
        movie_name=self.movie_name
        characters=self.characters
        script=self.script
        dfs = []  # empty list to store DataFrames for each character
        # print('\n',movie_name)## removed unnecessary
        #for character in tqdm(characters, desc='Parsing Character Dialogue', colour='green'):  # iterate over each character and extract their dialogue  ## removed unnecessary
        for character in characters:
            pattern = r'{}([\s\S]*?)(?=[A-Z]+\n)'.format(character)  # create a regex pattern to match the character with dialogue
            dialogue = re.findall(pattern, script)  # extract the dialogue for the character
            dialogue = [line.strip() for line in dialogue]  # cleaning up some of the white space
            # Check if a new character is encountered
            dialog_dict = {"character": [character] * len(dialogue),
                           "dialogue": dialogue}  # create a dictionary with the character and dialogue for each line spoken by the character
            dfs.append(pd.DataFrame.from_dict(dialog_dict))  # create a DataFrame from the dictionary and append it to the list of DataFrames
        df = pd.concat(dfs, ignore_index=True)  # concatenate each list into a DataFrame
        df.insert(loc=0, column='movie', value=movie_name)  # adds column for the movie name
        return df

class Sentiment_Analysis():
    def get_emotion(self, text):
        emotions = None
        with tqdm(total=1, desc='Getting Emotional', colour='blue') as pbar:
            emotions = emotion(text)[0]
            pbar.update(1)
        return emotions['label'], emotions['score']
