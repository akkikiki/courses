#TODO: use word embedding feature
#TODO: output what featureswe used
#TODO: Implement Stanford NER with preprocessing enabled
#TODO: That means, do the preprocessing before NER is ran.
#TODO: Use LIME

from csv import DictReader, DictWriter

import re
import argparse
import numpy as np
from numpy import array
from collections import defaultdict
from gensim.models import Word2Vec
#from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

from nltk.tag import StanfordNERTagger
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import nltk
ps = nltk.stem.porter.PorterStemmer()
#st = StanfordNERTagger('/Users/Fujinuma/Software/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz',
#                       '/Users/Fujinuma/Software/stanford-ner-2014-06-16/stanford-ner.jar',
#                       encoding='utf-8')

word_embeddings = False
if word_embeddings:
    w2v_model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


"""
How to build a pipeline:
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

About Hashing Vectorizer:
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html

How to add features:
https://github.com/chrisketelsen/csci5622notebooks/blob/master/15_Feature_Engineering.ipynb
"""

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kVERB_FIELD = 'verb'
kPAGE_FIELD = 'page'
kTROPE_FIELD = 'trope'
STOPWORDS = ['a', 'to', 'and', 'is', 'has', 'had', 'it', 'as', 'does', 'also']

def preprocess_sentence(sentence):
    punctuations = [',', '(', ')', '"', ':', ';', '*', '/', '!']
    #sentence = re.sub('([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen)', '__DIGIT__', sentence)

    for punctuation in punctuations:
        sentence = sentence.replace(punctuation, ' ')
    sentence = sentence.replace('.','')
    sentence = sentence.replace('-',' - ')
    sentence = sentence.replace('?',' ? ')
    sentence = sentence.strip()
    # TODO: Implement this
    return sentence

def read_genre_list():
    dic_geners = defaultdict(list)
    f = open("genres.list", errors='ignore')
    #f = open("genres.list")
    for line in f:
        columns = line.strip().split()
        #print(columns)
        if line.startswith('"'):
            title = re.findall('".*"', line)[0].replace('"', '').replace(" ", "")
            #print(title.lower(), columns[-1])
            if not columns[-1] in dic_geners[title.lower()]:
                dic_geners[title.lower()].append(columns[-1])
    return dic_geners
        
def read_running_list():
    dic_geners = defaultdict(int)
    f = open("running-times.list", errors='ignore')
    for line in f:
        columns = line.strip().split()
        if line.startswith('"') :
            if columns[-1][-1] == ")":
                columns = columns[:-1] # skipping "(DVD)"
            if re.match("^([0-9]+)$", columns[-1]):
                title = re.findall('".*"', line)[0].replace('"', '').replace(" ", "")
                dic_geners[title.lower()] += int(columns[-1])
    return dic_geners

dic_genres = read_genre_list()
dic_runningtime = read_running_list()

class Featurizer:
    def __init__(self):
        self.vectorizer = DictVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        top = 100
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-top:]
            bottom10 = np.argsort(classifier.coef_[0])[:top]
            print("Pos: %s" % ", ".join(feature_names[top10]))
            print("Neg: %s" % ", ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-top:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))


def extract_feature(X, x, stanford_tokens):
    features_used = []# TODO: Implement this

    dic_text_field = defaultdict(int)
    movie_title = x[kPAGE_FIELD]
    # dic_text_field["MOVIE_TITLE=" + movie_title] = 1

    """
    Normalizing the specific trope name
    """
    if "WHAM" in x[kTROPE_FIELD]:
        x[kTROPE_FIELD] = x[kTROPE_FIELD].replace("WHAM", "Wham")
    #if not movie_title.lower() in dic_genres:
    #    print("%s not found" % movie_title.lower())
    genres = dic_genres.get(movie_title.lower(), ["NONE"])
    # runningtime = dic_runningtime.get(movie_title.lower(), 1)
    if not args.exclude_genres:
        for genre in genres:
            if genre == "NONE":
                continue
            dic_text_field["GENRE=" + genre] = 1
            if not args.exclude_tropes:
                dic_text_field[x[kTROPE_FIELD] + "_GENRE=" + genre] = 1

    # TODO: Just Split based on Capital Characters
    # segment_tropes(x)

    if not args.exclude_tropes:
        target_tropes = ["Kill", "Spoil", "Gambit", "Wham", "Heroic", "Ending",
                         "Death", "Dead", "Die", "Chekhovs", "Machina", "Genre", "Foreshadowing"]
        #"Diabolus ex Machina"
        for target_trope in target_tropes:
            if target_trope in x[kTROPE_FIELD]:
                dic_text_field["TROPE=" + target_trope] = 1

        ending_tropes = ["Ending", "Machina", "Kill"]
        for ending_trope in ending_tropes:
            if ending_trope in x[kTROPE_FIELD]:
                dic_text_field["ENDING_RELATED_TROPES"] += 1

    # dic_text_field["RUNNING_TIME="] = np.log(runningtime)
    sentence = x[kTEXT_FIELD]
    punctuations = [',', '(', ')', '"', ':', ';', '*', '/', '!']
    if '"' in sentence:
        dic_text_field["QUOTATION_MARK"] = 1

    sentence = preprocess_sentence(sentence)

    for punctuation in punctuations:
        sentence = sentence.replace(punctuation, ' ')
    #sentence = sentence.replace('.',' .')
    #sentence = sentence.replace('.',' . ')
    sentence = sentence.replace('.','')
    sentence = sentence.replace('-',' - ')
    sentence = sentence.replace('?',' ? ')
    sentence = sentence.strip()
    #print(sentence)
    #sentence = x[kTEXT_FIELD].replace('.','')
    #sentence = re.sub('([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen)', '__DIGIT__', sentence)

#
    if args.stanford:
        tokens = stanford_tokens
    elif args.ner:
        tokenized_sentence = nltk.word_tokenize(sentence)
        tagged_sentence = nltk.pos_tag(tokenized_sentence)
        tokens = nltk.ne_chunk(tagged_sentence)
    else:
        tokens = sentence.split()

    #    """
    #for stopword in STOPWORDS:
    #    http://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-python-list
    #    """

    processed_words = []
    for token in tokens:
        if not args.stanford and not args.ner:
            word = token
        else:
            word = token[0]
        """
        Stanford NER
        """
        if args.stanford and token[1] != 'O':
            word = "__" + token[1] + "__"
        """
        NLTK ACE-trained NER
        """
        if hasattr(token, 'label') and token.label:
            word = "__" + token.label() + "__"

        if word == "sucide":
            word = "suicide"

        if word_embeddings:
            w2v_feature(dic_text_field, word) # pre-stemmed word

        if not args.exclude_tropes:
            for target_trope in target_tropes:
                if target_trope in x[kTROPE_FIELD]:
                    dic_text_field["TROPE=" + target_trope + "__" + word] = 1

        if word.lower() in ["kill", "kills", "killed", "killing", "killer",
                    "murder", "murders", "murdered", "murdering", "murderer",
                    "suicide", "suicides",
                    "death", "died", "die", "dead"]:
            dic_text_field["DEATH_RELATED_KEYWORD_NUM"] += 1


        #else:
        #    dic_text_field["__POS_" + token[1] + "__"] += 1
            
    #for word in words:
        if word.lower() in STOPWORDS:
            # ignore stopwords
            continue

        digits = ['^(one)$', '^(two)$', '^(three)$', '^(four)$', '^(five)$', '^(six)$', '^(seven)$', '^(eight)$', '^(nine)$']
        for k, digit in enumerate(digits):
            word = re.sub(digit, str(k), word.lower())


        word = ps.stem(word)
        word = re.sub('^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)$', '__NUMBERING__', word.lower())

        #word = word.lower()
        m = re.match("([0-9]+)x([0-9]+)", word)
        if m:
            print(word, m.group(1), m.group(2))
            # dic_text_field["NUM__season"] += int(m.group(1))
            #dic_text_field["NUM__epiode"] = int(m.group(2))

            if int(m.group(2)) >= 10:
                dic_text_field["NUM__episode"] = 1
            else:
                dic_text_field["NUM__episode"] = 0
            dic_text_field["season__digit_"] += 1
            dic_text_field["episod__digit_"] += 1
            continue
        else:
            if re.match("^([0-9]+)$", word):
                word = "_DIGIT_"
            """
            Unigram features, skip if word embedding feature is here
            """
            if not args.exclude_unigrams:
                dic_text_field[word] += 1

            #if word in ["kill", "die", "end", "final", "ultim", "reveal", "eventu", "later"]:
            #    dic_text_field[word] += 1

            #if word in ["often"]:
            #    dic_text_field[word] += 1

            #dic_text_field[word + "_GENRE=" + genre] = 1
            if not args.exclude_unigrams and not args.exclude_tropes:
                dic_text_field["TROPE_" + x[kTROPE_FIELD] + "_" + word] = 1

        processed_words.append(word)

    if not args.exclude_bigrams:
        for bigram in zip(processed_words, processed_words[1:]):
            if bigram[0].lower() in STOPWORDS:
                continue

            elif bigram[1].lower() in ["season", "episod"] \
                    and bigram[0] in ["__NUMBERING__", "final", "last", "following", "next", "later"]:
                # print(bigram) # seems to be a lot!
                dic_text_field[bigram[1].lower() + "__digit_"] += 1

                if bigram[0] in ["final", "last", "following", "next", "later"]:
                    dic_text_field["__FUTURE__EPISODE/SEASON__"] += 1

            # STEMMED!
            elif bigram[0].lower() in ["episod", "season"] and re.match("^([0-9]+)$", bigram[1]):

                if int(bigram[1]) >= 10:
                    dic_text_field["NUM__episode"] = 1
                else:
                    dic_text_field["NUM__episode"] = 0
                # dic_text_field["NUM__"+ bigram[0].lower()] += int(bigram[1])
                dic_text_field[bigram[0].lower() + "__digit_"] += 1
                print(bigram)
            else:
                dic_text_field["_".join(bigram).lower()] += 1
                # pass
    X.append(dic_text_field)

def w2v_feature(dic_text_field, word):
    """
        Word embedding features
        """
    if word in w2v_model.vocab:
        for w2v_dim, elem in enumerate(w2v_model[word]):
            dic_text_field["w2v_dim_" + str(w2v_dim)] = elem


def segment_tropes(x):
    trope_words = []
    trope_word = ""
    for i, char in enumerate(x[kTROPE_FIELD]):
        if char.isupper() and i != 0:
            trope_words.append(trope_word)
            # dic_text_field["TROPE=" + trope_word] = 1
            trope_word = ""
        trope_word += char


def extract_feature_from_data(train):
    X_extracted = []
    for x in train:
        extract_feature(X_extracted, x, [])
    return X_extracted


def run_stanford_ner(train):
    # train_sentences = [x[kTEXT_FIELD] for x in train]
    train_sentences = [preprocess_sentence(x[kTEXT_FIELD]) for x in train]
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in train_sentences]
    stanford_tokens = st.tag_sents(tokenized_sentences)
    return stanford_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument('--stanford', action='store_true', default=False, required=False,
                        help="Use Stanford NER as the NE tagger.")
    parser.add_argument('--ner', action='store_true', default=False, required=False,
                        help="Use NLTK NER as the NE tagger (trained on the ACE corpus).")
    parser.add_argument('--exclude_unigrams', action='store_true', default=False, required=False,
                        help="Do not use unigrams as a feature.")
    parser.add_argument('--exclude_bigrams', action='store_true', default=False, required=False,
                        help="Do not use bigrams as a feature.")
    parser.add_argument('--exclude_tropes', action='store_true', default=False, required=False,
                        help="Do not use tropes as a feature.")
    parser.add_argument('--exclude_genres', action='store_true', default=False, required=False,
                        help="Do not use tropes as a feature.")
 
    args = parser.parse_args()

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    val = [] # empty list by default

    feat = Featurizer()

    labels = []

    if args.limit > 0:
        print("Data limit: %i out of %i" % (args.limit, len(train)))
        val = train[args.limit:]
        train = train[:args.limit]
    else:
        args.limit = len(train)

    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))

    # Preprocessing using Stanford NER
    if args.stanford:
        stanford_tokens_train = run_stanford_ner(train)
        stanford_tokens_test = run_stanford_ner(test)
        stanford_tokens_val = run_stanford_ner(val)

    X_train = []
    X_test = []
    X_val = []
    if args.stanford:
        for x, stanford_token in zip(train, stanford_tokens_train):
            extract_feature(X_train, x, stanford_token)
        for x, stanford_token in zip(test, stanford_tokens_test):
            extract_feature(X_test, x, stanford_token)

        if val:
            for x, stanford_token in zip(val, stanford_tokens_val):
                extract_feature(X_val, x, stanford_token)
    else:
        X_train = extract_feature_from_data(train)
        X_test = extract_feature_from_data(test)
        if val:
            X_val = extract_feature_from_data(val)

    x_train = feat.train_feature(X_train)
    x_test = feat.test_feature(X_test)
    if val:
        x_val = feat.test_feature(X_val)

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))
    y_val = array(list(labels.index(x[kTARGET_FIELD])
                         for x in val))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)

    if val:
        predictions = lr.predict(x_val)
        print(np.logical_xor(predictions, y_val))
        print(np.sum(np.logical_xor(predictions, y_val))) # misclassfied
        val_accuracy = (1.0*len(predictions) - np.sum(np.logical_xor(predictions, y_val)))/ len(predictions)
        print("validation set accuracy: %s" % val_accuracy) 
        o = DictWriter(open("predictions_val.csv", 'w'), ["spoiler", "gold label", "text"])
        o.writeheader()
        for x, pp, gg in zip([text for text in val], predictions, y_val):
            d = {'spoiler': labels[pp], 'gold label': bool(gg), 'text': x}
            o.writerow(d)

    # For LIME
    #c = make_pipeline(feat.vectorizer, lr)
    #class_names = ['Spoiler', 'Not Spoiler']
    #explainer = LimeTextExplainer(class_names=class_names)
    #sample_sentence = val[1929][kTEXT_FIELD]
    #print(sample_sentence)
    #X_sample = []
    #extract_feature(X_sample, val[1929], [])
    ## exp = explainer.explain_instance(X_sample[0], c.predict_proba, num_features=6)
    ## print(exp.as_list())
    #kill_indice = lr.coef_[0].index("kill")

    #print("weight: %s" % feat.vectorizer.feature_names[kill_indice])
