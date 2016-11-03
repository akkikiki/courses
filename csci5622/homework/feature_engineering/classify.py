from csv import DictReader, DictWriter

import re
import argparse
import numpy as np
from numpy import array
from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import nltk
ps = nltk.stem.porter.PorterStemmer()

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
#TROPE_LIST = ['ChekhovsGun', 'AnyoneCanDie', 'WhamEpisode', 'Foreshadowing']
TROPE_LIST = ['WhamEpisode', 'Foreshadowing']
STOPWORDS = ['a', 'to', 'and', 'is', 'has', 'had', 'it', 'as', 'does', 'also']

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
    #f = open("genres.list")
    for line in f:
        columns = line.strip().split()
        #print(columns)
        if line.startswith('"') :
            if columns[-1][-1] == ")":
                columns = columns[:-1] # skipping "(DVD)"
            if re.match("^([0-9]+)$", columns[-1]):
                title = re.findall('".*"', line)[0].replace('"', '').replace(" ", "")
                #print(title.lower(), columns[-1])
                dic_geners[title.lower()] += int(columns[-1])
    return dic_geners
 

dic_genres = read_genre_list()
dic_runningtime = read_running_list()

class Featurizer:
    def __init__(self):
        #self.vectorizer = CountVectorizer()
        #self.bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        #self.vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        self.vectorizer = DictVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-20:]
            bottom10 = np.argsort(classifier.coef_[0])[:20]
            print("Pos: %s" % ", ".join(feature_names[top10]))
            print("Neg: %s" % ", ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-20:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))


def extract_feature(X):
    dic_text_field = defaultdict(int)
    #sentence = x[kTEXT_FIELD].lower()
    movie_title = x[kPAGE_FIELD]
    #if not movie_title.lower() in dic_genres:
    #    print("%s not found" % movie_title.lower())
    genres = dic_genres.get(movie_title.lower(), ["NONE"])
    runningtime = dic_runningtime.get(movie_title.lower(), 1)
    
    for genre in genres:
        dic_text_field["GENRE=" + genre] = 1
        dic_text_field[x[kTROPE_FIELD] + "_GENRE=" + genre] = 1
    #dic_text_field["RUNNING_TIME="] = np.log(runningtime)

    sentence = x[kTEXT_FIELD]
    sentence = sentence.replace(',',' ')
    sentence = sentence.replace('(',' ')
    sentence = sentence.replace(')',' ')
    sentence = sentence.replace('"',' ')
    sentence = sentence.replace(':',' ')
    sentence = sentence.replace(';',' ')
    sentence = sentence.replace('*',' ')
    #sentence = sentence.replace('...','')
    #sentence = sentence.replace('.',' .')
    #sentence = sentence.replace('.',' . ')
    sentence = sentence.replace('.','')
    sentence = sentence.replace('-',' - ')
    sentence = sentence.replace('/',' ')
    sentence = sentence.replace('!',' ')
    sentence = sentence.replace('?',' ? ')
    sentence = sentence.strip()
    #print(sentence)
    #sentence = x[kTEXT_FIELD].replace('.','')
    #if sentence[-1] == ".":
    #    sentence = sentence[-1]
    #sentence = x[kTEXT_FIELD]
    #sentence = re.sub('([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen)', '__DIGIT__', sentence)

    words = sentence.split()
    #for stopword in STOPWORDS:
    #    # remove stopwords
    #    """
    #    http://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-python-list
    #    for Python 2
    #    """
    #    #words = filter(lambda word: word != stopword, words)
    #    """
    #    for Python 3
    #    """
    #    words = list(filter((stopword).__ne__, words))
    #print(words)
    processed_words = []
    for word in words:
        if word.lower() in STOPWORDS:
            # ignore stopwords
            continue

        if word[0].isupper() and not word[-1].isupper():
            # avoid Acronyms
            #dic_text_field["NUM_UPPER_CASE_WORDS"] += 1
        #    # Likely to be a named entity
        #    # Do not add as unigrams
            processed_words.append(word)
            continue

        word = re.sub('^(one)$', '1', word.lower())
        word = re.sub('^(two)$', '2', word.lower())
        word = re.sub('^(three)$', '3', word.lower())
        word = re.sub('^(four)$', '4', word.lower())
        word = re.sub('^(five)$', '5', word.lower())
        word = re.sub('^(six)$', '6', word.lower())
        word = re.sub('^(seven)$', '7', word.lower())
        word = re.sub('^(eight)$', '8', word.lower())
        word = re.sub('^(nine)$', '9', word.lower())
        #word = re.sub('^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)$', '__NUMBERING__', word)

        word = ps.stem(word)
        #word = word.lower()
        m = re.match("([0-9]+)x([0-9]+)", word)
        if m:
            print(word, m.group(1), m.group(2))
            dic_text_field["NUM__season"] += int(m.group(1))
            #dic_text_field["NUM__episode"] = min(int(m.group(2)), 10)
            dic_text_field["NUM__episode"] += int(m.group(2))
            dic_text_field["season_digit"] += 1
            dic_text_field["episode_digit"] += 1
            continue
        else:
            if re.match("^([0-9]+)$", word):
                word = "DIGIT"
            dic_text_field[word] += 1#.0/len(words)
            dic_text_field["TROPE_" + x[kTROPE_FIELD] + "_" + word] = 1

        dic_text_field["TROPE_" + x[kTROPE_FIELD]] = 1
        #dic_text_field["TROPE_" + x[kTROPE_FIELD]] = 1

        #dic_text_field["TROPE_" + x[kTROPE_FIELD] + "_" + word] = 1
        #if x[kTROPE_FIELD] in TROPE_LIST:
        #    dic_text_field["TROPE_" + x[kTROPE_FIELD] + "_" + word] = 1
        processed_words.append(word)

    #print(processed_words)
    num_bigrams = len([zip(processed_words, processed_words[1:])])
    for bigram in zip(processed_words, processed_words[1:]):
        if bigram[0].lower() in STOPWORDS:
            continue
        #dic_text_field["_".join(bigram).lower()] += 1
        #dic_text_field["_".join(bigram).lower()] += 1

        #if bigram[0].lower() in ["season", "episode", "seri", "day"] and re.match("^([0-9]+)$", bigram[1]):
        elif bigram[0].lower() in ["season", "episode"] and re.match("^([0-9]+)$", bigram[1]):
            #dic_text_field["NUM__"+ bigram[0].lower()] = min(int(bigram[1]), 10)

            #if bigram[0].lower() == "episode":
            dic_text_field["NUM__"+ bigram[0].lower()] += int(bigram[1])
            dic_text_field[bigram[0].lower() + "_digit"] += 1#.0/num_bigrams
            print(bigram)
        else:
            dic_text_field["_".join(bigram).lower()] += 1#.0/num_bigrams
        # print("_".join(bigram))
    #for trigram in zip(processed_words, processed_words[1:], processed_words[2:]):
    #    if trigram[0].lower() in STOPWORDS:
    #        continue
    #    dic_text_field["_".join(trigram)] += 1
    #dic_text_field["VERB_" + x[kVERB_FIELD]] = 1
    #dic_text_field["PAGE_" + x[kPAGE_FIELD]] = 1

    #dic_text_field["TROPE_" + x[kTROPE_FIELD]] = 1
    X.append(dic_text_field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
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

    X_train = []
    X_test = []
    X_val = []
    for x in train:
        extract_feature(X_train)
    for x in test:
        extract_feature(X_test)


    if val:
        for x in val:
            extract_feature(X_val)

    #x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
    x_train = feat.train_feature(X_train)
    #x_test = feat.train_feature(x[kTEXT_FIELD] for x in test)
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
        o = DictWriter(open("predictions_val.csv", 'w'), ["id", "spoiler", "gold label", "text"])
        o.writeheader()
        for x, ii, pp, gg in zip([text for text in test], [x['id'] for x in test], predictions, y_val):
            d = {'id': ii, 'spoiler': labels[pp], 'gold label': bool(gg), 'text': x}
            o.writerow(d)
