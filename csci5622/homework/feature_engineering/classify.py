from csv import DictReader, DictWriter

import re
import argparse
import numpy as np
from numpy import array
from collections import defaultdict

from nltk.tag import StanfordNERTagger
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import nltk
ps = nltk.stem.porter.PorterStemmer()
st = StanfordNERTagger('/Users/Fujinuma/Software/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz',
                       '/Users/Fujinuma/Software/stanford-ner-2014-06-16/stanford-ner.jar',
                       encoding='utf-8')


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


def extract_feature(X, x, stanford_tokens):
    dic_text_field = defaultdict(int)
    movie_title = x[kPAGE_FIELD]
    if "WHAM" in x[kTROPE_FIELD]:
        x[kTROPE_FIELD] = x[kTROPE_FIELD].replace("WHAM", "Wham")
    #if not movie_title.lower() in dic_genres:
    #    print("%s not found" % movie_title.lower())
    genres = dic_genres.get(movie_title.lower(), ["NONE"])
    runningtime = dic_runningtime.get(movie_title.lower(), 1)
    # print(x[kTROPE_FIELD])
    for genre in genres:
        if genre == "NONE":
            continue
        dic_text_field["GENRE=" + genre] = 1
        dic_text_field[x[kTROPE_FIELD] + "_GENRE=" + genre] = 1
    # TODO: Just Split based on Capital Characters
    # trope_words = []
    # trope_word = ""
    # for i, char in enumerate(x[kTROPE_FIELD]):
    #     if char.isupper() and i != 0:
    #         trope_words.append(trope_word)
    #         # dic_text_field["TROPE=" + trope_word] = 1
    #         trope_word = ""
    #     trope_word += char

    # print(trope_words)


    #


    # death_tropes = ["Kill", "Death", "Dead", "Die"]
    # for target_trope in death_tropes:
    #     if target_trope in x[kTROPE_FIELD]:
    #         dic_text_field["TROPE=Death_related_trope"] = 1

    # target_tropes = ["Kill", "Spoiler", "Ending", "Reveal", "Hero", "Die", "Gambit", "Wham"]
    target_tropes = ["Kill", "Spoil", "Gambit", "Wham", "Heroic", "Ending", "Death", "Dead", "Chekhovs"]
    target_tropes = ["Kill","Ending", "Death"]
    for target_trope in target_tropes:
        if target_trope in x[kTROPE_FIELD]:
            dic_text_field["TROPE=" + target_trope] = 1

    # dic_text_field["RUNNING_TIME="] = np.log(runningtime)

    sentence = x[kTEXT_FIELD]
    # episode_title = re.match('"([\w ]+)"', sentence)
    # if episode_title:# and re.match("([0-9]+)x([0-9]+)", sentence):
    #     print(episode_title.group(1))
    punctuations = [',', '(', ')', '"', ':', ';', '*', '/', '!']
    if '"' in sentence:
        dic_text_field["QUOTATION_MARK"] = 1

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
    else:
        tokenized_sentence = nltk.word_tokenize(sentence)
        #tokens = st.tag(tokenized_sentence)
        tagged_sentence = nltk.pos_tag(tokenized_sentence)
        tokens = nltk.ne_chunk(tagged_sentence)
        #words = sentence.split()

    #    """
    #for stopword in STOPWORDS:
    #    http://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-python-list
    #    """

    processed_words = []
    for token in tokens:
        #print(token)
        word = token[0]
        # if word in ["father", "mother"]:
        #     word = "parent"
        # if word in ["husband", "wife"]:
        #     word = "spouse"
        # if word in ["her", "she", "he", "his", "him"]:
        #     word = "PRONOUN"


        """
        Stanford NER
        """
        if args.stanford and token[1] != 'O':
            word = "__" + token[1] + "__"

        """
        NLTK ACE-trained NER
        """

        if hasattr(token, 'label') and token.label:
            #word = "__PERSON__"
            word = "__" + token.label() + "__"

        for target_trope in target_tropes:
            if target_trope in x[kTROPE_FIELD]:
                dic_text_field["TROPE=" + target_trope + "__" + word] = 1

        # for trope_word in trope_words:
        #     dic_text_field["TROPE=" + trope_word + "__" + word] = 1

        if word == "sucide":
            word = "suicide"
            # target_tropes = ["Kill","Ending", "Death"]

        # if word in ["death", "die", "dead"]:
        #     word = "death"
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
            # dic_text_field["NUM__season"] += int(m.group(1))
            #dic_text_field["NUM__epiode"] = min(int(m.group(2)), 10)

            if int(m.group(2)) >= 10:
                dic_text_field["NUM__episode"] = 1
            else:
                dic_text_field["NUM__episode"] = 0
            # dic_text_field["NUM__episode"] += int(m.group(2))
            dic_text_field["season__digit_"] += 1
            dic_text_field["episod__digit_"] += 1
            continue
        else:
            if re.match("^([0-9]+)$", word):
                word = "_DIGIT_"
            dic_text_field[word] += 1#.0/len(words)

            #dic_text_field[word + "_GENRE=" + genre] = 1
            dic_text_field["TROPE_" + x[kTROPE_FIELD] + "_" + word] = 1

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

        #if bigram[0].lower() in ["season", "episode", "seri", "day"] and re.match("^([0-9]+)$", bigram[1]):

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

    #dic_text_field["TROPE_" + x[kTROPE_FIELD]] = 1
    X.append(dic_text_field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument('--stanford', action='store_true', default=False, required=False,
                        help="Use Stanford NER as the NE tagger.")
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

    stanford_tokens_train = []
    stanford_tokens_test = []
    stanford_tokens_val = []
    # Preprocessing using Stanford NER
    if args.stanford:
        train_sentences = [x[kTEXT_FIELD] for x in train]
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in train_sentences]
        stanford_tokens_train = st.tag_sents(tokenized_sentences)
        #print(stanford_tokens_train)
        test_sentences = [x[kTEXT_FIELD] for x in test]
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in test_sentences]
        stanford_tokens_test = st.tag_sents(tokenized_sentences)
        #print(stanford_tokens_test)
        val_sentences = [x[kTEXT_FIELD] for x in val]
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in val_sentences]
        stanford_tokens_val = st.tag_sents(tokenized_sentences)
        #print(stanford_tokens_val)


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
        for x in train:
            extract_feature(X_train, x, [])
        for x in test:
            extract_feature(X_test, x, [])

        if val:
            for x in val:
                extract_feature(X_val, x, [])



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
