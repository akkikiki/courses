from csv import DictReader, DictWriter

import argparse
import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

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


class Featurizer:
    def __init__(self):
        #self.vectorizer = CountVectorizer()
        #self.bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % ", ".join(feature_names[top10]))
            print("Neg: %s" % ", ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

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

    text_field = []
    for x in train:
        #print(x[kTEXT_FIELD])
        text_field.append(x[kTEXT_FIELD])
    print(text_field)

    x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
    if val:
        x_val = feat.test_feature(x[kTEXT_FIELD] for x in val)

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
        o = DictWriter(open("predictions_val.csv", 'w'), ["id", "spoiler", "gold label"])
        o.writeheader()
        for ii, pp, gg in zip([x['id'] for x in test], predictions, y_val):
            d = {'id': ii, 'spoiler': labels[pp], 'gold label': bool(gg)}
            o.writerow(d)
