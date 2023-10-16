import os
import sys
import pandas as pd
import re
import numpy as np
import random
import csv
import json
import datetime
import joblib
from bs4 import BeautifulSoup

import spacy
import sklearn
import scipy.stats as stats
import sklearn_crfsuite.metrics
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.preprocessing import LabelBinarizer
import sklearn_crfsuite as crfsuite
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics 
from sklearn.model_selection import RandomizedSearchCV

# Function to remove span tags while preserving their text content
def remove_span_tags(tag):
    if tag.name == 'span':
        return tag.text
    result = []
    for child in tag.contents:
        if isinstance(child, str):
            result.append(child)
        else:
            result.append(remove_span_tags(child))
    return ''.join(result)

def json_to_relations(output_data):
    texts, annotations, relations = [], [], []
    for dict_ in output_data:
        soup = BeautifulSoup(dict_['text'], 'html.parser')
        pure_text = remove_span_tags(soup.body)
        texts.append(pure_text)
        A = []
        B = []
        for annot in dict_['entities']:
            x = annot['text']
            y = annot['type']
            A.append([x,y])
        for rel in dict_['relations']:
            x = rel['src']['text']
            y = rel['trg']['text']
            B.append((x,y))
        annotations.append(A)
        relations.append(B)

    return texts, annotations, relations


def tokens_tags(tokenized_sentences, annotations):
    nlp = spacy.load('de_core_news_sm')
    tags = []
    sen_tokens_info = []
    tokens = [[token for token in tokens_sen] for tokens_sen in tokenized_sentences]
    
    for i, tokens_sen in enumerate(tokens):
        temp_tags = ['O' for _ in range(len(tokens_sen))]
        annots = annotations[i]     #of sentence ith
        annot_terms = []
        annot_labels = []
        tokens_info = []
        for a, annot in enumerate(annots):
            tokens_annot_term = nlp(annot[0])
            for t, token_annot in enumerate(tokens_annot_term):
                annot_terms.append(str(token_annot.text))
                if t == 0:
                    annot_labels.append('B-' + annot[1])
                else:
                    annot_labels.append('I-' + annot[1])
                     
        for t, token in enumerate(tokens_sen):
            if str(token.text) in annot_terms:
                index = annot_terms.index(str(token.text))
                temp_tags[t] = annot_labels[index]

        for j, token in enumerate(tokens_sen):
            tokens_info.append((str(token.text), token.pos_, temp_tags[j]))
               
        tags.append(temp_tags)
        sen_tokens_info.append(tokens_info)
        
    return tokens, tags, sen_tokens_info


def word2features(sent, i):#, word2cluster):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1
        ])
    else:
        features.append('BOS')

    if i > 1: 
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.extend([
            '-2:word.lower=' + word2.lower(),
            '-2:word.istitle=%s' % word2.istitle(),
            '-2:word.isupper=%s' % word2.isupper(),
            '-2:postag=' + postag2
        ])        

        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1
        ])
    else:
        features.append('EOS')

    if i < len(sent)-2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.extend([
            '+2:word.lower=' + word2.lower(),
            '+2:word.istitle=%s' % word2.istitle(),
            '+2:word.isupper=%s' % word2.isupper(),
            '+2:postag=' + postag2
        ])

        
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))] 

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


def main(path_dir,file_in):
    #file_in : sentences_annotated.json 
    if os.path.exists(path_dir):
        os.chdir(path_dir) #set the current working directory

        #reading the json file
        if os.path.getsize(file_in) > 0:
            with open(file_in, 'r', encoding='utf-8') as f:
                #load the annotations file into the dictionary
                lines = f.readlines()
                output_data = []
                for line in lines:
                    output_dict = json.loads(line)
                    output_data.append(output_dict)
        
            #convert the json to relations list
            sentences, annotations, relations = json_to_relations(output_data)
            #load the German language model for spaCy
            nlp = spacy.load('de_core_news_sm')
            #tokenizing the sentence 
            tokenized_sentences = [nlp(sentence) for sentence in sentences]
            tokens, tags, data_sets = tokens_tags(tokenized_sentences, annotations)

            train_size = int(0.8 * len(sentences))
            valid_size = int(0.1 * len(sentences))

            train_indices = random.sample(range(len(sentences)), train_size)
            '''with open('train_indices.txt', 'r') as file:
                # Read the lines and convert them to integers
                train_indices = [int(line.strip()) for line in file]
            '''    
            train_sents = [data_sets[i] for i in train_indices]

            valid_test_indices = [i for i in range(len(sentences)) if i not in train_indices]
            valid_indices = random.sample(valid_test_indices, valid_size)
            '''with open('valid_indices.txt', 'r') as file:
                valid_indices = [int(line.strip()) for line in file]
            '''
            dev_sents = [data_sets[i] for i in valid_indices]
            
            exclude_set = set(train_indices + valid_indices)
            test_indices = [i for i in range(len(sentences)) if i not in exclude_set]
            '''with open('test_indices.txt', 'r') as file:
                test_indices = [int(line.strip()) for line in file]
            '''    
            test_sents = [data_sets[i] for i in test_indices]
            
            # Define the label mapping
            id2label = {0: "O", 1: "B-Unternehmen", 2: "I-Unternehmen", 3: "B-Auslagerung", 4: "I-Auslagerung", 5: "B-Ort", 6: "I-Ort", 7: "B-Software", 8: "I-Software"}
            label2id = {"O": 0, "B-Unternehmen": 1, "I-Unternehmen": 2, "B-Auslagerung": 3, "I-Auslagerung": 4, "B-Ort": 5, "I-Ort": 6, "B-Software": 7, "I-Software": 8}


            X_train = [sent2features(s) for s in train_sents] 
            y_train = [sent2labels(s) for s in train_sents]

            X_dev = [sent2features(s) for s in dev_sents] 
            y_dev = [sent2labels(s) for s in dev_sents]

            X_test = [sent2features(s) for s in test_sents]
            y_test = [sent2labels(s) for s in test_sents]

            #CRF
            crf = crfsuite.CRF(
                verbose='true',
                algorithm='lbfgs',
                max_iterations=100
            )

            crf.fit(X_train, y_train, X_dev = X_dev, y_dev = y_dev)

            #Evaluation
            y_pred = crf.predict(X_test)
            
            example_sent = test_sents[0]
            print(example_sent)
            print("\nSentence:", ' '.join(sent2tokens(example_sent)))
            print("\nPredicted:", ' '.join(crf.predict([sent2features(example_sent)])[0]))
            print("\nCorrect:  ", ' '.join(sent2labels(example_sent)))

            labels = list(crf.classes_)
            labels.remove("O")
            y_pred = crf.predict(X_test)
            sorted_labels = sorted(
                labels,
                key=lambda name: (name[1:], name[0])
            )

            print("Train prediction:\n", metrics.flat_classification_report(y_train, crf.predict(X_train), labels=sorted_labels, digits=3))
            print("Test prediction:\n", metrics.flat_classification_report(y_test, crf.predict(X_test), labels=sorted_labels, digits=3))

            x=j
            #Finding the optimal hyperparameters            
            crf = CRF(
                algorithm='lbfgs',
                max_iterations=100,
                all_possible_transitions=True
            )

            params_space = {
                'c1': stats.expon(scale=0.5),
                'c2': stats.expon(scale=0.05),
            }

            f1_scorer = make_scorer(metrics.flat_f1_score,
                                    average='weighted', labels=labels)

            rs = RandomizedSearchCV(crf, params_space,
                                    cv=3,
                                    verbose=1,
                                    n_jobs=-1,
                                    n_iter=50,
                                    scoring=f1_scorer)
            rs.fit(X_train, y_train)
            best_crf = rs.best_estimator_

            print('best params:', rs.best_params_)
            print('best CV score:', rs.best_score_)
            print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

            print("\nTrain prediction:\n")
            train_prediction = best_crf.predict(X_train)
            print(metrics.flat_classification_report(
                y_train, train_prediction, labels=sorted_labels, digits=3))

            print("\nTest prediction:\n")
            test_prediction = best_crf.predict(X_test)
            print(metrics.flat_classification_report(
                y_test, test_prediction, labels=sorted_labels, digits=3))

            

            OUTPUT_PATH = path_dir
            OUTPUT_FILE = "best_crf_model"

            if not os.path.exists(OUTPUT_PATH):
                os.mkdir(OUTPUT_PATH)

            joblib.dump(best_crf, os.path.join(OUTPUT_PATH, OUTPUT_FILE))
                        
        else:
            print("empty ", file_in)
            sys.exit(1)
           
    else:
            print("Working directory does not exist")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No Argument!")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])


