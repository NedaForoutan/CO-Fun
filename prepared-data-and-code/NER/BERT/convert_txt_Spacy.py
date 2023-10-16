import pandas as pd
import numpy as np
import json

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans



#read text files
with open('relations_training.txt', encoding='utf-8') as jsonfile:
    file = json.load(jsonfile)

training_data = {'classes' : ["AUSLAGERUNG", "UNTERNEHMEN", "ORT", "SOFTWARE"], 'annotations' : []}
for example in file:
    temp_dict = {}
    temp_dict['text'] = example["document"]
    temp_dict['entities'] = []
    for annotation in example['tokens']:
        if annotation["text"] != "None":
            start = annotation["start"]
            end = annotation["end"]
            label = annotation["entityLabel"].upper()
            temp_dict['entities'].append((start, end, label))
    training_data['annotations'].append(temp_dict)

#converting to binary format
nlp = spacy.blank("de") # load a new spacy model
doc_bin = DocBin() # create a DocBin object

for training_example  in tqdm(training_data['annotations']): 
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents 
    doc_bin.add(doc)

doc_bin.to_disk("training_data.spacy") # save the docbin object

                
