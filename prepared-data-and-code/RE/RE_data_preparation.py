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
import spacy
from bs4 import BeautifulSoup

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
    texts, annotations, span_texts, relations = [], [], [], []
    for dict_ in output_data:
        soup = BeautifulSoup(dict_['text'], 'html.parser')
        pure_text = remove_span_tags(soup.body)
        texts.append(pure_text)
        A = []
        B = []
        for annot in dict_['entities']:
            x = annot['text']
            y = annot['type']
            z = annot['id']
            A.append([x,y,z])
        span_texts.append(dict_['text'])
        for rel in dict_['relations']:
            '''x = rel['src']
            y = rel['trg']
            B.append([x,y])'''
            B.append(rel)
        annotations.append(A)
        relations.append(B)

    return texts, annotations, span_texts, relations

#remove nested tags
def correct_spans(span_text):
    span_text = span_text.replace('\"','"')
    span_text = span_text.replace('\/','/')

    soup = BeautifulSoup(span_text, 'html.parser')
    def merge_spans(span):
        for child in span.find_all('span'):
            child.unwrap()
        return span
    
    for span_tag in soup.find_all('span'):
        span_tag = merge_spans(span_tag)
    

    corrected_span_text = str(soup)
 
    return corrected_span_text

def list_tokens_relations(span_text, src_trg):
    nlp = spacy.load('de_core_news_sm')
    span_text = span_text.replace("<\/span>", "</span>" )
    pattern = r'<span[^>]*>.*?</span>'
    
    tokens = []
    entities_info = []
    
    for span in re.finditer(pattern, span_text):
        span_ = span_text[span.start():span.end()]
        soup = BeautifulSoup(span_, "html.parser" )
        text = soup.span.string #"Compliance"
        text = str(text).strip()
        entityLabel = soup.span['type'] #"Auslagerung"
        entity_id = soup.span['id']
        
        pre_spantext = span_text[:span.start()]
        soup = BeautifulSoup(pre_spantext, "html.parser" )
        pre_text = soup.get_text()
        start = len(pre_text) #0
        end = start + len(str(text)) #8

        token_start = len(nlp(pre_text)) #0
        token_end = token_start + len(nlp(str(text))) - 1 #0

        entities_info.append(dict(entity_id = entity_id, text = text, token_start = token_start, entityLabel = entityLabel))
        temp_tok = dict(text = text, start = start, end = end, token_start = token_start, token_end = token_end, entityLabel = entityLabel)
        tokens.append(temp_tok)

    relations = []
    for rel in src_trg:
        if rel['src']['id'] != rel['trg']['id']:
            id_src = rel['src']['id']
            for dic in entities_info:
                if dic['entity_id'] == id_src:
                    if rel['src']['type'] == 'Auslagerung':
                        head = dic['token_start'] #0
                        relationLabel = "Auslagerung_Unternehmen"
                    elif rel['src']['type'] == 'Unternehmen' and rel['trg']['type'] == 'Ort':
                        head = dic['token_start'] #0
                        relationLabel = "Unternehmen_Ort"
                    elif rel['src']['type'] == 'Unternehmen' and rel['trg']['type'] == 'Auslagerung':
                        child = dic['token_start'] #8
                        relationLabel = "Auslagerung_Unternehmen"
                    elif rel['src']['type'] == 'ort':
                        child = dic['token_start']
                        relationLabel = "Unternehmen_Ort"
                    break
            
            id_trg = rel['trg']['id']
            for dic in entities_info:
                if dic['entity_id'] == id_trg:
                    if rel['trg']['type'] == 'Unternehmen' and rel['src']['type'] == 'Auslagerung':
                        child = dic['token_start'] #8
                    elif rel['trg']['type'] == 'Unternehmen' and rel['src']['type'] == 'Ort':
                        head = dic['tok-en_start']
                    elif rel['trg']['type'] == 'Ort':
                        child = dic['token_start'] #8
                    elif rel['trg']['type'] == 'Auslagerung':
                        head = dic['token_start'] #0
                    break

            temp_rel = dict(child = child, head = head, relationLabel = relationLabel)
            relations.append(temp_rel)

    return tokens, relations

def relations_doc(sentences, annotations, span_texts, relations):
    nlp = spacy.load('de_core_news_sm')
    tokenized_sentences = [nlp(sentence) for sentence in sentences]
    tokens = [[token for token in tokens_sen] for tokens_sen in tokenized_sentences]
    
    doc = []
    for i, sen in enumerate(sentences):
        tokens_sen, relations_sen = list_tokens_relations(span_texts[i], relations[i])            
        temp_dict = dict(document= sen, tokens= tokens_sen, relations= relations_sen)
        doc.append(temp_dict)
    return doc


#write the list of datasets to their coresponding file
def write_list_to_file(file_name, data_list):
    try:
        # Open the file in write mode ('w')
        with open(file_name, 'w', encoding='utf-8') as file:
            # Write the list as a string to the file
            file.write(str(data_list))
        print("List data has been written to the file successfully.")
    except IOError:
        print(f"An error occurred while writing to the file: {file_path}")



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
            sentences, annotations, span_sens, relations = json_to_relations(output_data)
            
            #load the German language model for spaCy
            nlp = spacy.load('de_core_news_sm')
            #tokenizing the sentence 
            tokenized_sentences = [nlp(sentence) for sentence in sentences]
            tokens = [[token for token in tokens_sen] for tokens_sen in tokenized_sentences]
            #correcting the span texts
            span_sens = [correct_spans(span_text) for span_text in span_sens]

            train_size = int(0.8 * len(sentences))
            valid_size = int(0.1 * len(sentences))

            # Train
            train_indices = random.sample(range(len(sentences)), train_size)
            train_sens = [sentences[i] for i in train_indices]
            train_annotations = [annotations[i] for i in train_indices]
            train_span_sens = [span_sens[i] for i in train_indices]
            train_relations = [relations[i] for i in train_indices]
            
            train_doc = relations_doc(train_sens, train_annotations, train_span_sens, train_relations)
            train_doc_quot = json.dumps(train_doc)
            file_name = "relations_training.txt"
            # Call the function to write the list to the file
            write_list_to_file(file_name, train_doc_quot)
            
            # Valid
            valid_test_indices = [i for i in range(len(sentences)) if i not in train_indices]
            valid_indices = random.sample(valid_test_indices, valid_size)
            valid_sens = [sentences[i] for i in valid_indices]
            valid_annotations = [annotations[i] for i in valid_indices]
            valid_span_sens = [span_sens[i] for i in valid_indices]
            valid_relations = [relations[i] for i in valid_indices]

            valid_doc = relations_doc(valid_sens, valid_annotations, valid_span_sens, valid_relations)
            valid_doc_quot = json.dumps(valid_doc)
            file_name = "relations_dev.txt"
            # Call the function to write the list to the file
            write_list_to_file(file_name, valid_doc_quot)
            
            # Test
            exclude_set = set(train_indices + valid_indices)
            test_indices = [i for i in range(len(sentences)) if i not in exclude_set]
            test_sens = [sentences[i] for i in test_indices]
            test_annotations = [annotations[i] for i in test_indices]
            test_span_sens = [span_sens[i] for i in test_indices]
            test_relations = [relations[i] for i in test_indices]

            test_doc = relations_doc(test_sens, test_annotations, test_span_sens, test_relations)
            test_doc_quot = json.dumps(test_doc)
            file_name = "relations_test.txt"
            # Call the function to write the list to the file
            write_list_to_file(file_name, test_doc_quot)         
            

            #create binary spacy file for each train, valid and test
            #binary_converter.py
            '''print('whole = ', len(sentences), '\n train = ', len(train_indices), '\nvalid = ', len(valid_indices))
            print('test = ', len(test_indices))'''

                        
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

# sys.argv[1]: path_dir
# sys.argv[2]: json file_in
