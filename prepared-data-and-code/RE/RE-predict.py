import os
import sys
import random
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from rel_pipe import make_relation_extractor, score_relations
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

def main(path_dir,text):
    #file_in : sentences_annotated.json , file_out : next_sentence_output.json 
    if os.path.exists(path_dir):
        os.chdir(path_dir)

        # We load the NER model
        nlp_ner = spacy.load("model-best")
        #text = "• Teile der Interne Revision wurden ausgelagert an die PricewaterhouseCoopers GmbH Wirtschaftsprüfungsgesellschaft, Frankfurt am Main • Portfoliomanagement Das Portfoliomanagement für den Fonds wurde an die Joh."
        doc = nlp_ner(text)

        # We load the relation extraction (REL) model
        nlp2 = spacy.load("training/model-best")

        # We take the entities generated from the NER pipeline and input them to the REL pipeline
        for name, proc in nlp2.pipeline:
                  doc = proc(doc)
        # Here, we split the paragraph into sentences and apply the relation extraction for each pair of entities found in each sentence.
        relations =[]
        relations.append({'text' : text})
        for value, rel_dict in doc._.rel.items():
            for e in doc.ents:
                for b in doc.ents:
                    if e.start == value[0] and b.start == value[1]:
                        if rel_dict['Auslagerung_Unternehmen'] >=0.8 :
                            print(f" entities: {e.text, b.text} --> predicted relation: {rel_dict}")
                            relation_dict ={'entities' : (e.text, b.text), 'predicted_relation' : 'Auslagerung_Unternehmen'}
                            relations.append(relation_dict)
                        if rel_dict['Unternehmen_Ort'] >=0.8 :
                            print(f" entities: {e.text, b.text} --> predicted relation: {rel_dict}")
                            relation_dict ={'entities' : (e.text, b.text), 'predicted_relation' : 'Unternehmen_Ort'}
                            relations.append(relation_dict)


        with open('predicted_relations.txt', 'w+') as f:
            for rel in relations:
                f.write('%s\n' %rel)
             
        print("File written successfully")

    else:
        print("Working directory does not exist")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No Argument!")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
                
