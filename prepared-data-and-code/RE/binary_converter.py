import json
import sys

import typer
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer
from spacy.tokenizer import Tokenizer
from spacy.lang.de import German
from spacy.util import compile_infix_regex
import re
import spacy

nlp = spacy.blank("de")
nlp_de = spacy.load("de_core_news_sm")
# Create a blank Tokenizer with just the German vocab

msg = Printer()

SYMM_LABELS = ["Binds"]
MAP_LABELS = {
    "Auslagerung_Unternehmen": "Auslagerung_Unternehmen",
    "Unternehmen_Ort": "Unternehmen_Ort"
}


def main(in_file_name: Path, out_file_name: Path):
    """Creating the corpus from the annotations."""
    Doc.set_extension("rel", default={},force=True)
    vocab = Vocab()

    docs = {"train": [], "dev": [], "test": [], "total": []}
    ids = {"train": set(), "dev": set(), "test": set(), "total":set()}
    count_all = {"train": 0, "dev": 0, "test": 0,"total": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0,"total": 0}

    count = 0
    none_entity_sen = []#the annotation of this sentence has been annotated incompleted so wrong tokens which result in None Entities
    with open(in_file_name, encoding='utf-8') as jsonfile:
        file = json.load(jsonfile)
        for example in file:
            count = count + 1

            flag = True
            span_starts = set()
            neg = 0
            pos = 0
            # Parse the tokens
            tokens=nlp(example["document"])

            spaces=[]
            spaces = [True if tok.whitespace_ else False for tok in tokens]
            words = [t.text for t in tokens]
            doc = Doc(nlp.vocab, words=words, spaces=spaces)


            # Parse the GGP entities
            spans = example["tokens"]
            entities = []
            span_end_to_start = {}
            
            for span in spans:
                entity = doc.char_span(
                     span["start"], span["end"], label=span["entityLabel"]
                 )
                if entity == None:
                    none_entity_sen.append(example)
                    flag = False
                    break
                else:
                    span_end_to_start[span["token_start"]] = span["token_start"]
                    entities.append(entity)
                    span_starts.add(span["token_start"])

            if flag == True:
                doc.ents = entities

                # Parse the relations
                rels = {}
                for x1 in span_starts:
                    for x2 in span_starts:
                        rels[(x1, x2)] = {}
                relations = example["relations"]
                
                for relation in relations:
                    # the 'head' and 'child' annotations refer to the end token in the span
                    # but we want the first token
                    start = span_end_to_start[relation["head"]]
                    end = span_end_to_start[relation["child"]]
                    label = relation["relationLabel"]
                    
                    if label not in rels[(start, end)]:
                        rels[(start, end)][label] = 1.0
                        pos += 1

                # The annotation is complete, so fill in zero's where the data is missing
                for x1 in span_starts:
                    for x2 in span_starts:
                        for label in MAP_LABELS.values():
                            if label not in rels[(x1, x2)]:
                                neg += 1
                                rels[(x1, x2)][label] = 0.0

                doc._.rel = rels

                # only keeping documents with at least 1 positive case
                if pos > 0:
                        docs["total"].append(doc)
                        count_pos["total"] += pos
                        count_all["total"] += pos + neg

                    
                    
    print('Number of total snetences: ', count)
    docbin = DocBin(docs=docs["total"], store_user_data=True)
    docbin.to_disk(out_file_name)
    msg.info(
        f"{len(docs['total'])} sentences in {out_file_name}"
    )


main(sys.argv[1], sys.argv[2])

#sys.argv[1] : x.txt    eg.   relation_training.txt
#sys.argv[2] : x.spacy  eg.   relations_training.spacy

'''
ann = "path_to/relation_training.txt"
train_file='path_to/relations_training.spacy'
dev_file='path_to/relations_dev.spacy'
test_file='path_to/relations_test.spacy'
'''
