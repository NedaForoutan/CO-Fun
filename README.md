# CO-Fun
CO-Fun: A German Dataset on Company Outsourcing in Fund Prospectuses for Named Entity Recognition and Relation Extraction

The process of cyber mapping gives insights in relationships among financial entities and service providers. Centered around the outsourcing practices of companies within fund prospectuses in Germany, we introduce a dataset specifically designed for named entity recognition and relation extraction tasks. The labeling process on 948 sentences was carried out by three experts which yields to 5,969 annotations for four entity types (Outsourcing, Company, Location and Software) and 4,102 relation annotations (Outsourcing–Company, Company–Location). Furthermore, state-of-the-art deep learning models were trained on this dataset to recognize entities and extract relations. This repository is the anonymized version of the dataset, along with guidelines and the code used for model training. Additionally, this dataset is also hosted on [Zenodo](https://zenodo.org/records/12745116). 

In the following the content of each file is explained:
- CO-Fun-1.0-anonymized.jsonl file contains the raw data of CO-Fun consists of records formatted in JSON. Each entry has the annotated text which is present in form of HTML.  The annotation for each named entity in the text are specified with span tags. Below you can find an exmple of an entry in raw data:

{
  "datetime": "2023-05-04T14:15:54.501875783",
  "entities": [
    {
      "color": "rgb(255, 0, 0)",
      "text": "Ermittlung der tÃ¤glichen und jÃ¤hrlichen Steuerdaten",
      "id": "255c1d4a-d9b0-4fff-8779-6a68f803ce51",
      "type": "Auslagerung"
    },
    {
      "color": "rgb(0, 0, 255)",
      "text": "tba - the beauty aside GmbH",
      "id": "fad78727-1645-4b39-9478-daecb3b4bd2b",
      "type": "Unternehmen"
    }
  ],
  "text": "<html><head></head><body>• Die <span id=\"255c1d4a-d9b0-4fff-8779-6a68f803ce51\" type=\"Auslagerung\" class=\"annotation\" style=\"color: #ff0000\">Ermittlung der tÃ¤glichen und jÃ¤hrlichen Steuerdaten</span> fÃ¼r die Fonds wurde auf die <span id=\"fad78727-1645-4b39-9478-daecb3b4bd2b\" type=\"Unternehmen\" class=\"annotation\" style=\"color: #0000ff\">tba - the beauty aside GmbH</span> ausgelagert.</body></html>",
  "relations": [
    {
      "src": {: "rgb(255, 0, 0)",
        "text": "Ermittlung der tÃ¤glichen und jÃ¤hrlichen Steuerdaten",
        "id": "255c1d4a-d9b0-4fff-8779-6a68f803ce51",
        "type": "Auslagerung"
      },
        "color"
      "trg": {
        "color": "rgb(0, 0, 255)",
        "text": "tba - the beauty aside GmbH",
        "id": "fad78727-1645-4b39-9478-daecb3b4bd2b",
        "type": "Unternehmen"
      },
      "type": "Auslagerung-Unternehmen"
    }
  ]
}

 

- CO-Fun_Annotation-Guideline-EN.pdf is a graphical user interface in German to annotate a sentence with named entities and
relations.

- The prepared-data-and-code folder consists of datasets and python code files for Named Entity Recognition (NER) and Relation Extraction tasks. The training, development and test sets in text format for the CRF model, as well as in text and SpaCy formats for the BERT and RoBERTa models.
