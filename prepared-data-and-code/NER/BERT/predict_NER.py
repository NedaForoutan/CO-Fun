import os
import sys
import spacy

def main(path_dir,text_file):
    #file_in : sentence text file  
    if os.path.exists(path_dir):
        os.chdir(path_dir)

        with open('text_file', 'r', encoding='utf-8') as file:
            text = file.read()
        text = 'Die Gesellschaft hat die Funktion der Innenrevision, die Aufgaben der Rechtsabteilung, der Personalabteilung, insbesondere Personalverwaltung, -beschaffung, -betreuung, Aus- und Weiterbildung sowie Arbeitszeiterfassung und die Aufgaben der IT-Abteilung an die M.M.Warburg & CO (AG & Co.) KGaA, Ferdinandstr. 75, 20095 Hamburg ausgelagert.'
        
        nlp_ner = spacy.load("model-best")
        doc = nlp_ner(text)

        '''for doc in nlp.pipe(text,disable=["tagger"]):
            print(f"spans: {[(e.start, e.text, e.label_) for e in doc.ents]}")'''
            
        
        ents = list(doc.ents)
        Entities = []
        #Entities.append({'text' : text})
        for ent in ents:
            ent_dict = (ent.start, ent.text, ent.label_)
            print(ent_dict)
            Entities.append(ent_dict)

        with open('predicted_entities.txt', 'w+') as f:
            for entity in Entities:
                f.write('%s\n' %str(entity))
             
            print("File written successfully")
        
    else:
        print("Working directory does not exist")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No Argument!")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

    

