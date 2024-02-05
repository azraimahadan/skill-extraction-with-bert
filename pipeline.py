import pandas as pd
import numpy as np
import torch
import zipfile
import os

from transformers import BertTokenizer, BertForSequenceClassification
import contractions
import re 
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize




# Load pre-trained BERT model and tokenizer
# def load_model():
#     model_name = "./bert_fine_tuned/bert_fine_tuned" 
#     tokenizer = BertTokenizer.from_pretrained('./bert_fine_tuned/bert_tokens')
#     model = BertForSequenceClassification.from_pretrained(model_name)
#     return model, tokenizer
def load_model():
    model_name = "azrai99/bert-skills-extraction"
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    return model,tokenizer


def clean(desc):
    desc = contractions.fix(desc)
    desc = re.sub("[!@.$\'\'':()]", "", desc)
    return desc

def extract_POS(tagged):
    #pattern 1
    grammar1 = ('''Noun Phrases: {<DT>?<JJ>*<NN|NNS|NNP>+}''')
    chunkParser = nltk.RegexpParser(grammar1)
    tree1 = chunkParser.parse(tagged)

    # typical noun phrase pattern appending to be concatted later
    g1_chunks = []
    for subtree in tree1.subtrees(filter=lambda t: t.label() == 'Noun Phrases'):
        g1_chunks.append(subtree)
    
    #pattern 2
    grammar2 = ('''NP2: {<IN>?<JJ|NN>*<NNS|NN>} ''')
    chunkParser = nltk.RegexpParser(grammar2)
    tree2 = chunkParser.parse(tagged)

    # variation of a noun phrase pattern to be pickled for later analyses
    g2_chunks = []
    for subtree in tree2.subtrees(filter=lambda t: t.label() == 'NP2'):
        g2_chunks.append(subtree)
        
    #pattern 3
    grammar3 = (''' VS: {<VBG|VBZ|VBP|VBD|VB|VBN><NNS|NN>*}''')
    chunkParser = nltk.RegexpParser(grammar3)
    tree3 = chunkParser.parse(tagged)

    # verb-noun pattern appending to be concatted later
    g3_chunks = []
    for subtree in tree3.subtrees(filter=lambda t: t.label() == 'VS'):
        g3_chunks.append(subtree)
        
        
    # pattern 4
    # any number of a singular or plural noun followed by a comma followed by the same noun, noun, noun pattern
    grammar4 = ('''Commas: {<NN|NNS>*<,><NN|NNS>*<,><NN|NNS>*} ''')
    chunkParser = nltk.RegexpParser(grammar4)
    tree4 = chunkParser.parse(tagged)

    # common pattern of listing skills appending to be concatted later
    g4_chunks = []
    for subtree in tree4.subtrees(filter=lambda t: t.label() == 'Commas'):
        g4_chunks.append(subtree)
        
    return g1_chunks, g2_chunks, g3_chunks, g4_chunks

def tokenize_and_tag(desc):
    tokens = nltk.word_tokenize(desc.lower())
    filtered_tokens = [w for w in tokens if not w in stop_words]
    tagged = nltk.pos_tag(filtered_tokens)
    return tagged

def training_set(chunks):
    '''creates a dataframe that easily parsed with the chunks data '''
    df = pd.DataFrame(chunks)    
    df.fillna('X', inplace = True)
    
    train = []
    for row in df.values:
        phrase = ''
        for tup in row:
            # needs a space at the end for seperation
            phrase += tup[0] + ' '
        phrase = ''.join(phrase)
        # could use padding tages but encoder method will provide during 
        # tokenizing/embeddings; X can replace paddding for now
        train.append( phrase.replace('X', '').strip())

    df['phrase'] = train

    return df.phrase

def strip_commas(df):
    '''create new series of individual n-grams'''
    grams = []
    for sen in df:
        sent = sen.split(',')
        for word in sent:
            grams.append(word)
    return pd.Series(grams)

def generate_phrases(desc):
    tagged = tokenize_and_tag(desc)
    g1_chunks, g2_chunks, g3_chunks, g4_chunks = extract_POS(tagged)
    c = training_set(g4_chunks)       
    separated_chunks4 = strip_commas(c)
    phrases = pd.concat([training_set(g1_chunks),
                          training_set(g2_chunks), 
                          training_set(g3_chunks),
                          separated_chunks4], 
                            ignore_index = True )
    return phrases

def get_predictions(desc, model, tokenizer, threshold=0.6, return_probabilities=False):
    # Clean
    desc = clean(desc)
    

    phrases = generate_phrases(desc).tolist()
    phrases = [phrase.strip() for phrase in phrases]
    
    print(phrases)
    
    # Tokenize and prepare phrases for the model
    inputs = tokenizer(phrases, return_tensors="pt", truncation=True, padding=True)
    
    model,tokenizer = load_model()
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Get predicted classes based on the threshold
    predictions = (probs[:, 1] > threshold).to(torch.int32)
    
    # Return predicted skills as a list
    out = pd.DataFrame({'Phrase': phrases, 'Class': predictions})
    skills = out.loc[out['Class'] == 1]
    
    return skills['Phrase'].unique().tolist()

    # # Return predicted skills and probabilities as lists
    # out = pd.DataFrame({'Phrase': phrases, 'Class': predictions, 'Probability': probs[:, 1]})
    # skills = out.loc[out['Class'] == 1]

    # if return_probabilities:
    #     return skills['Phrase'].tolist(), skills['Probability'].tolist()
    # else:
    #     return skills['Phrase'].tolist()

def get_predictions_excel(filename):
    """description column must be titled Job Desc"""
    df = pd.read_csv(filename)
    df['Extracted skills'] = df['Job Description'].apply(lambda x: get_predictions(x))
    
    return df.to_csv('extracted.csv')