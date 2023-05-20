import time
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pandas as pd
import unidecode
from word2number import w2n
import contractions
import regex as re
import os 
import functions_framework

N=10
@functions_framework.http
def my_http_function(request):
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return main(request.args.get('message'))
    elif request_json and 'message' in request_json:
        return main(request_json['message'])
    else:
        return main("embroidered gown full length")
    
    
def preprocess(text):
    text = unidecode.unidecode(text)    # Remove accents
    text = contractions.fix(text)       # Expand contractions
    text = text.lower()                 # Convert to lowercase
    # re.sub(r'\d+', '', text)          # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.strip()                 # Remove extra whitespaces
    text = " ".join(text.split())       
    return text

model = SentenceTransformer('all-mpnet-base-v2')                                # Loading the Sentence Transformer Model

def main(sample):
    t= time.time()
    ajio_embeddings = pd.read_csv('embeddings.csv').to_numpy()                      # Loading the embeddings from the CSV file
    ajio_embeddings = torch.tensor(ajio_embeddings, dtype=torch.float32)            # Converting the embeddings to a tensor
    ajio = pd.read_csv('dataset.csv')                                               # Loading the Datastore from the CSV file

    sample = preprocess(sample)                                                     # Preprocessing the Search Query
    sample_emb = model.encode(sample, convert_to_tensor=True)                       # Getting the embeddings of the Search Query

    scores = np.array(util.cos_sim(sample_emb, ajio_embeddings).tolist()[0])        # Getting the cosine similarity scores of the Search Query with all the items
    indexes = scores.argsort()[::-1][:N]                                            # Getting the indexes of the top N items
    urls= np.array(ajio.iloc[indexes,1])                                            # Getting the links of the top N items
    # print(np.array(list(zip(urls,scores[indexes]))))                                # Printing the links and the scores of the top N items  

    print("Time Taken: ", time.time()-t)
    return {"urls":urls}

main("embroidered gown full length")
