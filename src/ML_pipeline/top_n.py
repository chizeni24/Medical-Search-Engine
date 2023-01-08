#defining function to define cosine similarity
from numpy import dot
from numpy.linalg import norm
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
import pandas as pd
from ML_pipeline.utils import read_data
from ML_pipeline.return_embed import get_mean_vector
from ML_pipeline.preprocessing import preprocessing_input

### define cosine simialrity function
def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b)) 


### define top_n function 
#function to return top n similar result
def top_n(query,model_name,column_name):

    df = read_data("../input/Dimension-covid.csv")
    if model_name=='Skipgram':
        word2vec_model = Word2Vec.load('../output/model_Skipgram.bin')
        K=pd.read_csv('../output/skipgram-vec-abstract.csv')
    else:        
        word2vec_model=Word2Vec.load('../output/model_Fasttext.bin')
        K=pd.read_csv('../output/Fasttext-vec-abstract.csv')
    #input vectors
    query = preprocessing_input(query)
    
    query_vector=get_mean_vector(word2vec_model,query)
    #Model Vectors
      #Loading our pretrained vectors of each abstract

    p=[]                          #transforming dataframe into required array like structure as we did in above step
    for i in range(df.shape[0]):
        p.append(K[str(i)].values)    
    x=[]
    #Converting cosine similarities of overall data set with input queries into LIST
    for i in range(len(p)):
        x.append(cos_sim(query_vector,p[i]))
    
    
    #store list in tmp to retrieve index
    tmp=list(x)
    
    #sort list so that largest elements are on the far right
    res = sorted(range(len(x)), key = lambda sub: x[sub])[-10:]
    sim=[tmp[i] for i in reversed(res)]
    
    #get index of the 10 or n largest element
    L=[]
    for i in reversed(res):
    
        L.append(i)
        
    df1 = read_data("../input/Dimension-covid.csv")    
    return df1.iloc[L, [1,2,5,6]],sim     #returning dataframe (only id,title,abstract ,publication date)

