import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from ML_pipeline.preprocessing import output_text

def model_train(df,column_name,model,vector_size,window_size):
    x = output_text(df,column_name)
    if model=='Skipgram':

        skipgram = Word2Vec(x, vector_size =vector_size, window = window_size, min_count=2,sg = 1)
        skipgram.save('../output/model_Skipgram.bin')
        return skipgram

    elif model=='Fasttext':

        fast_text= FastText(x,vector_size=vector_size, window=window_size, min_count=2, workers=5, min_n=1, max_n=2,sg=1)
        fast_text.save('../output/model_Fasttext.bin')
        return fast_text


    

