# [Medical-Search-Engine](https://chizeni24-medical-search-engine-medical-5hks9o.streamlit.app/)

## Objective
This project aims to create domain-specific medical word embeddings using Word2Vec and FastText in Python to develop a search engine and Streamlit UI. Embeddings are a feature learning technique that maps words or phrases to vectors of real numbers, enabling the representation of words as semantically meaningful dense vectors. This method overcomes many of the problems of other techniques like one-hot encodings and TFIDF. In this project, Word2Vec and FastText are used to build the models that represent distributed representations of words in a corpus. The search engine and Streamlit UI can help medical professionals and the public access accurate and relevant information about medical topics.

## Data description
The project will be based on a clinical trials dataset related to Covid-19, which can be accessed from the provided [link](https://chizeni24-medical-search-engine-medical-5hks9o.streamlit.app/). The dataset contains 10666 rows and 21 columns of data. The project will focus on two essential columns of the dataset:
- Title 
- Abstract

## Tech stack
- Language - Python
- Libraries and Packages - pandas, numpy, matplotlib, plotly, gensim, streamlit,
nltk.

## Approach 
1. Importing the required libraries
2. Reading the dataset
3. Pre-processing
4. Exploratory Data Analysis (EDA)
 Data Visualization using word cloud
5. Training the ‘Skip-gram’ model
6. Training the ‘FastText’ model
7. Model embeddings – Similarity
8. PCA plots for Skip-gram and FastText models
9. Convert abstract and title to vectors using the Skip-gram and FastText model
10.Use the Cosine similarity function
11.Perform input query pre-processing
12.Define a function to return top ‘n’ similar results
13.Result evaluation
14.Run the Streamlit Application

## Modular Code
``` your tree
├── input
│   └── Dimension-covid.csv
├── src
    ├── engine.py
    └── ML_pipeline
        ├── preprocessing.py
        ├── return_embed.py
        ├── top_n.py
        ├── train_model.py
        └── utils.py
 lib
│   └── Reference
│       └── Medical Embeddings_Final.ipynb
├── output
│   ├── FastText-vec-abstract.csv
│   ├── FastText-vec-title.csv
│   ├── model_Fasttext.bin
│   ├── model_Fasttext.bin.wv.vectors_ngrams.npy
│   ├── model_Skipgram.bin
│   ├── skipgram-vec-abstract.csv
│   └── skipgram-vec-title.csv
├── requirements.txt
├── Medical Embeddding_Final.ipynb
└──  Medical.py
```
## Execution Instructions 
Dowload the repo or clone 

```markdown 
git clone https://github.com/chizeni24/Medical-Search-Engine.git
```
The folder structure should be as:
1. input
2. src
3. output
4. lib

## Folder description

1. **Input folder:-**  contains the data that we use for our analysis. A clinical trials dataset is considered for our project, which is based on Covid-19.
  - Dimension-covid.csv.
 There are 10666 rows and 21 columns present in the dataset.
2. **Src folder:-**  This is the most important folder of the project. This folder contains
all the modularized code for all the above steps in a modularized manner. This folder consists of:
- engine.py
- ML_pipeline
The ML_pipeline is a folder that contains all the functions put into different
python files, which are appropriately named. These python functions are
then called inside the engine.py file.
3. **Output folder:-**  The output folder contains the best fitted model that we trained
for this data. This model can be easily loaded and used for future use and the
user need not have to train all the models from the beginning.
**Note:-** This model is built over a chunk of data. One can obtain the model for
the entire data by running engine.py by taking the quality data to train the
models.
4. **Lib folder:**  This is a reference folder. It contains,
  - The original ipython notebook for step by step processing :wrench:.
  - The Medical.py notebook  used for running Streamlit UI
