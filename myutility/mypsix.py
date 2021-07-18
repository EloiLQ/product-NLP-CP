import pandas as pd

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer  

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.snowball import EnglishStemmer

import plotly.graph_objects as go
import plotly.express as px

def tsneplot(fig, title, showlegend=False, width=210, height=220):
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title, 
                         margin=dict(l=20, r=40, t=40, b=20),
                         width=width,
                         height=height,
                        )
    fig.update_layout(showlegend=showlegend)
    fig.update_traces(marker=dict(size=3.5))
    return fig

def fillmetrics(output, nb_clusters, inertia,
                inputvar, labels, truth, encoding):
    output['n_clusters'] = nb_clusters
    output['Inertia'] = inertia
    output['Silhouette'] = metrics.silhouette_score(inputvar, labels)
    output['Calinski'] = metrics.calinski_harabasz_score(inputvar, labels)
    output['Davies'] = metrics.davies_bouldin_score(inputvar, labels)
    output['ARI'] = metrics.adjusted_rand_score(truth, labels)
    output['Model'] = encoding
    return output

def myTrainTest(df, train_idx, test_idx):
    '''
    séparation du dataframe df selon
    la séparation train_idx / test_idx
    '''
    
    df_idx = df.index
    
    # évite les valeurs manquantes 
    train_idx = train_idx.intersection(df_idx)
    test_idx = test_idx.intersection(df_idx)
    
    # sélection des échantillon train / test
    train = df.loc[train_idx]
    test  = df.loc[test_idx]
    
    # suppression échantillon NaN
    train = train.dropna()
    test = test.dropna()
    
    return train, test

def bestCVandTest(df_cv, test_score, model):
    '''
    Réunit les résultats d'une classification sur CV et sur évaluation
    dans un même dataframe
    
    - df_cv : dataframe obtenu en sortie d'une recherche
    sur grille de scikit-learn
    - test_score : score obtenu sur jeu d'évaluation
    - model : nom du modèle utiliser pour encoder le texte
    '''
    
    # prend le meilleur score sur cv
    df = df_cv.head(1)
    
    # suppresion des informations sur les hyperparamètres 
    # de la recherche sur grille
    df = df.filter(regex='^(?!param)\w+$')
    
    # ajout de la colonne : type de résultats
    df['Results'] = 'CV'
    
    # ajout du résultat sur jeu d'évaluation
    df = df.append({'Results' : 'Evaluation',
                    'eval_score' : test_score},
                    ignore_index=True)
    
    # type de modèle utilisé pour l'encodage du texte
    df['Model'] = model
    
    return df

def pstemmer(text, stop_words):
    # tokenize
    tokenized = word_tokenize(text)
    
    out = []
    for word in tokenized:
        # écriture en minuscule
        word = word.lower()
        
        # retire la ponctuation et les stopwords
        if word.isalpha() and word not in stop_words:
            # racination 
            word = EnglishStemmer().stem(word)
            out.append(word)
    
    # stemmed text
    return ' '.join(out) + ' '


def wordsCateg(df, var):
    vecto = CountVectorizer()
    
    # représentation sac de mots
    matrix = vecto.fit_transform(df[var])
    
    return matrix.shape[1]


def topWords(df, var, categ):
    vecto = CountVectorizer()
    
    # représentation sac de mots
    matrix = vecto.fit_transform(df[var])
    
    # conversion matrice éparse -> dataframe
    counts = pd.DataFrame(matrix.toarray(),
                          columns=vecto.get_feature_names())
    
    # 10 premiers mots
    top10 = counts.sum().sort_values().tail(10)
    
    tmphist = px.bar(y=top10.index, x=top10.values, orientation='h')
    tmphist.update_layout(yaxis_title_text='',
                          xaxis_title_text='count',
                          title = categ, margin_r=30,
                          height=400, width=280)
    
    return tmphist

def meanEmbedding(corpus, embedding):
    '''
    récupère la valeur moyenne de chaque texte du corpus 
    dans l'espace d'embedding
    
    corpus (series pandas) : liste des textes à plonger
    embedding : modèle de plongement de mots, récupéré avec gensim
    '''
    
    df_emb = pd.DataFrame()
    
    stop_words = stopwords.words('english')
    
    for doc in corpus.str.lower(): # boucle sur chaque document
        
        # création d'un dataframe par document
        tempdf = pd.DataFrame()
    
        # boucle sur chaque mot du document
        for word in word_tokenize(doc): 
            # suppression des stopwords et des mots non alphanumériques
            if word.isalpha() and word not in stop_words: 
                
                # si le mot fait parti du modèle d'embedding
                try:
                    word_emb = embedding[word] 
                    tempdf = tempdf.append(pd.Series(word_emb),
                                           ignore_index = True)
                except:
                    pass
                
        doc_vector = tempdf.mean() # moyenne sur chaque colonne (w0, w1, w2,........w300)
        # ajoute chaque document au dataframe final
        df_emb = df_emb.append(doc_vector, ignore_index = True)
        
    return df_emb
