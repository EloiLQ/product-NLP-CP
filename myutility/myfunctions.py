import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ipywidgets as ipw
import plotly.express as px

histSize = {'width' : 370, 'height' : 370}

def completeness(data, dtype=True):
    '''
    affiche les informations sur les variables du dataframe :
    complétude, cardinalité, type de données
    '''

    ## complétude des variables 
    count = data.count().rename('count')
    
    ## cardinalité des variables
    nunique = data.nunique().rename('nunique')
    
    ## type des variables
    dtypes = data.dtypes.rename('dtype')
    
    ## concatènation des séries
    out = pd.concat([count, nunique], axis=1)
    if dtype:
        out = pd.concat([out, dtypes], axis=1)
    return out


## fonction pour aligner plusieurs figures côte à côte
def hboxing(figlist, nb_h = 3):
    fig_out = []
    ## converti la list de figure en list de widgets figures
    for elt in figlist:
        fig_out.append(go.FigureWidget(elt))
        
    
    ## liste des hbox
    hboxlist = []
    
    ## créer une hbox toutes les nb_h figures
    for elt in range(0, len(fig_out), nb_h):
        tmpHBox = ipw.HBox(fig_out[elt:elt+nb_h])
        hboxlist.append(tmpHBox)
        
    
    return ipw.VBox(hboxlist)


def firstQuantile(series, quantile):
    '''
    sélectionne les individues de la series 
    contenue dans le premier quantile 
    '''
    
    # index 95% des clients qui ont le moins dépensé
    cut = series.quantile(quantile)

    # sélection des 95% les moins dépensiés
    series = series[series < cut]
    
    return series

