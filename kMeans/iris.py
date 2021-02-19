import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.cluster import KMeans
import itertools
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

iris = {k:v for k,v in datasets.load_iris().items() if k in ('data', 'feature_names', 'target', 'target_names')}
iris = pd.concat([pd.DataFrame(iris['data'], columns = iris['feature_names']),
                  pd.Series([iris['target_names'][ix] for ix in iris['target']], name = 'type')],
                  axis = 1)

## Show data
sns.set_theme(context = 'paper', style = 'darkgrid')
sns.pairplot(iris, hue = 'type')
plt.show()

## Types map
iris_types = {ix : t for ix, t in zip(range(3), iris['type'].unique())}

## Simple kMeans on each pair of features (assuming that the k is known)
def cluster(df, sample = False):
    ## Pairs
    results = {pair: None for pair in list(itertools.combinations([c for c in df.columns if c != 'type'],2))}
    ## Full results
    #full_cols = ['split', 'f_1', 'f_1_value','f_2', 'f_2_value', 'cluster']
    #full_df = pd.DataFrame(columns = full_cols)

    ## Plot figure
    fig = make_subplots(rows = len(results), cols = 1)

    if not sample:
        for ix, p in enumerate(results.keys()):
            cl = KMeans(n_clusters = 3, random_state = 44).fit(df.loc[:,p])
            out_df = pd.concat([df.loc[:,p],
                                pd.Series(cl.labels_, name = 'cluster'), 
                                pd.Series(df.loc[:,'type'], name = 'label')],axis =1)

            results[p] = {'out_df': out_df,
                            'centroids': cl.cluster_centers_,
                            'labels': cl.labels_,
                            'inertia': cl.inertia_,
                            'match': {l: out_df[out_df['label'] == l]['cluster'].value_counts().max()/out_df[out_df['label'] == l].shape[0] \
                                for l in out_df['label'].unique()},}
            
            ## Add to the figure
            fig.add_trace(go.Scatter(out_df, x = p[0], y = p[1], color = 'cluster'))

            #out_df['split'] = ix 
            #out_df['f_1'] = p[0]
            #out_df['f_1_value'] = out_df.loc[:,p[0]]
            #out_df['f_2'] = p[1]
            #out_df['f_2_value'] = out_df.loc[:,p[1]]
            #full_df = full_df.append(out_df.loc[:,full_cols], ignore_index = True)

    ## Facet Grid    
    fg = sns.FacetGrid(full_df, row = 'split', hue = 'cluster')
    fg.map(sns.scatterplot, 'f_1_value', 'f_2_value')
    plt.show()

    return results, full_df

r, f = cluster(iris)





