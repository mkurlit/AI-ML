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

## Show data - inspection
#sns.set_theme(context = 'paper', style = 'darkgrid')
#sns.pairplot(iris, hue = 'type')
#plt.show()

## Simple kMeans on each pair of features (assuming that the k is known)
def cluster(df):
    ## Pairs    
    results = {pair: None for pair in list(itertools.combinations([c for c in df.columns if c != 'type'],2))}
    
    ## Plot figure
    fig = make_subplots(rows = len(results), cols = 1, subplot_titles = [str(ix+1) for ix, p in enumerate(results)])
    fig.update_layout(template = 'plotly', 
                        width = 800, 
                        height = 300 * len(results))
    ## Set Colours
    def colours(x):
        colours_ = {'setosa' : '#41285e', 'versicolor': '#ab84da', 'virginica': '#6809d9', 'incorrect': '#ee3223'}
        return colours_[x]

    ## Show incorrect assignments
    def marks(x):
        shapes = {1: 'circle', 0: 'x'}
        return shapes[x]
    
    ## Track best feature pair
    best = {'pair': 1, 'accuracy': -1}
    
    for ix, p in enumerate(results.keys()):
        ## KMEANS++
        cl = KMeans(n_clusters = 3, random_state = 44, algorithm = 'full').fit(df.loc[:,p])
        out_df = pd.concat([df.loc[:,p],
                            pd.Series(cl.labels_, name = 'cluster'), 
                            pd.Series(df.loc[:,'type'], name = 'label')],axis =1)
        ## Map assignments
        cl_map = {out_df[out_df['label'] == t]['cluster'].value_counts().idxmax() : t for t in out_df['label'].unique()}
        out_df['cluster'] = out_df.apply(lambda r: cl_map[r['cluster']], axis = 1)
        out_df['correct_cluster'] = [int(b) for b in out_df['label'] == out_df['cluster']]
        out_df['cluster_adj'] = [cl if ok else 'incorrect' for cl, ok in zip(out_df['cluster'], out_df['correct_cluster'])]
        results[p] = {'out_df': out_df,
                        'centroids': cl.cluster_centers_,
                        'labels': cl.labels_,
                        'inertia': cl.inertia_,
                        'match': {l: out_df[out_df['label'] == l]['correct_cluster'].sum()/out_df[out_df['label'] == l].shape[0] \
                            for l in out_df['label'].unique()},
                        'accuracy': out_df.correct_cluster.sum()/out_df.shape[0],}
        
        ## check if total accuracy surpassed the best
        if results[p]['accuracy'] > best['accuracy']:
            best.update({'pair': ix+1, 'accuracy': results[p]['accuracy']})

        ## Add to the figure
        fig.add_trace(go.Scatter(x = np.array(out_df[p[0]]), 
                                y = np.array(out_df[p[1]]),
                                opacity = .8,
                                marker = dict(color = list(map(colours, out_df['cluster_adj'])), 
                                             symbol = list(map(marks, out_df['correct_cluster'])),
                                              showscale = False),
                                mode = 'markers',
                                showlegend = False),
                                row = ix+1, 
                                col = 1)

        fig.update_xaxes(title_text = p[0][:p[0].rfind(' ')], 
                            row = ix+1, 
                            col = 1, 
                            title_standoff = 5)
        fig.update_yaxes(title_text = p[1][:p[1].rfind(' ')], 
                            row = ix+1, 
                            col = 1, 
                            title_standoff = 5)
        
        fig.update_layout(font = {'family': 'Calibri', 'size': 12})
        
        fig['layout']['annotations'][ix].update(text = \
            f'<b>CLUSTERING {ix+1}</b>'+\
            f'<br><b>Inertia</b>: <i>{cl.inertia_:.2f}</i>'+\
            f'<br><b>Accuracy</b>: {", ".join([str(k)+" = <i>"+str(v)+"</i>" for k,v in results[p]["match"].items()])}'+\
            f'<br>TOTAL: {out_df.correct_cluster.sum()/out_df.shape[0]:.2f}')
    
    ## Mark best pair split
    fig['layout']['annotations'][best['pair']-1]['text'] += ' <b>**BEST RUN</b>'
    
    return results, fig

r, fg = cluster(iris)
fg.show()





