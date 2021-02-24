import pandas as pd
import numpy as np
import os

from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

file_ = 'rocks-assessment3.arff'
load = arff.loadarff(os.path.join('kMeans',file_))
input_ = pd.DataFrame(load[0])

## Matrix plot to determine distributions
fg = px.scatter_matrix(input_)
fg.show()

## k range <2,10> //(2,4) should suffice but broader range to test it
def cluster(df: pd.DataFrame, k_min = 2, k_max = 10, multivariate = True) -> dict:
    """
    @df -> input data\n
    @k_min -> minimum cluster number\n
    @k_max -> maximum cluster number\n
    @multivariate -> boolean multivariate clustering vs pairwise -> to be developed\n
    """

    if multivariate:

        results = {k: None for k in range(k_min,k_max+1)}
        
        for ix, k in enumerate(results.keys()):

            cl = KMeans(n_clusters = k, 
                        random_state = 44, 
                        algorithm = 'full', 
                        n_init=5, 
                        init = 'k-means++').fit(df)

            out_df = pd.concat([df,pd.Series(cl.labels_, name = 'cluster')], axis = 1)

            results[k] = {'centroids': cl.cluster_centers_,
                          'labels': cl.labels_,
                          'inertia': cl.inertia_,
                          'df': out_df,
                          'figure': go.Figure(go.Splom(dimensions = [{'label': lab, 'values': out_df[lab]} for lab in \
                                                                            [d for d in out_df.columns if d != 'cluster']],
                                                      showupperhalf = False, 
                                                      marker = {'color': out_df['cluster'], 'showscale' : False, 'colorscale': 'inferno'},
                                                      opacity = .8,
                                                      diagonal_visible = False,
                                                      ),)}
            
        return results

## Plot 
def plot_clusters(results: dict, k = -1):
    """
    @results -> dictionary with clustering results\n
    @k -> which split should be plotted (default all)\n
    return None (fig.show())
    """

    if k != -1:
        fig = results[k]['figure']
        fig.update_layout(title = f'<b>KMEANS (<i>k = {k}</i>)</b><br>INERTIA: <i>{results[k]["inertia"]:.2f}</i>',
                          width = 1200,
                          height = 800,
                          font = {'family': 'Calibri', 'size': 12},)
        fig.show()

## k evaluation
def evaluation(results: dict, method = 'elbow'):
    """
    @results -> dictionary with clustering results\n
    @method -> 'elbow' or 'silhouette'\n
    return None (fig.show)
    """

    ## Elbow evaluation
    if method == 'elbow':
        fig = go.Figure(data = go.Scattergl(x = np.array(list(results.keys())),
                                            y = np.array([results[k]['inertia'] for k in results.keys()]),
                                            mode = 'lines',))
        fig.update_layout(title = '<b>ELBOW EVALUATION</b>')
        fig.show()
    
    ## Silhoutte
    elif method == 'silhouette':
        ## Create plot figure
        fig = make_subplots(rows = len(results), cols = 1, subplot_titles = [f'<b>KMEANS (k = {k})</b><br><i>SILHOUETTE EVALUTAION</i>' for k in results.keys()])
        fig.update_layout(template = 'plotly', 
                            width = 1200, 
                            height = 400 * len(results))
        
        axis_min = -.2
        shapes_ = []

        ## Colours 
        colours = px.colors.qualitative.swatches()['data'][np.random.randint(0,15)]['marker']['color']
        
        ## Run through all k's
        for ix, k in enumerate(results.keys()):
            
            ## Get the result df
            df = results[k]['df']
            
            ## Silhoutte scores (average value for all samples)
            sil_avg = silhouette_score(df.loc[:,[c for c in df if c != 'cluster']],np.array(df.loc[:,'cluster']))
            ## Silhoutte scores for each sample
            sample_sil_vals = silhouette_samples(df.loc[:,[c for c in df if c != 'cluster']],np.array(df.loc[:,'cluster']))

            ## Aggregate silhoutte scores for each cluster
            y_low = max(results.keys())
            for kk in range(k):
                i_cluster_sils = sample_sil_vals[df.loc[:,'cluster'] == kk]
                i_cluster_sils.sort()

                size_cluster = i_cluster_sils.shape[0]
                y_high = y_low + size_cluster

                colour = colours[int(kk/k*len(colours)) if k != kk else len(colours)-1]

                fig.add_trace(go.Scatter(y = np.arange(y_low,y_high),
                                          x = i_cluster_sils,
                                          mode = 'lines',
                                          showlegend = False,
                                          line = {'width': .7, 'color': colour,},
                                          fill = 'tozerox'),
                                row = ix+1,
                                col = 1)
                
                y_low = y_high + max(results.keys())
                
                ## Adjust axis left range
                if min(i_cluster_sils) < axis_min:
                    axis_min = min(i_cluster_sils)

            ## Average silhouette score
            shapes_.append({'type' : 'line',
                            'xref' : 'x' if ix == 0 else f'x{ix+1}',
                            'yref' : 'y' if ix == 0 else f'y{ix+1}',
                            'x0' : sil_avg,
                            'y0': 0,
                            'x1' : sil_avg,
                            'y1' : y_high,
                            'line_width' : 1,
                            'line_color': 'black',
                            'line_dash': 'dash'})
        
        fig.update_layout(shapes = shapes_)

        fig.update_layout(font = {'family': 'Calibri', 'size': 12},)
        fig.update_xaxes(range = [axis_min, 1])
        fig.update_yaxes(showticklabels = False)

        fig.show()


clust_ = cluster(input_,2,5)
plot_clusters(clust_,4)
evaluation(clust_, 'silhouette')
