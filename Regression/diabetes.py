import pandas as pd
import numpy as np
import os

from scipy.io import arff
from sklearn import linear_model as lm
from sklearn.metrics import roc_curve, auc 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

file_ = 'diabetes.arff'
load = arff.loadarff(os.path.join('Regression',file_))
input_ = pd.DataFrame(load[0])

## Binarise test class
test_map = {'positive': 1, 'negative': 0}
input_['class'] = input_.apply(lambda r: test_map[r['class'].decode('utf-8')[r['class'].decode('utf-8').rfind('_')+1:]], 
                                axis = 1)

## Input scan
def plot_input(df: pd.DataFrame, corrs = False):
    """
    @df -> dataframe with input data\n
    @corrs -> should correlation matrix be plotted (default : False)\n
    return None (fig.show())
    """
    
    if not corrs:
        ## Visualise data distribution
        fg = px.scatter_matrix(df, 
                               dimensions = [c for c in df.columns if c != 'class'], 
                               color = 'class',)
        fg.update_traces(diagonal_visible = False, showupperhalf = False)
        fg.show()
    
    else:
        ## Check correlation of data
        heat = go.Heatmap(z = np.array(df.corr('pearson')),
                          x = df.columns,
                          y = df.columns,
                          xgap = 5, ygap = 5,
                          colorscale = 'ylorrd',
                          reversescale = True)
        fg = go.Figure(data = heat,
                       layout = go.Layout(width = 800,
                                          height = 800,
                                          xaxis_showgrid = False,
                                          yaxis_showgrid = False,))
        fg.show()

plot_input(input_, True)


def log_reg(df: pd.DataFrame, target : str, multivariate = True, train = .7, cross_val = False):
    """
    @df -> input data\n
    @target -> target attribute name\n
    @multivariate -> pairwise vs multivariate regression\n
    @train -> training ration (default .7)\n
    @cross_val -> should cross validation be used (defalut : True)\n
    return out_df, score, fg -> resulting df, score, figure
    """

    if multivariate:
            
        ## Split train, test
        mask = np.zeros(df.shape[0], dtype = int)
        mask[:int(train*len(mask))] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)
        X_train = df.loc[mask, [c for c in df.columns if c != target]]
        y_train = df.loc[mask, target]
        X_test = df.loc[~mask, [c for c in df.columns if c != target]]
        y_test = df.loc[~mask, target]
        
        ## Fit model
        ## No Cross Validation
        if not cross_val:
            lr_model = lm.LogisticRegression(max_iter = 200)
        ## Cross Validation
        else:
            lr_model = lm.LogisticRegressionCV(cv = 5, 
                                               random_state = 44,
                                               max_iter = 200,)

        lr_model.fit(X_train, y_train)
        predictions = lr_model.predict(X_test)
        score = lr_model.score(X_test, y_test)
        out_df = pd.concat([X_test.reset_index(drop = True), 
                            y_test.reset_index(drop = True), 
                            pd.Series(predictions, name = 'prediction')], axis = 1)
        
        ## Confusion matrix
        TP = sum(np.logical_and(out_df[target] == out_df['prediction'], out_df['prediction'] == 1))
        TN = sum(np.logical_and(out_df[target] == out_df['prediction'], out_df['prediction'] == 0))
        FP = sum(np.logical_and(out_df[target] != out_df['prediction'], out_df['prediction'] == 1))
        FN = sum(np.logical_and(out_df[target] != out_df['prediction'], out_df['prediction'] == 0))
        CM = np.array([[TP, FP],[FN, TN]])

        ## Create figure for Confusion Matrix and ROC
        fg = make_subplots(rows = 2, cols = 1, 
                           subplot_titles = ['CM', 'ROC'], 
                           specs = [[{'t':.05}],[{}]], 
                           horizontal_spacing = .05, 
                           vertical_spacing = .1)
        
        fg.update_layout(template = 'plotly',
                          width = 1200,
                          height = 1600,
                          margin_t = 16,)

        ## Confusion Matrix Plot
        hm_fig = ff.create_annotated_heatmap(z = np.flip(CM, axis = 0), 
                                             x = ['Actual Positive', 'Actual Negative'],
                                             y = ['Predicted Positive', 'Predicted Negative'],
                                             colorscale = 'blues',
                                             name = 'CM')
        fg.add_trace(hm_fig.data[0],
                     row = 1,
                     col = 1)
        fg['layout']['annotations'][0].update(text = f'<b>CONFUSION MATRIX</b><br>'+\
                                                     f'<i>PRECISION: <b>{(TP)/(TP+FP):.2f}</b></i><br>'+\
                                                     f'<i>RECALL: <b>{TP/(TP+FN):.2f}</b></i><br>'+\
                                                     f'<i>ACCURACY: <b>{(TP+TN)/(out_df.shape[0]):.2f}</b></i>',
                                              align = 'center',
                                              xanchor = 'center',)
        fg['layout']['annotations'] += hm_fig['layout']['annotations']
        
        ## ROC Plot
        fpr, tpr, thresholds = roc_curve(y_test, lr_model.predict_proba(X_test)[:,1])
        fg.add_trace(go.Scatter(x = fpr,
                                y = tpr,
                                fill = 'tozeroy',
                                fillcolor = 'rgba(8, 48, 107, .6)',
                                mode = 'none',
                                name = 'ROC',
                                opacity = .3,),
                     row = 2,
                     col =1,)
        fg['layout']['annotations'][1].update(text = f'<b>ROC</b><br><i>(Area under the Curve: <b>{auc(fpr, tpr):.4f}</b></i>)')
        fg['layout']['xaxis2'].update({'title': 'False Positive'})
        fg['layout']['yaxis2'].update({'title': 'True Positive'})
        
        return out_df, score, fg

df, score, f = log_reg(input_, 'class', cross_val= True)
f.show()

