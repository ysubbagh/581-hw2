'''After recipes have been mapped to ingredient space and flavor space
Plot tsne clustering
Plot with bokeh interactive plotting
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, ColumnDataSource, Legend
import nltk
nltk.data.path.append('/Users/yasminesubbagh/nltk_data')
# nltk.download()

# convert rgb colors to hex value colors, used to have a lot of colors for all the cuisines
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), 
        int(rgb[1] * 255), 
        int(rgb[2] * 255)
    )

#take some regional cuisines, tsne clustering, and plotting
def tsne_cluster_cuisine(df,sublist):
    lenlist=[0]
    df_sub = df[df['cuisine']==sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine']==cuisine]
        df_sub = pd.concat([df_sub, temp],axis=0,ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine','recipeName'],axis=1)
    print(df_X.shape, lenlist)

    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed', init='random').fit_transform(dist)


    palette = sns.color_palette("hls", len(sublist))
    plt.figure(figsize=(10,10))
    for i,cuisine in enumerate(sublist):
        plt.scatter(tsne[lenlist[i]:lenlist[i+1],0],\
        tsne[lenlist[i]:lenlist[i+1],1],c=palette[i],label=sublist[i])
    plt.legend()

#interactive plot with boken; set up for four categories, with color palette; pass in df for either ingredient or flavor
def plot_bokeh(df, sublist, filename):
    lenlist = [0]
    df_sub = df[df['cuisine'] == sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine'] == cuisine]
        df_sub = pd.concat([df_sub, temp], axis=0, ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine', 'recipeName'], axis=1)
    print(df_X.shape, lenlist)

    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed', init='random').fit_transform(dist)

    # Define colors for each cuisine
    #palette = ['red', 'green', 'blue', 'yellow']
    palette = sns.color_palette("hsv", len(sublist))
    hex_colors = [rgb_to_hex(color) for color in palette]
    colors = []
    for i in range(len(sublist)):
        colors.extend([hex_colors[i]] * (lenlist[i+1] - lenlist[i]))
    #print(colors)

    # Set up Bokeh plot
    output_file(filename)
    source = ColumnDataSource(data=dict(
        x=tsne[:, 0],  
        y=tsne[:, 1],  
        cuisine=df_sub['cuisine'].tolist(),
        recipe=df_sub['recipeName'].tolist(),
        fill_color=colors  
    ))

    hover = HoverTool(tooltips=[
        ("cuisine", "@cuisine"),
        ("recipe", "@recipe")
    ])

    p = figure(width=1000, height=1000, tools=[hover],
               title="flavor clustering")

    circles = p.circle('x', 'y', size=10, source=source, fill_color='fill_color')

    # Add a legend
    legend_items = []
    for i, cuisine in enumerate(sublist):
        legend_items.append((cuisine, [p.circle(x=[None], y=[None], size=10, fill_color=hex_colors[i])]))
    legend = Legend(items=legend_items)
    p.add_layout(legend, 'right')
    p.legend.location = "top_left"

    show(p)



if __name__ == '__main__':
    yum_ingr = pd.read_pickle('Flavor-Network-master/data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('Flavor-Network-master/data/yummly_ingrX.pkl')
    yum_tfidf = pd.read_pickle('Flavor-Network-master/data/yum_tfidf.pkl')

    ''' the old code to only cluster 4 cuisines

    #select four cuisines and plot tsne clustering with ingredients
    sublist = ['Italian','French','Japanese','Indian']
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']
    tsne_cluster_cuisine(df_ingr,sublist)

    #select four cuisines and plot tsne clustering with flavor
    sublist = ['Italian','French','Japanese','Indian']
    df_flavor = yum_tfidf.copy()
    df_flavor['cuisine'] = yum_ingr['cuisine']
    df_flavor['recipeName'] = yum_ingr['recipeName']
    tsne_cluster_cuisine(df_flavor,sublist)
    
    #select four cuisines and do interactive plotting with bokeh
    plot_bokeh(df_flavor,sublist, 'test1.html')
    plot_bokeh(df_ingr,sublist, 'test2.html')
    '''

    # modified code to show all the cuisines clustered
    all_cuisines = yum_ingr['cuisine'].unique() 

    # Use all cuisines for ingredient clustering
    df_ingr = yum_ingrX.copy() # the ingrredient files
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']
    tsne_cluster_cuisine(df_ingr, all_cuisines)

    # Use all cuisines for flavor clustering
    df_flavor = yum_tfidf.copy() # the taste / flavor file (dataset)
    df_flavor['cuisine'] = yum_ingr['cuisine']
    df_flavor['recipeName'] = yum_ingr['recipeName']
    tsne_cluster_cuisine(df_flavor, all_cuisines)

    # Perform interactive plotting with Bokeh using all cuisines
    plot_bokeh(df_flavor, all_cuisines, 'flavor_all_cuisines.html')
    plot_bokeh(df_ingr, all_cuisines, 'ingr_all_cuisines.html')