# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:12:51 2020

@author: Administrateur
"""
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots
from scipy.stats import norm
import pydotplus
from socket import socket
np.random.seed(123) #ensure reproducibility


pd.options.mode.chained_assignment = None  #hide any pandas warnings

df = pd.read_csv("creditcard.csv")
df_inti = df.copy()

df.shape
describe = df.describe()

df.groupby('Class').size()

df['Class'].value_counts()/len(df)*100


""" On détecte que le temps est en seconde
plt.figure()
time_val = df['Time'].values
sns.distplot(time_val, color='b')
plt.title('Ditribution des transactions bancaires dans le temps' , fontsize = 14)
plt.xlim([min(time_val),max(time_val)])
plt.xlabel("Temps en seconde")
plt.ylabel("Freq")
"""


second_day = 60*60*24
index_day1 = np.where(df['Time']<second_day)[0]
index_day2 = np.where(df['Time']>=second_day)[0]

""" 

plt.figure(figsize = (10,5))
sns.distplot(df['Time'][index_day1], color = 'b')
plt.xlabel("Time 1")
plt.ylabel("Freq")
plt.title("Day 1")
plt.show()


plt.figure(figsize = (10,5))
sns.distplot(df['Time'][index_day2], color = 'b')
plt.xlabel("Time 2")
plt.ylabel("Freq")
plt.title("Day 2")
plt.show()

df['Amount'].describe()

#Les transactions bancaires où Amount est à 0
zeroamount = df[ df['Amount']==0 ]
zeroamount.groupby("Class").size()
#Il y a 1825 transactions bancaires où Amount est à 0
#np.median(df['Amount']) est le 50% du df.describe()


plt.boxplot(df["Amount"])

"""

np.median(df['Amount'])

df.groupby('Class').describe()["Amount"]

#pyplot.close() 




x = df["Time"][df["Class"]==1]



"""
plt.figure()
sns.distplot(x[x<second_day])
plt.title("Hitogramme des transctions frauduleuses pour le jour 1")

plt.figure()
sns.distplot(x[x >= second_day])
plt.title("Hitogramme des transctions frauduleuses pour le jour 2")


plt.bloxplot(df["Amount"])







def plot_hists( df, selection='fraud' ):
    ''' plot the histograms for all the variables '''
    # ...
    # preparation of selection and color
    if selection is 'all':
        selection = df['Class'] > -1
        color = 'blue'
    elif selection is 'fraud':
        selection = df['Class'] == 1
        color = 'red'
    elif selection is 'regul':
        selection = df['Class'] == 0
        color = 'green'
    # ...
    # create graphic
    f, axs = plt.subplots(4,7, figsize=(15, 8))
    plt.subplots_adjust( wspace=0.3, hspace=0.5 )
    # ...
    irow = 0
    icol = -1
    for ivar in range(1,29):
        # variable and row / column selection
        var_name = 'V'+str(ivar)
        icol += 1
        if icol == 7:
            irow += 1
            icol = 0
        axe = axs[irow,icol]
        # ...
        # plot
        sns.distplot( df[var_name][selection],ax=axe,
                     fit=norm, color=color)
        # graphic properties
        axe.set_xlabel(None)
        axe.set_title(var_name)
        axe.set_xlim([df[var_name].min(),df[var_name].max()])
    # ...
    return None





def compare_hists( df ):
    ''' compare the fit curves between fraud and not fraud '''
    # ...
    # preparation of selection
    fraud = df['Class'] == 1
    regul = df['Class'] == 0
    # reate graphic
    f, axs = plt.subplots(4,7, figsize=(15, 8))
    plt.subplots_adjust( wspace=0.3, hspace=0.5 )
    # ...
    irow = 0
    icol = -1
    for ivar in range(1,29):
        # variable and row / column selection
        var_name = 'V'+str(ivar)
        icol += 1
        if icol == 7:
            irow += 1
            icol = 0
        axe = axs[irow,icol]
        if ivar == 1:
            label1 = 'regular'
            label2 = 'fraud'
        else:
            label1 = None
            label2 = None
        # plots
        sns.distplot( df[var_name][regul],ax=axe, color='green', hist=False, label=label1)
        sns.distplot( df[var_name][fraud],ax=axe, color='red', hist=False, label=label2)
        # graphic properties
        axe.set_xlabel(None)
        axe.set_title(var_name)
        axe.set_xlim([df[var_name].min(),df[var_name].max()])
    # ...
    return None



plot_hists( df, selection='regul' )
plot_hists( df, selection='fraud' )
compare_hists( df )
    
"""

# exemple 1 : uniquement les N premières lignes
# sub_df = df[1:5000]
# exemple 2 : un sample
sub_df = df.sample(5000)
# exemple 3 :
fraud = df['Class'] == 1
regul = df['Class'] == 0
sub_df = pd.concat(( df[fraud], df[regul].sample(5000) )).sample(frac=1)
# ...
sub_df #.head(5)
#axs = scatter_matrix(sub_df[sub_df.columns[[10,12,14]]])

"""
axs = scatter_matrix(sub_df[sub_df.columns[7:14]])


#Uniquement les transactions bancaires frauduleuses
axs = scatter_matrix(df[df.columns[7:14]].loc[df["Class"]==1], color='red')


plt.plot(df[df.columns[10]].loc[df['Class'] == 0], df[df.columns[14]].loc[df['Class'] == 0], '.', color="blue")
plt.plot(df[df.columns[10]].loc[df['Class'] == 1], df[df.columns[14]].loc[df['Class'] == 1], '.', color="red")
plt.xlabel("V10")
plt.ylabel("V14")
blue_patch = mpatches.Patch(color='blue', label='No Fraud')
red_patch = mpatches.Patch(color='red', label='Fraud')
plt.legend(handles=[blue_patch, red_patch])


plt.plot(df[df.columns[14]].loc[df['Class'] == 0], df[df.columns[12]].loc[df['Class'] == 0], '.', color="blue")
plt.plot(df[df.columns[14]].loc[df['Class'] == 1], df[df.columns[12]].loc[df['Class'] == 1], '.', color="red")
plt.xlabel("V14")
plt.ylabel("V12")
blue_patch = mpatches.Patch(color='blue', label='No Fraud')
red_patch = mpatches.Patch(color='red', label='Fraud')
plt.legend(handles=[blue_patch, red_patch])




plt.plot(df[df.columns[4]].loc[df['Class'] == 0], df[df.columns[14]].loc[df['Class'] == 0], '.', color="blue")
plt.plot(df[df.columns[4]].loc[df['Class'] == 1], df[df.columns[14]].loc[df['Class'] == 1], '.', color="red")
plt.xlabel("V4")
plt.ylabel("V14")
blue_patch = mpatches.Patch(color='blue', label='No Fraud')
red_patch = mpatches.Patch(color='red', label='Fraud')
plt.legend(handles=[blue_patch, red_patch])




sns.pairplot(df.loc[0:10000,['V4', 'V12','V14','Class']], hue="Class")

"""


sns.pairplot(sub_df[['V3','V4','V9','V10','V11','V12','V14','V16','V17','V18','Class']].sample(frac=0.2), hue="Class")

















    