# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import cufflinks as cf
import warnings
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff




df = pd.read_csv('googleplaystore.csv')

df.dtypes
df.shape
df.shape[0]
df.shape[1]


#10 premières valeurs du vecteur App
df['App'][0:10]
#ou
df[df.columns[0]][0:10]

#Nom des variables
df.columns

#7 appli au hasard
df.sample(7)


######################
##Nettoyer le dataset#
######################

#Colonnes avec des valeurs manquantes
df.isnull().any()

#ratio valeurs manquantes
df.isnull().sum() / len(df)

# Test s'il y a des null dans la colonne en question
#ind = np.where(df['app'].isnull()) [0]
#ind


# afficher une valeur précise
df.loc[10472]
#on constate un bug
#on supprime la ligne
df = df.drop(10472)

#retire tous les tuples ayant "Free"
df = df[df['Installs'] !='Free']

df.shape[0]

#doublons
df.drop_duplicates(subset = 'App', inplace = True)
df.shape[0]

#affiche toutes les modalités
np.unique(df['Installs'])

#retirer les + et les ,
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+',''))
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',',''))

#convertir en integer
df['Installs'] = df['Installs'].apply(lambda x: int(x))

print(type(df['Installs'].values))
df.dtypes



#convertir en float
df['Installs'] = df['Installs'].apply(lambda x: float(x))
df.dtypes


#données manquantes 
any(pd.isna(df['Installs']))

#affichage distinct
np.unique(df['Size'])

#replacer "Varies with device" par "NaN"
df['Size'] = df['Size'].apply(lambda x : str(x).replace('Varies with device','NaN'))

#convertir en M
#retirer les M
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M',''))
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',',''))

#pour les k, on repasse en 
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df.dtypes


df['Size'] = df['Size'].apply(lambda x: float(x))

np.unique(df['Size'])


df.dtypes



#histogramme
#plt.hist(df['Size'])
plt.hist(df['Size'],range=(df['Size'].min(), df['Size'].max())) #pour la version de python en 3.6

plt.hist(df.Size[df.Size.notnull()])


plt.xlabel("Size [Mo]")
plt.ylabel("Freq")
plt.subplot(2,1,1)

#plt.show(block=True)
#plt.figure()
#plt.clf()


df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', ''))
df['Price'] = df['Price'].apply(lambda x: float(x))

df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
df['Type'] = pd.factorize(df['Type'])[0]
df = df[~pd.isna(df['Rating'])]


##################################
##Etape 2 : analyse univariable ##
##################################

#import matplotlib
import bokeh

#Librairies
plt.style.use('ggplot')
plt.subplot(2,1,2)
color = sns.color_palette()
sns.set(rc={'figure.figsize':(25,15)})

#connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected=True)


number_of_apps_in_category = df['Category'].value_counts().sort_values(ascending=True)

plt.bar(number_of_apps_in_category.index, number_of_apps_in_category) # bloc 59





plt.close('all')
plt.figure()
# ...
plt.subplot(2,3,1)
plt.hist(df['Size'])
plt.xlabel("Size [Mo]")
plt.ylabel("Freq")
# ...
plt.subplot(2,1,2)
plt.hist(df['Size'])
plt.xlabel("Size [Mo]")
plt.ylabel("Freq")



plt.hist(df['Size'], label='size')
plt.plot([0,100],[1500,1500], 'r-', label='mean')
plt.xlabel("Size [Mo]")
plt.ylabel("Freq")
plt.legend()


