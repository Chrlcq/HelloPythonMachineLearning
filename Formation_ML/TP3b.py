# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:15:40 2020

@author: Administrateur
"""
"""
#pour le temps d'execution du code
import time
time0 = time.time()
print(time.time()-time0); time0= time.time()
"""


# Import des librairies
import numpy as np # librairie de calcul numérique
import pandas as pd # librairie de statistiques
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt # librairie de tracé de figures
import matplotlib.patches as mpatches
import time

# Librairies Machine learning
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD # librairie d'analyse factorielle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import confusion_matrix #for model evaluation
import collections
from collections import Counter


df = pd.read_csv("creditcard.csv")
df_inti = df.copy()

def centrereduit(x):
    return (x- np.mean(x)) / np.std(x)

# centrer réduire les variables sauf la classe

for i in range (0, len(df.columns)-1):
    df[df.columns[i]] = centrereduit(df[df.columns[i]])


np.mean(df['V1'])

np.std(df['V1'])



#Centrer réduire de sorte à ce que la moyenne = 0 et l'écart-type = 1
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(df.drop("Class", axis=1).values)


'''

plt.figure()

plt.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(df["Class"] == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
plt.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(df["Class"] == 1), cmap='coolwarm', label='Fraud', linewidths=2)
plt.title('ACP sur les variables quantitatives centrées réduites', fontsize=14)
plt.grid(True)
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")

blue_patch = mpatches.Patch(color='blue', label='No Fraud')
red_patch = mpatches.Patch(color='red', label='Fraud')
plt.legend(handles=[blue_patch, red_patch])
plt.show()

'''


'''Etape 2 : Création d'une base d'apprentissage et d'une base de test qui
 soient représentatif du nombre de transactions bancaires frauduleuses


En effet, il y a 492
 transactions frauduleuses sur 284.315 transactions bancaires normales.
 Pour cette application spécifique à notre cas d'étude, peut-être est-ce pertinent 
 de découper une base d'apprentissage où seront représentées 80% des transactions bancaires 
 frauduleuses.
 
 '''


# Option pour créer une base d'apprentissage : celle qui permet d'avoir une base de données de "petite taille"
# et équilibrée par rapport à la variable de sortie "Class"

ratio_apprentissage = 0.8 # proportion de la base de données utilisée pour l'apprentissage

# sous-calculs
df_shuffle = df.sample(frac=1) # désordre aléatorie du jeu de données
num_fraude = df_shuffle.groupby('Class').size()[1] #combien de fraudes dans notre jeu de données
num_train = int(ratio_apprentissage * num_fraude + 1)

# base d'entrainement
df_fraude = df_shuffle.loc[df_shuffle['Class'] == 1][0:num_train]
df_non_fraude = df_shuffle.loc[df_shuffle['Class'] == 0][0:num_train]
df_train = pd.concat([df_fraude, df_non_fraude])
df_train = df_train.sample(frac=1)

# base de test
df_fraude = df_shuffle.loc[df_shuffle['Class'] == 1][num_train:num_fraude]
df_non_fraude = df_shuffle.loc[df_shuffle['Class'] == 0][num_train:num_fraude]
df_test = pd.concat([df_fraude, df_non_fraude])
df_test = df_test.sample(frac=1)

print( "Répartition dans le jeu d'apprentissage :", df_train.groupby('Class').size(),
      "\nRépartition dans le jeu de test :", df_test.groupby('Class').size() )





#Fonction qui détecte les outliers
def detect_outliers(df, variables):
    '''détection les individus dont une des variables d'entrée est en dehors de l'intervalle acceptable'''
    individus = [] # initialisation de la liste des individus outliers"
    # ...
    for var in variables:
        m = df[var].mean(axis=0)
        std = df[var].std(axis=0)
        mini = m - 3*std # seuil bas pour un outlier
        maxi = m + 3*std # seuil haut pour un outlier
        # ...
        individus_nouveaux = np.where( np.logical_or( df[var] < mini, df[var] > maxi ) )[0].tolist()
        individus = np.unique( individus + individus_nouveaux ).tolist()
    # ...
    return individus

# exécution de la fonction
variables = df_train.columns[1:29]
outliers = detect_outliers(df_train, variables)
#print( len(outliers), 'outliers sur', df_train.shape[0])

#df_train = df_train.drop(df_train.index[outliers]); # suppressions des valeurs


X = df.drop('Class', axis=1)
y = df['Class']

X_train = df_train.drop('Class', axis=1)
y_train = df_train['Class']

X_test = df_test.drop('Class', axis=1)
y_test = df_test['Class']








"""Etape 3 : Réduction de dimensions"""

### 1) Analyse en composantes principales (ACP)

# PCA Implementation
X_reduced_pca = PCA(n_components=2).fit_transform(X_train.values)
# other options for PCA :
#X_reduced_pca = PCA(n_components=2, svd_solver='full').fit_transform(X_train.values)
#X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X_train.values) # if the first one is too long to apply



"""

plt.figure()

plt.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
plt.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)
plt.title('ACP', fontsize=14)
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.grid(True)

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
plt.legend(handles=[blue_patch, red_patch])
plt.show()

"""
mypca = PCA(n_components=2)
mypca.fit_transform(X_train.values)
mypca.explained_variance_ratio_


sum(mypca.explained_variance_ratio_)



mypca.components_[1]



# T-SNE Implementation
tsne = TSNE(n_components=2, random_state=99) # model
X_reduced_tsne = tsne.fit_transform(X_train.values) # fit
#https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

"""
fig, ax = plt.subplots()
plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=y_train, cmap='coolwarm', linewidths=2)
ax.legend(*scatter.legend_elements()) # produce a legend with the unique colors from the scatter
# ax.legend(scatter.legend_elements(prop="colors")[0], ['regular','fraud'], title="Class") # produce a legend with the unique colors from the scatter
# ...
plt.title('t-SNE', fontsize=14)
plt.grid(True)

"""


"""Etape 4 : Méthodes de clustering (apprentissage automatique non supervisé)"""

""" 4.1. K-means appliqué sur les données projetées par t-SNE """


from sklearn.cluster import KMeans

# exécution des k-means sur le nuage de point projeté par t-sne
kmeans = KMeans(n_clusters=3, random_state=0) # model
kmeans.fit( X_reduced_tsne[:,0:2] ) # fit

# résultats du clustering
kmeans.labels_ # numéro du cluster dans lequel sont chacun des individus (index : ligne) du dataframe
kmeans.cluster_centers_ # centres de chaque cluster




# représentation sur la projection de t-sne le résultat de k-means
cluster = kmeans.predict(X_reduced_tsne[:,0:2]) # ici, équivalent à : kmeans.labels_ car on le fait sur les donnée d'apprentissage
# ...

"""
plt.figure()
scatter = plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=cluster) #cmap='coolwarm'
leg = plt.legend(*scatter.legend_elements(prop="colors"), title='Cluster') # produce a legend with the unique colors from the scatter
plt.plot( kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 'sr', markersize=10) # centroids
# ...
plt.title('kMeans', fontsize=14)
plt.xlabel("T-SNE var 1")
plt.ylabel("T-SNE var 2")

plt.show()
"""




""" 4.2. DBSCAN appliqué sur les données projetées par t-SNE"""

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=5, min_samples=2)
dbscan.fit(X_reduced_tsne)


"""
# représentation sur la projection de t-sne le résultat de k-means
cluster = dbscan.labels_
# ...
scatter = plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=cluster) #cmap='coolwarm'
leg = plt.legend(*scatter.legend_elements(prop="colors"), title='Cluster') # produce a legend with the unique colors from the scatter
# ...
plt.title('DBSCAN', fontsize=14)
plt.xlabel("T-SNE var 1")
plt.ylabel("T-SNE var 2")
"""





"""Etape 5 : Apprentissage automatique supervisé"""

"""5.0. Pré-requis : méthodes d'analyse génériques"""

#Pour faciliter l'analyse, on crée une fonction qui affiche les métriques indiquant la qualité de la prévision

from sklearn.metrics import confusion_matrix # for model evaluation

def print_metrics( y_test, y_esti ):
    ''' confusion_matrix_test = confusion_matrix(y_test, y_esti_tree) '''
    # ...
    confusion_matrix_test = confusion_matrix( y_test, y_esti )
    # ...
    true_negative = confusion_matrix_test[0,0]
    false_positive = confusion_matrix_test[0,1]
    false_negative = confusion_matrix_test[1,0]
    true_positive = confusion_matrix_test[1,1]
    # ...
    sensitivity = true_positive / ( true_positive + false_negative )
    specificity = true_negative / ( true_negative + false_positive )
    accuracy = ( true_positive + true_negative ) / ( true_positive + true_negative + false_positive + false_negative )
    # ...
    print('accuracy :',np.round(100*accuracy,2),'%')
    print('sensitivity :',np.round(100*sensitivity,2),'%')
    print('specificity :',np.round(100*specificity,2),'%')
    print('true_positive :',true_positive)
    print('true_negative :',true_negative)
    print('false_positive :',false_positive)
    print('false_negative :',false_negative)
    # ...
    return None


from sklearn.metrics import roc_curve, auc # for model evaluation

def plot_roc_curve( y_test, y_pred_proba ):
    ''' plot the ROC curve and compute AUC '''
    # ...
    fpr, tpr, thresholds = roc_curve( y_test, y_pred_proba )
    # ...
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, '.-')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # ...
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    # ...
    # Aire sous la courbe ROC (AUC)
    auc_metric = auc(fpr, tpr)
    print('Area under curve (AUC) :',auc_metric)
    # ...
    return None


"""5.1. Arbre de décision"""

from sklearn.tree import DecisionTreeClassifier

# Classification par les arbres de décision
dtc = DecisionTreeClassifier() # model
dtc.fit(X_train, y_train) # fit


# Prévision sur les données de test
y_pred = dtc.predict(X_test) # predict
dtc.score(X_test, y_test) # first evaluation based on accuracy


# Matrice de confusion et métriques
confusion_matrice_tree = confusion_matrix( y_test, y_pred )
print("Arbre de décision")
print_metrics( y_test, y_pred )




#Visualisation de l'arbre méthode 3
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(dtc, filled=True, max_depth=5, fontsize=10)



"""5.2. Random Forest"""

from sklearn.ensemble import RandomForestClassifier

# Classification par une forêt aléatoire
rf = RandomForestClassifier(max_depth=10) # model / construction
rf.fit(X_train, y_train) # fit / apprentissage


# Prévision sur les données de test
y_pred = rf.predict(X_test) # predict / prévision : booléen True/False
y_pred_proba = rf.predict_proba(X_test)[:,1] # predict : avec probabilité d'être True/False entre 0 et 1
# The predicted class probability is the fraction of samples of the same class in a leaf.


# Matrice de confusion et métriques
confusion_matrice_rf = confusion_matrix( y_test, y_pred )
print("Random Forest")
print_metrics( y_test, y_pred )


# plot ROC curve and compute area under curve (AUC)
plot_roc_curve( y_test, y_pred_proba )




"""5.3. K plus proches voisins (KNN)"""


# Application de la méthode de prévision sur le jeu d'entrainement
neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)


# évaluation de la matrice de confusion sur le jeu de test
y_esti_knn = neigh.predict(X_test)
y_esti_knn


confusion_matrice_knn = confusion_matrix(y_test, y_esti_knn)
confusion_matrice_knn


sensitivity_knn = confusion_matrice_knn[1,1]/(confusion_matrice_knn[1,1]+confusion_matrice_knn[1,0])
sensitivity_knn






mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=20000,
                    solver='adam')
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))



#print("Test set score: %f" % mlp.score(X_test, y_test))
prediction = mlp.predict(X_test)

confusion_matrice = confusion_matrix(y_test, prediction)
print(confusion_matrice)





specificity = confusion_matrice[0,0]/(confusion_matrice[0,0]+confusion_matrice[0,1])
print('Specificity : ', specificity)

sensitivity = confusion_matrice[1,1]/(confusion_matrice[1,1]+confusion_matrice[1,0])
print('Sensitivity : ', sensitivity )


from sklearn.model_selection import cross_validate
cross_validate(mlp, X, y, cv=5)






perm = PermutationImportance(mlp).fit(X_test, y_test)
eli5.show_weights(perm,feature_names = X_train.columns.tolist(),top=31)



### Sans le drop
mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=100, solver='lbfgs', verbose=False)
mlp.fit(X_train, y_train)
print('Taux de précision sans drop', mlp.score(X_test, y_test)*100)

### Avec le drop
X_train_2=X_train.drop(['V18','V15','V25'],axis=1)
X_test_2=X_test.drop(['V18','V15','V25'],axis=1)

mlp2 = MLPClassifier(hidden_layer_sizes=(10), max_iter=20, solver='lbfgs', verbose=False)
mlp2.fit(X_train_2, y_train)
print('Taux de précision avec drop de V18, V15, V25', mlp2.score(X_test_2, y_test)*100)