# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:54:22 2020

@author: Administrateur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots
import pydotplus
np.random.seed(123) #ensure reproducibility
pd.options.mode.chained_assignment = None  #hide any pandas warnings



df = pd.read_csv("heart.csv")

#Renommer les noms de colonnes
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 
              'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
              'exercise_induced_angina', 'st_depression', 'st_slope', 
              'num_major_vessels', 'thalassemia', 'target']

#Types des variables
df.dtypes

#Définir les types appropriés : les variables numériques discrètes deviennent de type object car elles ne sont pas continues
df['sex'] = df['sex'].astype('object')
df['chest_pain_type'] = df['chest_pain_type'].astype('object')
df['fasting_blood_sugar'] = df['fasting_blood_sugar'].astype('object')
df['rest_ecg'] = df['rest_ecg'].astype('object')
df['exercise_induced_angina'] = df['exercise_induced_angina'].astype('object')
df['st_slope'] = df['st_slope'].astype('object')
df['thalassemia'] = df['thalassemia'].astype('object')

#Vérification des nouveaux types
df.dtypes




#split the data
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), 
                                                    df['target'], 
                                                    test_size = .2, 
                                                    random_state=10) 


#Nombre de lignes dans le jeu d'apprentissage
X_train.shape[0]

#Nombre de lignes dans le jeu de test
X_test.shape[0]

#Arbre de décision
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

#Estime la variable cible de la base de test
y_esti_tree = dtc.predict(X_test)

#Affiche les 10 premiers résultats
y_esti_tree[1:10]

#Matrice de confusion
confusion_matrice_tree = confusion_matrix(y_test, y_esti_tree)


confusion_matrice_tree
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
#En ligne : les classes réelles
#En colonnes : les classes prédites

tn, fp, fn, tp = confusion_matrix(y_test, y_esti_tree).ravel()
print("True Negative : " + str(tn))
print("False Positive : " + str(fp))
print("False Negative : " + str(fn))
print("True Positive : " + str(tp))


sensitivity_tree = confusion_matrice_tree[1,1]/(confusion_matrice_tree[1,1]+confusion_matrice_tree[1,0])
sensitivity_tree



specificity_tree = confusion_matrice_tree[0,0]/(confusion_matrice_tree[0,0]+confusion_matrice_tree[0,1])


#Nombre de prédiction correctes (VP+VN normalisé)
dtc.score(X_test, y_test)

#https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score


#Visualisation de l'arbre méthode 1
tree.export_graphviz(dtc)
#http://webgraphviz.com/

#http://viz-js.com/
#tree.export_graphviz(dtc,out_file='graph.txt')

"""
#Visualisation de l'arbre méthode 2
tree.export_graphviz(dtc, out_file="tree.dot")
with open("tree.dot", 'w') as my_file:
    tree.export_graphviz(dtc)
    
dot_data = tree.export_graphviz(dtc, out_file=None,
            feature_names=X_train.columns,
            class_names=['sick','not sick'],
            filled=True, rounded=True,
            special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("heart.pdf")

"""

#Visualisation de l'arbre méthode 3
from sklearn.tree import plot_tree

plt.figure()
plot_tree(dtc, filled=True)
plt.show()


plt.figure()
plot_tree(dtc, filled=True, max_depth=1)
plt.show()





#Analyse de sensibilité
profondeur = dtc.get_depth()
sensi_depth_app = np.ones(profondeur) * np.nan
sensi_depth_test = np.ones(profondeur) * np.nan

for i in range(1,profondeur+1):
    #model
    dtc1 = DecisionTreeClassifier(max_depth=i)
    dtc1.fit(X_train, y_train)
    #predict sur test
    y_esti_tree = dtc1.predict(X_test)
    confusion_matrice = confusion_matrix(y_test, y_esti_tree)
    sensi_depth_test[i-1] = confusion_matrice[1,1]/(confusion_matrice[1,1]+confusion_matrice[1,0])
    #predict sur app
    y_esti_tree = dtc1.predict(X_train)
    confusion_matrice = confusion_matrix(y_train, y_esti_tree)
    sensi_depth_app[i-1] = confusion_matrice[1,1]/(confusion_matrice[1,1]+confusion_matrice[1,0])

plt.grid()
plt.plot(range(1,profondeur+1), sensi_depth_app, color="blue")
plt.plot(range(1,profondeur+1), sensi_depth_test, color="red")
plt.ylabel("Sensibilité")
plt.xlabel("Profondeur de l'arbre [Nombre de niveaux]")
plt.legend(['Apprentissage','Test'])
plt.title('Arbre de décision')
plt.show()


"""RANDOM FOREST"""

#le modèle
rf = RandomForestClassifier(max_depth=5)

#Apprentissage
rf.fit(X_train, y_train)

#Prédiction sur les données de tests
y_esti_rf = rf.predict(X_test)

#Affiche les 10 premiers résultats
y_esti_rf[1:10]


#Matrice de confusion
confusion_matrice_rf = confusion_matrix(y_test, y_esti_rf)
confusion_matrice_rf

specificity_rf = confusion_matrice_rf[0,0]/(confusion_matrice_rf[0,0]+confusion_matrice_rf[0,1])
specificity_rf

sensitivity_rf = confusion_matrice_rf[1,1]/(confusion_matrice_rf[1,1]+confusion_matrice_rf[1,0])
sensitivity_rf



#Projection des résultats d'une matrice de confusion sur l'espace ROC
fpr_pred, tpr_pred, thresholds_pred = roc_curve(y_test, y_esti_rf)

fig, ax = plt.subplots()
ax.plot(fpr_pred, tpr_pred, '.')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)



#class probability predictions
y_esti_quant_rf = rf.predict_proba(X_test)[:,1]
y_esti_quant_rf



#Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_esti_quant_rf)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, '.-')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)




#Aire sous la courbe ROC (AUC)
auc(fpr, tpr)

#Nombre de prédiction correctes (VP+VN normalisé)
rf.score(X_test,y_test)




"""Permutation importance"""

perm = PermutationImportance(rf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


#Code pour afficher sur l'IDE (méthode 1)
print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=X_test.columns.tolist())))


#Code pour afficher sur l'IDE (méthode 2)
perm = PermutationImportance(rf, random_state=1).fit(X_test, y_test)
html_obj = eli5.show_weights(perm, feature_names = X_test.columns.tolist())
with open('permutation-importance.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))




#Analyse de sensibilité 
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier

rf_model = RandomForestClassifier()

parameters = [{"n_estimators":[1,5,10,20,30,40,49,50,51], 'max_depth': [2,3,4, 5, 10, 15]}]
grid_bag = GridSearchCV(estimator=rf_model, param_grid=parameters, cv=5, scoring="recall")

grid = grid_bag.fit(X_train, y_train)



grid.best_score_

grid.best_params_