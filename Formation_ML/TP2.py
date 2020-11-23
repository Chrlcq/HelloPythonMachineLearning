# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:08:03 2020

@author: Administrateur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #for plotting


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

df.describe()

#Analyse des moyennes des variables discrétisées par la variable cible target (deux modalitéS 0 ou 1, sain ou malade respectivement)
df.groupby('target').mean()
#Elle sera effectuée uniquement sur les variables quantitatives



"""Analyse univariable de la variable cible"""

#Nombre d'individus discrétisés par la variable cible
df.target.value_counts()

plt.figure(figsize=(40,20))
#Variable cible
plt.subplot(3,4,1)
sns.countplot(x="target", data=df)
plt.title('Distribution des patients non atteint et atteint de maladie cardiovasculaire')
plt.xlabel("Cible")


"""Analyse univariable du sexe"""

plt.subplot(3,4,2)
sns.countplot(x='sex', data=df)
plt.xlabel("Sexe (0 = Femme, 1 = Homme)")
plt.title("Distribution des patients par leur sexe")



#Tableau croisé
ax = plt.subplot(3,4,3)
#pd.crosstab(df["sex"], df["target"])



#pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6))
pd.crosstab(df.sex,df.target).plot(kind="bar", ax=ax)
plt.title('Distibution des patients en fonction du sexe et discrétisé par la variable cible')
plt.xlabel('Sexe (0 = Femme, 1 = Homme)')
plt.xticks(rotation=0)
plt.legend(["Sain", "Malade"])
plt.ylabel('Freq')
#plt.show()


ax = plt.subplot(3,4,4)
#pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
pd.crosstab(df.age,df.target).plot(kind="bar", ax=ax)
plt.title('Distibution des patients en fonction de l âge et discrétisé par la variable cible')
plt.xlabel('Age')
plt.ylabel('Freq')
#plt.show()

#plt.savefig('heartDiseaseAndAges.png')

## Analyse univariable de la tension artérielle au repos

#Variable resting_blood_pressure

plt.subplot(3,4,5)

bx = plt.boxplot(df['resting_blood_pressure'])
plt.ylabel('Tension artérielle au repos [mm Hg]')
plt.title('Distribution de la tension artérielle au repos')



plt.subplot(3,4,6)

#Récupérer les outliers
#whiskers = terme pour dire . en dehors des boite à moustache
seuil = bx['whiskers'][1]._yorig[1]
outliers = df[df["resting_blood_pressure"]> seuil]
print(len(outliers))



#Figure
sns.countplot(x='target', data=outliers)
plt.title('Distribution des outliers de la tension artérielle au repos')



#Figure distribution des tension artérielles au repos discrétisé par variable cible
#plt.figure(figsize=(10,5))
plt.subplot(3,4,7)
plt.boxplot(df[df['target']==0]['resting_blood_pressure'])
plt.ylim([90,210])
plt.ylabel('Tension artérielle au repos [mm Hg]')
plt.title('Patients sains')

plt.subplot(3,4,8)
plt.boxplot(df[df['target']==1]['resting_blood_pressure'])
plt.ylim([90,210])
plt.ylabel('Tension artérielle au repos [mm Hg]')
plt.title('Patients malades')




plt.subplot(3,4,9)
plt.scatter(x=df.age[df.target==1], y=df.max_heart_rate_achieved[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.max_heart_rate_achieved[(df.target==0)])
plt.legend(["Malade", "Sain"])
plt.xlabel("Age")
plt.ylabel("Fréquence cardiaque maximale atteinte")
plt.show()

ax = plt.subplot(3,4,10)
#Equivalent
#df.groupby('target').boxplot(column='resting_blood_pressure', figsize=(15,6))
#df.groupby('target').boxplot(column='resting_blood_pressure')
#plt.show()


plt.subplot(3,4,11)
plt.scatter(x=df.age[df.target==1], y=df.max_heart_rate_achieved[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.max_heart_rate_achieved[(df.target==0)])
plt.legend(["Malade", "Sain"])
plt.xlabel("Age")
plt.ylabel("Fréquence cardiaque maximale atteinte")
plt.show()



ax = plt.subplot(3,4,12)
pd.crosstab(df.fasting_blood_sugar, df.target).plot(kind="bar", ax=ax)
plt.title('Distribution des patients discrétisés par la glycémie à jeun')
plt.xlabel('Glycémie à jeun - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Sain", "Malade"])
plt.ylabel('Fréquence des patients malades ou non')
#plt.show()

df

#Boites à moustaches
df.boxplot()


