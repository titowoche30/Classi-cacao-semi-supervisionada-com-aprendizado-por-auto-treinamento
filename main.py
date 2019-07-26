##----DUPLA----##
#Claudemir Woche
#Francisco Yuri

import numpy as np
import random
from sklearn import datasets,model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from ClassificadorSemiSupervisionado import ClassificadorSemiSuper


# Leitura dos dados e preenchimento dos sem classe
S = datasets.load_breast_cancer()
#S = datasets.load_digits()
#S = datasets.load_iris()
observacoes = S.data
classe = S.target

m, n = observacoes.shape

print('Numero de exemplos:', m)
print('Numero de caracteristicas:', n,"\n")

# Particao em (treinamento,teste)

(dadosTre, dadosTes, classeTre, classeTes) = model_selection.train_test_split(
    S.data, S.target, train_size=2 / 3, random_state=7)

print('Tamanho do conjunto de treinamento:', len(dadosTre))
print('Tamanho do conjunto de teste:', len(dadosTes))

numExTre = len(classeTre)
lista = list(range(0, numExTre))
random.shuffle(lista)
lista = random.sample(lista, int(numExTre / 2))

for i in lista:
    classeTre[i] = -1

unique, counts = np.unique(classeTre, return_counts=True)
dic = dict(zip(unique, counts))
print('Numero de exemplos sem classe:', dic[-1],"\n")

clfV = []


##--------------SE QUISER MUDAR O K, MUDE AQUI EM BAIXO------------##
k=20
##-----------------------------------------------------------------##


##-----COMENTE UM CLASSIFICADOR PARA QUE ELE NÃO SEJA USADO--------##
dict_classifiers = {
    "Floresta Aleatoria": RandomForestClassifier(n_estimators=100),
    "Vizinhos mais próximos": KNeighborsClassifier(),
    #"Support Vector Machines Linear": SVC(gamma = "scale"),
    #"Árvore de Decisão": DecisionTreeClassifier(),
    #"Rede Neural": MLPClassifier(max_iter = 1000),
    #"Naive Bayes": GaussianNB()
    #"Regressão Logistica": LogisticRegression(solver = "newton-cg", max_iter = 1000,multi_class="auto"),
    #"Classificador de Boosting de Gradiente ": GradientBoostingClassifier(),
}
##-----------------------------------------------------------------##
print('k =',k)
z = 0

for value in dict_classifiers.values():
    classfi = ClassificadorSemiSuper(value,k)
    clfV.append(classfi)
    clfV[z].fit(dadosTre, classeTre)
    z+=1


for key,value in zip(dict_classifiers,clfV):
    prec = float("{0:.3f}".format(value.score(dadosTes, classeTes)))
    print("Precisao com classificador {} = {} ".format(key,prec))

