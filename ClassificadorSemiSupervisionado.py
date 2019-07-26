##----DUPLA----##
#Claudemir Woche
#Francisco Yuri

import copy
import numpy as np
from sklearn.base import BaseEstimator


class ClassificadorSemiSuper(BaseEstimator):
    def __init__(self, classificador, k):
        self.__classificador = classificador  
        self.__k = k                          #Num de classificados em cada iteração
        self.__n_caracteristicas = 0
        self.__n_exemplos = 0
        self.__xDA = []  # Com classe
        self.__yDA = []

        self.__xDB = []  # Sem classe
        self.__yDB = []

    def fit(self, X, y):
        self.__n_exemplos, self.__n_caracteristicas = X.shape
        indices = []
        for i in range(self.__n_exemplos):
           if y[i] != -1:                           #Quem tiver classe -1 vai pro conjunto DB
                self.__xDA.append(X[i])             #Quem não, vai pro DA
                self.__yDA.append(y[i])
           else:
                self.__xDB.append(X[i])
                self.__yDB.append(y[i])

   
        # ---TREINA COM DA(COM CLASSE)---#
        self.__classificador.fit(self.__xDA, self.__yDA)

        
        # ---CLASSFICA K DBs (SEM CLASSE) SEM A NECESSIDADE DE OS K SEREM ALEATÓRIOS---#
        n = len(self.__yDB)
        p = self.__k
        
        
        while(p < n):
            dadosTes = []
            for i in range(self.__k):
                
                if len(self.__xDB) <= i:
                    break
               
                dadosTes.append(list(self.__xDB[i]))        #Seleciona os dados a serem classificados
                self.__xDA.append(list(self.__xDB[i]))      #Inclui eles em xDA
                del self.__xDB[i]                           #Deleta os dados de DB
                del self.__yDB[i]
                
            dadosTes_np = np.array(dadosTes)
            previsao = self.__classificador.predict(dadosTes_np)    #Classifica os dados 
            
            for prv in previsao:
                self.__yDA.append(prv)                      #Inclui os dados classificados em yDA
                
            x = copy.deepcopy(self.__xDA)               
            y = copy.deepcopy(self.__yDA)
            self.__classificador.fit(x, y)                  #Treina com os novos dados
            
            n = n - self.__k

            
        #Classifica os dados que sobraram e treina com eles
        
        dadosTes = []
        n = len(self.__yDB)
        i = 0
        
        for i in range(n):
            dadosTes.append(list(self.__xDB[i]))
            self.__xDA.append(list(self.__xDB[i]))
            #del self.__xDB[i]
            #del self.__yDB[i]
        
        dadosTes_np = np.array(dadosTes)
        previsao = self.predict(dadosTes_np)
        
        for prv in previsao:
            self.__yDA.append(prv)
       
        x = copy.deepcopy(self.__xDA)
        y = copy.deepcopy(self.__yDA)
        self.__classificador.fit(x, y)
        
        return self
    
    
    def predict(self, x_test):
        return self.__classificador.predict(x_test)
    
    
    def score(self, x_test, y_test):
        return self.__classificador.score(x_test, y_test)
