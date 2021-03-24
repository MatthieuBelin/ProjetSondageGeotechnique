# Ce fichier contient toutes les fonctions concernant les regresseurs

import sklearn.preprocessing as prepro
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, validation_curve, StratifiedKFold, \
    RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.multioutput import  MultiOutputRegressor

# Fonctions pour convertir les resultats de regresseur en sol

def norme(X1, X2):
    """X1 et X2 sont des vecteurs, cette fonction calcule la norme euclidienne"""
    return np.sqrt((X2[0] - X1[0]) ** 2 + (X2[1] - X1[1]) ** 2)


def plusProcheVoisin(X):
    """Fonction qui à un point X renvoie le sol qui est le plus proche suivant les criteres maisons"""

    i_min, d_min = 0, norme(X, [logGranulometrie[0], argilosite[0]])
    for i in range(1, len(logGranulometrie)):
        d = norme(X, [logGranulometrie[i], argilosite[i]])
        if d < d_min:
            i_min, d_min = i, d

    return groundType[i_min]


def conversionPredictionSol(data):
    """Prend en argument le tableau de label que le regressor fait et renvoie le tableau de sol associer"""
    sol = []
    for d in data:
        sol.append(plusProcheVoisin(d))

    return sol


def scorePrediction(prediction_sol, sol_attendu):
    """fonction qui renvoie le pourcentage de bonnes réponses"""

    assert len(prediction_sol) == len(sol_attendu)

    nombre_juste = sum([(prediction_sol[i] == sol_attendu[i]) for i in range(len(prediction_sol))])

    return nombre_juste / len(prediction_sol)

   
# Pour normaliser notre classification
groundType = ['A', 'AL', 'AM', 'AS', 'B', 'C', 'G', 'GY', 'L', 'M', 'MC', 'MS', 'R', 'SL', 'S']
logGranulometrie = np.array([-2.68, -2.3, -2, -1.82, 3.6, 2.7, 0.9, 3., -1.7, 2., 2.18, 1.78, -1.3, -1., -0.3])
argilosite = np.array([8, 7, 8, 6, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 1])
logGranulometrie = (logGranulometrie - logGranulometrie.mean())/logGranulometrie.std()
argilosite = (argilosite - argilosite.mean())/argilosite.std()

def crossValidationMLPR(X, Y):
    """Fonction qui essaie plusieurs possibilités"""

    # On découpe le set en set d'enrtainement et de validation
    print("***Decoupe le set de validation***")
    x_train, x_validation, y_train, y_validation_txt = train_test_split(X, Y, stratify=Y, test_size=0.2, shuffle=True)
    y_train, y_validation = transformerGranuArgi(y_train), transformerGranuArgi(y_validation_txt)

    print('***Definition des parametres a tester***')
    param = {
        'hidden_layer_sizes': [tuple(np.random.randint(20, 35, np.random.randint(3, 5, 1))) for _ in range(5)]
    }

    print('***Definition des modeles a entrainer***')
    mlpr = [MLPRegressor(solver='adam', max_iter=1000, alpha=1e-5, activation='tanh', hidden_layer_sizes=param['hidden_layer_sizes'][i]) for i in range(len(param['hidden_layer_sizes']))]
    multioutput_rna = [MultiOutputRegressor(modele) for modele in mlpr]

    # Score de resultat justes sur le set de validation
    resultat_sur_validation = [0 for _ in range(len(param['hidden_layer_sizes']))]

    for i, modele in enumerate(multioutput_rna):
        print(f"[Entrainement du modele {i}] Couches de neurones : {param['hidden_layer_sizes'][i]}")
        modele.fit(x_train, y_train)
        print(modele.score(x_validation, y_validation))
        y_res = modele.predict(x_validation)
        y_res = conversionPredictionSol(y_res)
        print(scorePrediction(y_res, np.array(y_validation_txt)))
        print('\n')
