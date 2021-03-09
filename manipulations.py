# Ce fichier comporte toutes les fonctions qui manipules les donnees.

import pandas as pd
import numpy as np
import xlsxwriter as xlw
import seaborn as sns

groundType = ['A', 'AL', 'AM', 'AS', 'B', 'C', 'G', 'GY', 'L', 'M', 'MC', 'MS', 'R', 'SL', 'S']


# On nettoie la BDD en enlevant les np.nan

def replaceGroundsById(data):
    """
    On remplace les types de sol par leurs index
    """
    mapping = dict(zip(groundType, range(len(groundType))))
    data['sol'].replace(mapping, inplace=True)


def replaceIdByGrounds(data):
    """
    On remplace les types de sol par leurs index
    """
    mapping = dict(zip(range(len(groundType)), groundType))
    data['sol'].replace(mapping, inplace=True)


def cleanBDD(data):
    """
    Fonction qui prend en argument un dataframe et qui lui enlève tout les np.nan
    """

    # On remplace les noms des sols par leurs indices dans groundType pour pouvoir utiliser dropna
    replaceGroundsById(data)
    manque_valeur = data.isna()['sol']
    for i in range(len(manque_valeur)):
        if (manque_valeur[i]):
            if (data['sol'][i-1]==data['sol'][i+1]):
                data['sol'][i]=data['sol'][i-1]

    data.dropna(axis=0, subset=['sol'], inplace=True)

    # On remet les noms des sols dans la colonne
    replaceIdByGrounds(data)

    # On re-index data
    data.reset_index(drop=True, inplace=True)


def isClean(data):
    """
    Vérification de l'abscence de np.nan
    """

    # On remplace les noms des sols par leurs indices dans groundType pour pouvoir utiliser np.isnan
    replaceGroundsById(data)

    for elem in data['sol']:
        if np.isnan(elem):
            # On remet les noms des sols dans la colonne
            replaceIdByGrounds(data)
            return False

    # On remet les noms des sols dans la colonne
    replaceIdByGrounds(data)
    return True


# Quelques statistiques sur nos bases de donnees

def compteur(data):
    """
    Compte le nombre de couche de chaque sol pour un sondage (data) et la longueur de chaque sol.
    Retourne une liste comportant 2 listes : la premiere est celle du nombre de couche correspondant à chaque sol de
        groundType et la deuxieme la longueur pour chaque sol.
    """
    nbtypes = len(groundType)

    nbr = [0 for i in range(nbtypes)]
    longueur = [0 for i in range(nbtypes)]
    z0 = 0
    nbr[groundType.index(data['sol'][0])] += 1
    for i in range(1, len(data['sol'])):
        if (data['sol'][i - 1] != data['sol'][i]):
            nbr[groundType.index(data['sol'][i])] += 1
            longueur[groundType.index(data['sol'][i - 1])] += data['z'][i - 1] - z0
            z0 = data['z'][i - 1]
    longueur[groundType.index(data['sol'][len(data['sol']) - 1])] += data['z'][(len(data['sol'])) - 1] - z0

    return [nbr, longueur]


def createRecap(dict_data):
    """
    Utilise compteur pour toutes les feuilles et enregistre le recap en .xlsx
    """

    nbtypes = len(groundType)

    workbook = xlw.Workbook('BDD_recap.xlsx')
    sheet = workbook.add_worksheet("recap")

    for i in range(nbtypes):
        sheet.write(1, 2 + 2 * i, "nb cches")
        sheet.write(1, 2 + 2 * i + 1, "longueur")

    forages = []
    for sheet_name in dict_data.keys():
        if dict_data[sheet_name].shape[0] == 0:
            continue
        forages.append(compteur(dict_data[sheet_name]))

    for row in range(len(dict_data.keys())):
        sheet.write(2 + row, 1, list(dict_data.keys())[row])
        for col in range(nbtypes):
            sheet.write(2 + row, 2 + 2 * col, forages[row][0][col])
            sheet.write(2 + row, 2 + 2 * col + 1, forages[row][1][col])

    workbook.close()


# Affichage de graphs

def gatherSheets(dict_data):
    """
    Fonction qui assemble toutes les feuilles en une seule base de donnee qu'elle retourne
    """

    keys = list(dict_data.keys())
    data = dict_data[keys[0]]
    for i in range(1, len(keys)):
        data = data.append(dict_data[keys[i]], ignore_index=True)
    return data


# Construction de la base de test et d'entrainement en conservant les couches intactes

def rassemblementCouches(init_data):
    """
    Fonction qui assemble toutes les feuilles en une seule base de donnee qu'elle retourne
    """

    if type(init_data) == dict:
        keys = list(init_data.keys())
    else:
        keys = [i for i in range(len(init_data))]
    data = init_data[keys[0]]
    for i in range(1, len(keys)):
        data = data.append(init_data[keys[i]], ignore_index=True)
    return data

def repartition(tab_couches, test_size):
    """
    Cette fonction repartie le mieux possible le tableau en 2 tableau conservant chaque couche
    """

    base_test, base_train = [], []
    nbr_elem_couche = [len(tab_couches[i]) for i in range(len(tab_couches))]
    nbr_elem = sum(nbr_elem_couche)
    nbr_elem_test = test_size * nbr_elem

    arg_min_max = np.argsort(nbr_elem_couche).tolist()
    nbr_elem_couche.sort()

    indices_base_test = []
    nbr_elem_test_actu = 0

    # S'il n'y a qu'une couche, on la met dans la base d'entrainement
    if len(tab_couches) == 1:
        base_train = tab_couches
        return base_train, base_test

    arret = False

    while nbr_elem_test_actu < int(nbr_elem_test) and len(arg_min_max) > 0:
        indice_max, nbre_max = arg_min_max.pop(), nbr_elem_couche.pop()
        if nbr_elem_test_actu + nbre_max <= nbr_elem_test:
            nbr_elem_test_actu += nbre_max
            indices_base_test.append(indice_max)

    for i in range(len(tab_couches)):
        if not (i in indices_base_test):
            base_train.append(tab_couches[i])
        else:
            base_test.append(tab_couches[i])

    if len(base_train) > 0:
        base_train = rassemblementCouches(base_train)
    if len(base_test) > 0:
        base_test = rassemblementCouches(base_test)

    return base_train, base_test

def train_test_split_couche(data, test_size=0.2):
    """
    Cette fonction prend les donnéees qui sont assemblées mais pas mélangées et renvoie 2 dataframes, le test set
    et le train set
    """

    ## On crée le dictionnaire qui regroupe toutes les couches

    # Initialisation
    dico_couche = {}

    base_train, base_test = {}, {}
    for type_sol in groundType:
        dico_couche[type_sol] = []
        base_train[type_sol] = []
        base_test[type_sol] = []

    # Remplissage de chaque tableau
    debut_couche, fin_couche = 0, 1
    for i in range(1, len(data['sol'])):
        if (data['sol'][i - 1] != data['sol'][i]):
            fin_couche = i
            dico_couche[data['sol'][debut_couche]].append(data[debut_couche:fin_couche])
            debut_couche = i
    # Ne pas oublier la derniere couche
    dico_couche[data['sol'][debut_couche]].append(data[debut_couche:])

    ## On cherche à séparer chaque couche en 2 bases : une pour l'entrainement, une pour le test, et ce sans séparer les
    ## couches

    for type_sol in groundType:
        train_test_tab = repartition(dico_couche[type_sol], test_size)
        base_train[type_sol] = train_test_tab[0]
        base_test[type_sol] = train_test_tab[1]

    return rassemblementCouches(base_train), rassemblementCouches(base_test)

if __name__ == '__main__':
    
    # Exemple d'utilisation des dernières fonctions
    import files_manager as fm
    import numpy as np
    import seaborn as sns
    import preprocessing as pp
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as stats

    excel_table = fm.readMultipleCsv('names')
    test = gatherSheets(excel_table)
    test[test == 0] = np.nan
    test.drop('Vr', axis=1, inplace=True)
    test.dropna(axis=0, inplace=True)
    test.reset_index(drop=True, inplace=True)

    base_train, base_test = train_test_split_couche(test)

    print(base_train)
    print(base_test)

