# Ce fichier comporte toutes les fonctions qui manipules les donnees.

import xlsxwriter as xlw
from PIL import Image, ImageFont, ImageDraw
from sklearn.preprocessing import StandardScaler

from preprocessing import *

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
            if (data['sol'][i - 1] == data['sol'][i + 1]):
                data['sol'][i] = data['sol'][i - 1]

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


# Automatisation de quelques taches sur la BDD
def BDDminimal(data, suppr_Pr=False):
    """Fonction qui trie toute les lignes ou il manque des données"""
    bdd_mini = gatherSheets(data)

    if suppr_Pr:
        bdd_mini.drop('Pr', axis=1, inplace=True)
    bdd_mini[bdd_mini == 0] = np.nan
    bdd_mini.drop('Vr', axis=1, inplace=True)
    bdd_mini.dropna(axis=0, inplace=True)
    bdd_mini.reset_index(drop=True, inplace=True)

    return bdd_mini


def featuresLabel(data):
    """Separe les donnees en features et label"""
    features = data.drop('sol', axis=1)
    label = data['sol']

    return features, label


def scaling(X_train, X_test):
    """Fonction qui scale les features"""

    # scalerX = MinMaxScaler()
    scalerX = StandardScaler()

    X_train = scalerX.fit_transform(X_train)
    X_test = scalerX.transform(X_test)


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


def featuresLabel(data):
    """Separe les donnees en features et label"""
    features = data.drop('sol', axis=1)
    label = data['sol']

    return features, label


def printConfusionmatrix(X_test, y_test, model):
    # Trace la matrice de confusion en pourcentage, normalisée par ligne
    poids = np.unique(y_test,
                      return_counts=True)
    disp, ax = plt.subplots(figsize=(10, 10))
    conf_mat = confusion_matrix(y_test, model.predict(X_test), labels=poids[0], normalize='true')
    conf_mat = conf_mat * 100
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat[0])):
            conf_mat[i][j] = round(conf_mat[i][j], 3)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=poids[0])
    disp.plot(ax=ax)
    disp.ax_.set_title('Matrice de confusion normalisée par label véritable')
    plt.show()


def SortiePhoto(X_test, y_test, model):
    coupures = [0]
    compteur_erreurs = []  # contient les indices des erreurs de prédiction
    nombre_strate = len(y_test)
    for i in range(2, len(y_test)):
        if y_test[i - 1] != y_test[i]:
            coupures.append(i)  # Contient les indices des changements de couches

    prediction = model.predict(X_test)

    H = 800  # hauteur
    L = 300  # largeur
    font = ImageFont.truetype(r'arial.ttf', 12)
    im = Image.new('RGB', (L, H + 100), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    for j in range(0, nombre_strate):
        print(prediction[j], y_test[j])
        if prediction[j] != y_test[j]:
            compteur_erreurs.append(j)
            draw.rectangle((30, 50 + (j / nombre_strate) * H, 210, 50 + ((j + 1) / nombre_strate) * H),
                           fill=(255, 0, 0), outline=(255, 0, 0))
        else:
            draw.rectangle((30, 50 + (j / nombre_strate) * H, 210, 50 + ((j + 1) / nombre_strate) * H),
                           fill=(0, 255, 0),
                           outline=(0, 255, 0))

    for i in range(1, len(coupures)):
        draw.line((15, 50 + (coupures[i] / nombre_strate) * H, 20, 50 + (coupures[i] / nombre_strate) * H),
                  fill=(0, 0, 200), width=2)
        draw.text((0, ((coupures[i] - coupures[i - 1]) / 2 + coupures[i - 1]) * H / nombre_strate + 50),
                  y_test[coupures[i - 1]], font=font, fill='blue')
    draw.text(
        (0, ((nombre_strate - coupures[len(coupures) - 1]) / 2 + coupures[len(coupures) - 1]) * H / nombre_strate + 50),
        y_test[nombre_strate - 1], font=font, fill='blue')
    draw.line((15, 50, 20, 50), fill=(0, 0, 200), width=2)
    draw.line((15, H + 50, 20, H + 50), fill=(0, 0, 200), width=2)
    draw.line((20, H + 50, 20, 50), fill=(0, 0, 200), width=2)
    print(coupures)
    print(compteur_erreurs)
    print(100 - ((len(compteur_erreurs) / nombre_strate) * 100))

    im.save('test.jpg', quality=100)


def score_par_type_de_sol(model):
    '''Fonction qui renvoie les graphes d'apprentissage par type de sol  '''

    excel_table = fm.readMultipleCsv('names')
    data_usable = BDDminimal(excel_table, suppr_Pr=True)

    base_train, base_test = train_test_split_couche(data_usable)

    x_train, y_train = featuresLabel(base_train)
    x_test, y_test = featuresLabel(base_test)

    # Préparation de la base test

    scalerX = StandardScaler()
    x_train1 = scalerX.fit_transform(x_train)
    x_test = scalerX.transform(x_test)
    data_test = pd.DataFrame(x_test, columns=["z", "VIA", "Po", "Pi", "Cr"])
    y_test = pd.DataFrame.to_numpy(y_test)
    Dico_sol_test = {}
    data_test['sol'] = y_test
    for type in groundType:
        Dico_sol_test[type] = []
    i = 0
    for type_sol in data_test['sol']:
        Dico_sol_test[type_sol].append(data_test.iloc[i])
        i += 1
    # Préparation de la base d'entrainement

    y_train1 = y_train
    data = pd.DataFrame(x_train1, columns=["z", "VIA", "Po", "Pi", "Cr"])
    y_train1 = pd.DataFrame.to_numpy(y_train1)
    data['sol'] = y_train1

    Dico_sol = {}

    for type in groundType:
        Dico_sol[type] = []
    i = 0
    for type_sol in data['sol']:
        Dico_sol[type_sol].append(data.iloc[i])
        i += 1

    clefs = list(Dico_sol.keys())

    # Coupe de la base d'entraiement à la proportion p

    proportion = np.linspace(41, 100, 30)
    list_graph = []
    list_prop_graph = []
    for p in proportion:
        proportions_grap = []
        Dico_sol1 = {}

        print(p)

        for clef in clefs:
            p_sol = int(len(Dico_sol[clef]) * (p / 100))
            proportions_grap.append(p_sol)
            Dico_sol1[clef] = Dico_sol[clef][0:p_sol]

        x_train_def = []
        y_train1 = []

        for i in range(len(clefs)):
            for j in range(len(Dico_sol1[clefs[i]])):
                x_train_def.append(pd.DataFrame.to_numpy(Dico_sol1[clefs[i]][j]))
                y_train1.append(Dico_sol1[clefs[i]][j][-1])

        x_train1 = np.delete(x_train_def, 5, 1)

        # Entraînement du modèle
        model.fit(x_train1, y_train1)

        # Evaluation du score

        score_sol = []

        for i in range(len(clefs)):
            x_test1 = Dico_sol_test[clefs[i]]
            if len(x_test1) == 0:
                score_sol.append(0)
            else:
                x_test1 = np.delete(x_test1, 5, 1)
                y_pred = model.predict(x_test1)
                score = 0
                for j in range(0, len(y_pred)):
                    if y_pred[j] == clefs[i]:
                        score += 1
                score_sol.append(100 * score / len(y_pred))

        list_graph.append(score_sol)
        list_prop_graph.append(proportions_grap)

    # Tracé des graphes
    list_graph = np.transpose(list_graph)
    list_prop_graph = np.transpose(list_prop_graph)

    for i in range(0, len(clefs)):
        plt.plot(list_prop_graph[i], list_graph[i])
        plt.title(clefs[i])
        plt.show()


if __name__ == '__main__':
    # Exemple d'utilisation
    import files_manager as fm
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    excel_table = fm.readMultipleCsv('names')
    test = gatherSheets(excel_table)
    test[test == 0] = np.nan
    test.drop('Vr', axis=1, inplace=True)
    test.dropna(axis=0, inplace=True)
    test.reset_index(drop=True, inplace=True)

    base_train, base_test = train_test_split_couche(test)

    print(base_train)
    print(base_test)
