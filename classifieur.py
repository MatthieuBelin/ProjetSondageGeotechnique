# Ce fichier regroupe toutes les fonctions en lien avec les classifieurs

from preprocessing import *
from manipulations import *
from file_manager import *

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, validation_curve, StratifiedKFold, \
    RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier


def RNAClassifier(data):
    """Première fonction qui entraine et teste un RNA pour classifier les sols"""

    print("***Pré-traitement des donnees***")

    # On récupere toutes les donnees utilisables
    data_usable = BDDminimal(data)

    # On récupére les features et les label
    x, y = featuresLabel(data_usable)

    # On sépare la base de donnees en une base d'entrainement et une de test
    print("[Separation en set d'entrainement et de test]")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True)

    # Mise a l'echelle et encodage
    scaling(x_train, x_test)

    transformer = LabelEncoder()
    y_train_encode = transformer.fit_transform(y_train).ravel()
    y_test_encode = transformer.transform(y_test).ravel()

    print("[Preparation pour cross-validation]")

    # Separation base test / validation
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    # Parametres à tester
    param = {
        'alpha': [pow(10, -i) for i in range(5)],
        'hidden_layer_sizes': [tuple(np.random.randint(20, 35, np.random.randint(3, 5, 1))) for _ in range(8)]
    }
    # Modele
    modele = MLPClassifier(solver='adam', max_iter=1000)

    grid = GridSearchCV(modele, param_grid=param, cv=cv)

    grid.fit(x_train, y_train_encode)

    print(f'Le meilleur modele a obtenu un score de {grid.best_score_} pour les parametres {grid.best_params_}.')

    best_model = grid.best_estimator_
    print(f"Sur la base test, on obtient un score de {best_model.score(x_test, y_test_encode)}.")

    return grid.best_estimator_


def RNAClassifier_random_search(data, base_estimator, param, n_iter, n_jobs = -3):
    """Fonction qui entraine et teste un RNA pour classifier les sols, en cherchant des hyperparamètres à l'aide de RandomizedSearchCV"""
    # n_jobs est un argument de RandomizedSearchCV. Il indique le parallélisme, càd le nombre de processeurs alloués pour les calculs. S'il vaut 1, seul 1 processeur est alloué.
    # Sinon, il vaut -1, -2... et le nombre de processeurs alloués est alors n_CPU + 1 + n_jobs (avec n_CPU le nombre de CPU de notre PC)
    
    print("***Pré-traitement des donnees***")

    # On récupere toutes les donnees utilisables
    data_usable = BDDminimal(data)

    # On récupére les features et les label
    x, y = featuresLabel(data_usable)

    print("[Separation en set d'entrainement et de test]")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True)

    # Scale + encodage
    scaling(x_train, x_test)

    transformer = LabelEncoder()
    y_train_encode = transformer.fit_transform(y_train).ravel()
    y_test_encode = transformer.transform(y_test).ravel()

    print("[Preparation pour cross-validation]")

    # Separation base test / validation
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    print("***Optimisation***")

    # Search
    opti = RandomizedSearchCV(base_estimator, param_distributions=param, cv=cv,
                              n_iter=n_iter, n_jobs=n_jobs)  # Le nombre de processeurs alloués, l'estimateur de base, le nombre d'essais et le dictionnaire des paramètres sont donnés par l'utilisateur en argument

    opti.fit(x_train, y_train_encode)

    print(f'Le meilleur modele a obtenu un score de {opti.best_score_} pour les parametres {opti.best_params_}.')

    best_model = opti.best_estimator_
    print(f"Sur la base test, on obtient un score de {best_model.score(x_test, y_test_encode)}.")

    return opti.best_estimator_

def supprDoublon(tab):
    """Fonction qui supprime les doublons d'une liste"""
    liste_pure = []

    for element in tab:
        if element not in liste_pure:
            liste_pure.append(element)

    tab = []
    for element in liste_pure:
        tab.append(element)

def test_optimisation_RNAClassifier():
    
    # ----- Trace la distribution suivie par le log d'une variable suivant la distribution powerlognorm que l'on va choisir pour le paramètre alpha -----
        
    fig, ax = plt.subplots(1, 1)

    c, s = 1, 1
    y = stats.powerlognorm.rvs(c, s, scale=0.01, size=10000) # scale règle le centrage.

    ax.hist(np.log10(y), bins=30)

    plt.show()

    # ----- Le Randomized search -----
    
    # Importe les données
    excel_table = readMultipleCsv('names')
    
    # Estimateur de base, permettant notamment de sélectionner le solveur à tester
    base_estimator = MLPClassifier(solver="sgd", max_iter=1000, verbose=True) # ou solver="adam"
    
    hidden_layer_sizes = [tuple(np.random.randint(20, 35, np.random.randint(3, 5, 1))) for i in range(500)]
    supprDoublon(hidden_layer_sizes)
    
    # Parametres a modifier suivant ce que nous voulons tester
    param = {
        "alpha": stats.powerlognorm(1, 1, scale=0.01),
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": ["logistic", "tanh", "relu"],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": stats.powerlognorm(1, 1, scale=0.001),
        "batch_size": np.arange(200, 500, 10) # batch_size : nombre de données sur lesquelles l'estimateur s'entraine
    } # "learning_rate" n'est que pour le solver sgd

    RNAClassifier_random_search(excel_table, base_estimator, param, 1)

def RNA_cross_val(data, base_estimator, n_jobs=-3):
    print("***Pré-traitement des donnees***")

    # On récupere toutes les donnees utilisables
    data_usable = BDDminimal(data)

    # On récupére les features et les label
    x, y = featuresLabel(data_usable)

    print("[Separation en set d'entrainement et de test]")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True)

    # Scale + encodage
    scaling(x_train, x_test)

    transformer = LabelEncoder()
    y_train_encode = transformer.fit_transform(y_train).ravel()

    print("[Preparation pour cross-validation]")

    # Separation base test / validation
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    scores = cross_val_score(base_estimator, x_train, y_train_encode, cv=cv, n_jobs=n_jobs)

    print(scores)



def alpha_validation_curve(data, base_estimator, n_jobs=-3):
    print("***Pré-traitement des donnees***")

    # On récupere toutes les donnees utilisables
    data_usable = BDDminimal(data)

    # On récupére les features et les label
    x, y = featuresLabel(data_usable)

    print("[Separation en set d'entrainement et de test]")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True)

    # Scale + encodage
    scaling(x_train, x_test)

    transformer = LabelEncoder()
    y_train_encode = transformer.fit_transform(y_train).ravel()

    print("[Preparation pour cross-validation]")

    # Separation base test / validation
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    # ----- Paramétrage de la validation curve -----

    # Choix du paramètre et de son intervalle de recherche
    param_range = np.logspace(-11, 0, 55)

    # Calcul de la validation curve
    train_scores, valid_scores = validation_curve(base_estimator, x_train, y_train_encode,
                                                  "alpha", param_range,
                                                  cv=cv, n_jobs=n_jobs)

    # Tracé de la validation curve
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(valid_scores, axis=1)
    validation_scores_std = np.std(valid_scores, axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Validation curve on alpha for Adam")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Score")

    lw = 2
    ax.semilogx(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    ax.semilogx(param_range, validation_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    ax.fill_between(param_range, validation_scores_mean - validation_scores_std,
                    validation_scores_mean + validation_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

    return (train_scores, valid_scores)


def layers_validation_curve(data, base_estimator, n_jobs=-3):
    print("***Pré-traitement des donnees***")

    # On récupere toutes les donnees utilisables
    data_usable = BDDminimal(data)

    # On récupére les features et les label
    x, y = featuresLabel(data_usable)

    print("[Separation en set d'entrainement et de test]")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True)

    # Scale + encodage
    scaling(x_train, x_test)

    transformer = LabelEncoder()
    y_train_encode = transformer.fit_transform(y_train).ravel()

    print("[Preparation pour cross-validation]")

    # Separation base test / validation
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    # ----- Paramétrage de la validation curve -----

    # Choix du paramètre et de son intervalle de recherche
    param_range = range(1, 10)
    hidden_layers_range = [tuple([60 for _ in range(i)]) for i in param_range]
    # param_range = range(10, 171, 10)
    # hidden_layers_range = [(i,i,i,i,i) for i in param_range]

    # Calcul de la validation curve
    train_scores, valid_scores = validation_curve(base_estimator, x_train, y_train_encode,
                                                  "hidden_layer_sizes", hidden_layers_range,
                                                  cv=cv, n_jobs=n_jobs)

    # Tracé de la validation curve
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(valid_scores, axis=1)
    validation_scores_std = np.std(valid_scores, axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Validation curve on number of layers for SGD")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Score")

    lw = 2
    ax.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    ax.plot(param_range, validation_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    ax.fill_between(param_range, validation_scores_mean - validation_scores_std,
                    validation_scores_mean + validation_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

    return (train_scores, valid_scores)


def trace_VC():
    excel_table = readMultipleCsv('names')

    # === Modèles de départs ===

    # base_estimator = MLPClassifier(solver="sgd", activation='relu', batch_size=250, learning_rate='adaptive',
    #                                learning_rate_init=0.003803490162088419, max_iter=1000, verbose=True,
    #                                hidden_layer_sizes=(50, 30, 57, 51, 56), alpha=0.00579294106283857)

    # base_estimator = MLPClassifier(solver="adam", activation='relu', batch_size=440,
    #                                    learning_rate_init=0.0036049822428060574, max_iter=1000, verbose=True,
    #                                    hidden_layer_sizes= (31, 56, 58, 41), alpha=0.006600250942968936)

    # === Modèles standardisés ===

    base_estimator = MLPClassifier(solver="sgd", activation='relu', batch_size=250, learning_rate='adaptive',
                                   learning_rate_init=0.003803490162088419, max_iter=1000, verbose=True,
                                   hidden_layer_sizes=(60, 60, 60, 60), alpha=1e-5)

    # base_estimator = MLPClassifier(solver="adam", activation='relu', batch_size=440,
    #                                    learning_rate_init=0.0036049822428060574, max_iter=1000, verbose=True,
    #                                    hidden_layer_sizes= (60, 60, 60, 60), alpha=1e-5)

    # === Traçage des learning curves et recherche des scores moyens maximaux ===

    # train_scores, valid_scores = alpha_validation_curve(excel_table, base_estimator)
    train_scores, valid_scores = layers_validation_curve(excel_table, base_estimator)
    valid_mean = np.mean(valid_scores, axis=1)

    # arg_max = np.argmax(valid_mean)
    # print(np.logspace(-11, 0, 55)[arg_max])
    # print(valid_mean[arg_max])

    arg_max = np.argmax(valid_mean)
    print(range(1, 10)[arg_max])
    print(valid_mean[arg_max])
    
    
# Entraine SVC et affiche la precision
def training_SVC(x_train, y_train, x_test, y_test):
    model_SVC = SVC(kernel='linear', gamma='scale', shrinking=False)
    # Entrainement
    scalerX = StandardScaler()
    X_train_scaled = scalerX.fit_transform(x_train)
    X_test_scaled = scalerX.transform(x_test)
    encoder = LabelEncoder()
    y_train_encode = encoder.fit_transform(y_train)
    y_test_encode = encoder.transform(y_test)
    model_SVC.fit(X_train_scaled, y_train_encode)
    # calcul de précision
    print(f'precision SVC de: {model_SVC.score(X_test_scaled, y_test_encode)*100} %')

# Entraine Kneighbors et affiche la precision
def training_kneighbors(x_train, y_train, x_test, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    print(f'precision KNeighborsClassifier avec {k} voisins de: {model.score(x_test, y_test) *100} %')


# Entraine SGDClassifier et affiche la precision
def training_SGDClassifier(x_train, y_train, x_test, y_test):
    model = SGDClassifier(random_state=0)
    model.fit(x_train, y_train)
    print(f'precision SGDClassifier de: {model.score(x_test, y_test) *100} %')
    

def score_par_type_de_sol():
    """
    Fonction qui renvoie les graphes d'apprentissage par type de sol (spécifier le model dans la fonction)
    """
    
    excel_table = readMultipleCsv('names')
    data_usable = BDDminimal(excel_table)

    x, y = featuresLabel(data_usable)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True, random_state=6)

    # Préparation de la base test

    scalerX = StandardScaler()
    x_train1 = scalerX.fit_transform(x_train)
    x_test = scalerX.transform(x_test)
    data_test = pd.DataFrame(x_test, columns=["z", "VIA", "Po", "Pi", "Cr", "Pr"])
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
    data = pd.DataFrame(x_train1, columns=["z", "VIA", "Po", "Pi", "Cr", "Pr"])
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

    proportion = np.linspace(41, 100, 120)
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

        x_train1 = np.delete(x_train_def, 6, 1)

        # Choix du modèle

        '''model = MLPClassifier(solver="sgd", activation='relu', batch_size=250, learning_rate='adaptive',
                              learning_rate_init=0.003803490162088419, max_iter=1000, verbose=True,
                              hidden_layer_sizes=(60, 60, 60, 60), alpha=1e-6)'''
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(x_train1, y_train1)

        # Evaluation du score

        score_sol = []

        for i in range(len(clefs)):
            x_test1 = Dico_sol_test[clefs[i]]
            if len(x_test1) == 0:
                score_sol.append(0)
            else:
                x_test1 = np.delete(x_test1, 6, 1)
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

# Exemple de tests concernant sur les classifieurs
if __name__ == '__main__':
    excel_table = readMultipleCsv('names')
    data_usable = BDDminimal(excel_table)
    x, y = featuresLabel(data_usable)
    model = MLPClassifier(solver="sgd", activation='relu', batch_size=250, learning_rate='adaptive', learning_rate_init = 0.003803490162088419, max_iter = 1000, verbose = True, hidden_layer_sizes = (60, 60, 60, 60), alpha = 1e-6)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True)

    # transformer = LabelEncoder()
    # y_train_encode = transformer.fit_transform(y_train).ravel()
    # y_test_encode = transformer.transform(y_test).ravel()

    scalerX = StandardScaler()
    x_train = scalerX.fit_transform(x_train)
    x_test = scalerX.transform(x_test)

    model.fit(x_train, y_train)
    model.score(x_test, y_test)

    # data_ivry = pd.read_csv('BDD_CSV/Arthenay_SD5.csv')
    #
    # # data_ivry.drop('Unnamed: 0', axis=1, inplace=True)
    # # data_ivry[data_ivry == 0] = np.nan
    # data_ivry.drop('Vr', axis=1, inplace=True)
    # data_ivry.drop('Unnamed: 0', axis=1, inplace=True)
    # # data_ivry.dropna(axis=0, inplace=True)
    # # data_ivry.reset_index(drop=True, inplace=True)
    #
    # y_data_ivry = data_ivry['sol']
    # # y_data_ivry_encode = transformer.transform(y_data_ivry).ravel()
    # x_data_ivry = data_ivry.drop('sol', axis=1)
    # x_data_ivry = scalerX.transform(x_data_ivry)
    #
    # SortiePhoto(x_data_ivry, y_data_ivry, model)

