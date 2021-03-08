import pandas as pd
import os


def excelToCsv(path):
    """Saves each sheet of an excel table in a .csv file and writes the files' names in the text file named 'names'.
    Return the number of sheets."""
    data = pd.read_excel("BDD_triee_cor.xlsx", sheet_name=None, na_values=None)

    # On ouvre un fichier texte pour y écrire tous les noms des nouveaux fichiers csv
    names_file = open("BDD_CSV/names", 'w')

    if type(data) != dict:
        # Dans ce cas il n'y a qu'une feuille

        data.to_csv("BDD_CSV/BDD.csv")
        names_file.write("BDD.csv")

        names_file.close()
        return 1

    # Dans ce cas, il y a plusieurs feuilles
    names_text = ""
    for i, sheet_name in enumerate(list(data.keys())):
        # Sauvegarde du nom
        if i + 1 < len(list(data.keys())):
            names_text += f'{sheet_name}.csv\n'
        else:
            names_text += f'{sheet_name}.csv'

        # Sauvegarde de la feuille
        data[sheet_name].to_csv(f'BDD_CSV/{sheet_name}.csv')

    names_file.write(names_text)
    names_file.close()
    return len(list(data.keys()))


def readMultipleCsv(names_file):
    """Needs the name of the file which has all the names of the files we want to import.
    Return a dict of dataframes."""

    # On récupère le nom des fichiers qu'on veut importer
    names_to_import = []
    names_file = open(f'BDD_CSV/{names_file}', 'r')
    for line in names_file.readlines():
        names_to_import.append(line.rstrip())
    names_file.close()

    # On importe tous les fichiers dans un dict
    data = {}
    for name in names_to_import:
        data[name] = pd.read_csv(f'BDD_CSV/{name}')
        data[name].drop('Unnamed: 0', axis = 1, inplace = True)

    return data


def saveCsv(dict_data, names_file):
    """
    Remove files which are on the list.
    Save all the data on the files and update the 'names' file
    """

    assert type(dict_data) == dict

    for name in dict_data.keys():
        try:
            os.remove(f'BDD_CSV/{name}')
        except:
            continue

    # On récupère le nom des fichiers déjà enregistrés
    names_saved = []
    names_file = open(f'BDD_CSV/{names_file}', 'r+')
    for line in names_file.readlines():
        names_saved.append(line.rstrip())

    # On met à jour les noms des fichiers dans la BDD
    for name in dict_data.keys():
        if name not in names_saved:
            names_saved.append(name)
    names_text = ""
    for i, name in enumerate(names_saved):
        if i + 1 < len(names_saved):
            names_text += f'{name}\n'
        else:
            names_text += f'{name}'

    names_file.seek(0)
    names_file.write(names_text)
    names_file.truncate()
    names_file.close()

    # On enregistre
    for name, data in dict_data.items():
        data.to_csv(f'BDD_CSV/{name}')
