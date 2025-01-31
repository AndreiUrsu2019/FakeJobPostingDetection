from sklearn.metrics import adjusted_rand_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from itertools import product
from os import fwalk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import OneClassSVM

# Citim seturile de date
anunturi_reale = pd.read_csv('/content/drive/MyDrive/Practical_ML/Proiect_2/Real_Job_Postings.csv')
anunturi_false = pd.read_csv('/content/drive/MyDrive/Practical_ML/Proiect_2/Fake_Job_Postings.csv')

# Procesăm anunțurile false
anunturi_false = anunturi_false[['description']]
anunturi_false = anunturi_false.rename(columns={'description': 'descriere'})

# Calculăm lungimile pentru anunțurile false
anunturi_false["numar_cuvinte"] = anunturi_false["descriere"].apply(lambda x: len(x.split()))
anunturi_false["numar_caractere"] = anunturi_false["descriere"].apply(len)

# Grafic pentru numărul de cuvinte
plt.figure(figsize=(10, 5))
sns.histplot(anunturi_false["numar_cuvinte"], kde=True, bins=10, color='blue', edgecolor='black')
plt.title("Distribuția Numărului de Cuvinte", fontsize=16)
plt.xlabel("Număr de Cuvinte", fontsize=14)
plt.ylabel("Frecvență", fontsize=14)
plt.grid(axis="y")
plt.show()

# Grafic pentru numărul de caractere
plt.figure(figsize=(10, 5))
sns.histplot(anunturi_false["numar_caractere"], kde=True, bins=10, color='green', edgecolor='black')
plt.title("Distribuția Numărului de Caractere", fontsize=16)
plt.xlabel("Număr de Caractere", fontsize=14)
plt.ylabel("Frecvență", fontsize=14)
plt.grid(axis="y")
plt.show()

# Calculăm lungimile pentru anunțurile reale
anunturi_reale["numar_cuvinte"] = anunturi_reale["descriere"].apply(lambda x: len(x.split()))
anunturi_reale["numar_caractere"] = anunturi_reale["descriere"].apply(len)
anunturi_reale['numar_cuvinte'] = anunturi_reale.query('numar_cuvinte <= 200')["numar_cuvinte"]
anunturi_reale['numar_caractere'] = anunturi_reale.query('numar_caractere <= 200')["numar_caractere"]

# Grafic pentru numărul de cuvinte
plt.figure(figsize=(10, 5))
sns.histplot(anunturi_reale["numar_cuvinte"], kde=True, bins=10, color='blue', edgecolor='black')
plt.title("Distribuția Numărului de Cuvinte", fontsize=16)
plt.xlabel("Număr de Cuvinte", fontsize=14)
plt.ylabel("Frecvență", fontsize=14)
plt.grid(axis="y")
plt.show()

# Grafic pentru numărul de caractere
plt.figure(figsize=(10, 5))
sns.histplot(anunturi_reale["numar_caractere"], kde=True, bins=10, color='green', edgecolor='black')
plt.title("Distribuția Numărului de Caractere", fontsize=16)
plt.xlabel("Număr de Caractere", fontsize=14)
plt.ylabel("Frecvență", fontsize=14)
plt.grid(axis="y")
plt.show()

# Combinăm seturile de date
set_date_complet = pd.concat([anunturi_reale, anunturi_false])
set_date_complet.to_csv('set_date_complet.csv')

# Încărcăm și preprocesăm setul de date
set_date_complet = shuffle(set_date_complet)  # Asigurăm randomizarea
descrieri = set_date_complet['descriere']
etichete = set_date_complet['fraudulos']

# 1. Reprezentări ale caracteristicilor
# a) Reprezentare TF-IDF
vectorizator_tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
caracteristici_tfidf = vectorizator_tfidf.fit_transform(descrieri).toarray()

# b) Încorporări de cuvinte folosind GloVe pre-antrenat
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")

def incorporeaza_propozitie(propozitie):
    cuvinte = propozitie.split()
    vectori = [glove[cuvant] for cuvant in cuvinte if cuvant in glove]
    return np.mean(vectori, axis=0) if vectori else np.zeros(50)

incorporari = np.array([incorporeaza_propozitie(propozitie) for propozitie in descrieri])

# 2. Clustering cu One-Class SVM și Affinity Propagation
lista_caracteristici = [caracteristici_tfidf, incorporari]
nume_caracteristici = ["TF-IDF", "GloVe"]

def evalueaza_model(caracteristici, etichete, nume):
    # Standardizăm caracteristicile pentru SVM
    standardizator = StandardScaler()
    caracteristici_standardizate = standardizator.fit_transform(caracteristici)

    # Împărțim în set de antrenare și testare
    X_antrenare, X_testare, y_antrenare, y_testare = train_test_split(
        caracteristici_standardizate, etichete, test_size=0.3, random_state=42
    )
    y_testare = y_testare.reset_index(drop=True)

    # Definim grilele de parametri
    parametri_svm = {
        'nu': [0.01, 0.05, 0.1, 0.2],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.1, 1.0]
    }

    parametri_padure = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # One-Class SVM cu Căutare în Grilă
    print(f"\nEfectuăm Căutarea în Grilă pentru One-Class SVM pe caracteristicile {nume}...")
    svm = OneClassSVM()
    cautare_svm = GridSearchCV(
        svm,
        parametri_svm,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    cautare_svm.fit(X_antrenare)

    print("\nCei mai buni parametri SVM:", cautare_svm.best_params_)
    print("Cel mai bun scor SVM:", cautare_svm.best_score_)

    # Obținem predicțiile folosind cel mai bun model
    predictii = cautare_svm.best_estimator_.predict(X_testare)
    predictii_SVM = np.array([1 if y == 1 else 0 for y in predictii])

    # Indici clasificați greșit pentru SVM
    indici_gresiti = np.where(predictii_SVM != y_testare.values)[0]
    print(f"\nPrimele 10 intrări clasificate greșit pentru {nume} (SVM):")
    for idx in indici_gresiti[:10]:
        print(f"Index {idx}: Caracteristici = {caracteristici[idx]}, Etichetă Reală = {y_testare[idx]}")

    print("\nPerformanța One-Class SVM:")
    print("Acuratețe:", accuracy_score(y_testare, predictii_SVM))
    print("Raport de Clasificare:\n", classification_report(y_testare, predictii_SVM))

    # Vizualizarea rezultatelor SVM
    reductor_dim = PCA(n_components=2)
    X_testare_2d = reductor_dim.fit_transform(X_testare)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(
        X_testare_2d[predictii_SVM == 1, 0], X_testare_2d[predictii_SVM == 1, 1],
        c='blue', label='Inliers (SVM)', alpha=0.6
    )
    plt.scatter(
        X_testare_2d[predictii_SVM == 0, 0], X_testare_2d[predictii_SVM == 0, 1],
        c='red', label='Outliers (SVM)', alpha=0.6
    )
    plt.title(f"Clustere SVM ({nume})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    # Random Forest cu Căutare în Grilă
    print(f"\nEfectuăm Căutarea în Grilă pentru Random Forest pe caracteristicile {nume}...")
    padure = RandomForestClassifier(random_state=42)
    cautare_padure = GridSearchCV(
        padure,
        parametri_padure,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    cautare_padure.fit(X_antrenare, y_antrenare)

    print("\nCei mai buni parametri Random Forest:", cautare_padure.best_params_)
    print("Cel mai bun scor Random Forest:", cautare_padure.best_score_)

    # Obținem predicțiile folosind cel mai bun model
    predictii_padure = cautare_padure.best_estimator_.predict(X_testare)

    # Indici clasificați greșit pentru RF
    indici_gresiti_padure = np.where(predictii_padure != y_testare)[0]
    print(f"\nPrimele 10 intrări clasificate greșit pentru {nume} (Random Forest):")
    for idx in indici_gresiti_padure[:10]:
        print(f"Index {idx}: Caracteristici = {X_testare[idx]}, Etichetă Reală = {y_testare[idx]}, Prezis: {predictii_padure[idx]}")

    print("\nPerformanța Random Forest:")
    print(classification_report(y_testare, predictii_padure))

    # Vizualizarea rezultatelor RF
    plt.subplot(1, 2, 2)
    plt.scatter(
        X_testare_2d[predictii_padure == 1, 0], X_testare_2d[predictii_padure == 1, 1],
        c='blue', label='Clasa 1 (RF)', alpha=0.6
    )
    plt.scatter(
        X_testare_2d[predictii_padure == 0, 0], X_testare_2d[predictii_padure == 0, 1],
        c='red', label='Clasa 0 (RF)', alpha=0.6
    )
    plt.title(f"Clustere Random Forest ({nume})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Compararea performanței
    rezultate = {
        'SVM': accuracy_score(y_testare, predictii_SVM),
        'Random Forest': accuracy_score(y_testare, predictii_padure),
        'Șansă Aleatoare': 0.5
    }

    plt.figure(figsize=(8, 6))
    plt.bar(rezultate.keys(), rezultate.values(), color=['blue', 'green', 'red'])
    plt.title(f"Comparație Performanță ({nume})")
    plt.ylabel("Acuratețe")
    plt.ylim(0, 1)
    plt.show()

    # Returnăm cei mai buni parametri și scorurile lor
    return {
        'parametri_svm': cautare_svm.best_params_,
        'scor_svm': cautare_svm.best_score_,
        'parametri_padure': cautare_padure.best_params_,
        'scor_padure': cautare_padure.best_score_
    }

rezultate = {}
for caracteristici, nume in zip(lista_caracteristici, nume_caracteristici):
    print(f"\nEvaluăm caracteristicile {nume}...")
    rezultate[nume] = evalueaza_model(caracteristici, etichete, nume)

# Afișăm sumarul parametrilor optimi pentru fiecare set de caracteristici
print("\nSumar al Parametrilor Optimi:")
for nume, rezultat in rezultate.items():
    print(f"\n{nume}:")
    print("Parametri Optimi SVM:", rezultat['parametri_svm'])
    print("Scor Optim SVM:", rezultat['scor_svm'])
    print("Parametri Optimi Random Forest:", rezultat['parametri_padure'])
    print("Scor Optim Random Forest:", rezultat['scor_padure'])
    
def evalueaza_model_af(caracteristici, etichete, nume):
    # Standardizăm caracteristicile pentru clustering
    standardizator = StandardScaler()
    caracteristici_standardizate = standardizator.fit_transform(caracteristici)

    # Împărțim în set de antrenare și testare
    X_antrenare, X_testare, y_antrenare, y_testare = train_test_split(
        caracteristici_standardizate, etichete, test_size=0.3, random_state=42
    )

    # Parametri pentru căutarea în grilă
    valori_damping = np.arange(0.5, 0.95, 0.1)
    valori_preferinta = [-100, -50, -10]

    # Parametri pentru Random Forest
    parametri_padure = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Stocăm rezultatele
    cel_mai_bun_ari = -1
    cei_mai_buni_parametri = None
    cele_mai_bune_etichete = None
    dictionar_rezultate = {}

    print(f"\nEfectuăm căutarea în grilă pentru Affinity Propagation pe {nume}...")

    # Căutare în grilă pentru Affinity Propagation
    for damping, preferinta in product(valori_damping, valori_preferinta):
        try:
            model_afinitate = AffinityPropagation(
                damping=damping,
                preference=preferinta,
                random_state=42,
                max_iter=500
            )
            model_afinitate.fit(X_antrenare)

            # Calculăm scorul ARI
            scor_ari = adjusted_rand_score(y_antrenare, model_afinitate.labels_)

            dictionar_rezultate[(damping, preferinta)] = {
                'ari': scor_ari,
                'nr_clustere': len(np.unique(model_afinitate.labels_))
            }

            if scor_ari > cel_mai_bun_ari:
                cel_mai_bun_ari = scor_ari
                cei_mai_buni_parametri = {'damping': damping, 'preference': preferinta}
                cele_mai_bune_etichete = model_afinitate.labels_
                cel_mai_bun_model = model_afinitate

        except Exception as e:
            print(f"Eroare cu damping={damping}, preferință={preferinta}: {str(e)}")

    print("\nCei mai buni parametri Affinity Propagation:", cei_mai_buni_parametri)
    print("Cel mai bun scor ARI:", cel_mai_bun_ari)
    print("Număr de clustere:", len(np.unique(cele_mai_bune_etichete)))

    # Căutare în grilă pentru Random Forest
    print(f"\nEfectuăm căutarea în grilă pentru Random Forest pe {nume}...")
    padure = RandomForestClassifier(random_state=42)
    cautare_padure = GridSearchCV(
        padure,
        parametri_padure,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    cautare_padure.fit(X_antrenare, y_antrenare)

    print("\nCei mai buni parametri Random Forest:", cautare_padure.best_params_)
    print("Cel mai bun scor Random Forest:", cautare_padure.best_score_)

    # Obținem predicții folosind cele mai bune modele
    predictii_supervizate = cautare_padure.predict(X_testare)
    predictii_ap = cel_mai_bun_model.predict(X_testare)

    # Identificăm intrările clasificate greșit
    clasificari_gresite_supervizate = np.where(predictii_supervizate != y_testare)[0]
    print(f"\nPrimele 10 intrări clasificate greșit pentru Random Forest:")
    for idx in clasificari_gresite_supervizate[:10]:
        print(f"Index {idx}: Caracteristici = {X_testare[idx]}, Prezis = {predictii_supervizate[idx]}, Real = {y_testare[idx]}")

    print(f"\nPrimele 10 atribuiri de clustere pentru Affinity Propagation:")
    for idx in range(10):
        print(f"Index {idx}: Caracteristici = {X_testare[idx]}, Cluster = {predictii_ap[idx]}, Real = {y_testare[idx]}")

    # Vizualizarea performanței
    rezultate = {
        'Affinity Propagation (ARI)': cel_mai_bun_ari,
        'Random Forest (Acuratețe)': accuracy_score(y_testare, predictii_supervizate),
        'Șansă Aleatoare': 0.25
    }

    plt.figure(figsize=(10, 6))
    plt.bar(rezultate.keys(), rezultate.values(), color=['blue', 'orange', 'green'])
    plt.title(f"Comparație Performanță ({nume})")
    plt.ylabel("Scor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Vizualizarea performanței căutării în grilă
    plt.figure(figsize=(12, 6))
    scoruri_damping = {}
    for (damping, pref), rezultat in dictionar_rezultate.items():
        if damping not in scoruri_damping:
            scoruri_damping[damping] = []
        scoruri_damping[damping].append(rezultat['ari'])

    for damping, scoruri in scoruri_damping.items():
        plt.boxplot(scoruri, positions=[damping], widths=0.05)
    plt.title(f"Scoruri ARI vs Parametrul Damping ({nume})")
    plt.xlabel("Damping")
    plt.ylabel("Scor ARI")
    plt.show()

    # Vizualizare PCA
    pca = PCA(n_components=2)
    X_testare_pca = pca.fit_transform(X_testare)

    # Plotăm clusterele alături
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Clustering Model
    scatter1 = ax1.scatter(X_testare_pca[:, 0], X_testare_pca[:, 1],
                          c=predictii_ap, cmap='viridis', s=50, alpha=0.7)
    ax1.set_title(f"Clustere Affinity Propagation\n(damping={cei_mai_buni_parametri['damping']})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    plt.colorbar(scatter1, ax=ax1, label="Cluster")

    # Etichete Reale
    scatter2 = ax2.scatter(X_testare_pca[:, 0], X_testare_pca[:, 1],
                          c=y_testare, cmap='viridis', s=50, alpha=0.7)
    ax2.set_title("Etichete Reale")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    plt.colorbar(scatter2, ax=ax2, label="Etichetă Reală")

    plt.tight_layout()
    plt.show()

    return {
        'parametri_ap': cei_mai_buni_parametri,
        'scor_ari': cel_mai_bun_ari,
        'parametri_padure': cautare_padure.best_params_,
        'scor_padure': cautare_padure.best_score_
    }

# Funcție pentru clustering K-Means
def clustering_kmeans_cu_ari(caracteristici, nr_clustere, nume_caracteristici, etichete_reale):
    kmeans = KMeans(n_clusters=nr_clustere, random_state=42)
    etichete_cluster = kmeans.fit_predict(caracteristici)

    # Calculăm Indicele Rand Ajustat (ARI)
    scor_ari = adjusted_rand_score(etichete_reale, etichete_cluster)
    print(f"Indice Rand Ajustat (ARI) pentru {nume_caracteristici}: {scor_ari}")

    return etichete_cluster, scor_ari

rezultate = {}
for caracteristici, nume in zip(lista_caracteristici, nume_caracteristici):
    print(f"\nEvaluăm caracteristicile {nume}...")
    rezultate[nume] = evalueaza_model_af(caracteristici, etichete, nume)

# Afișăm sumarul parametrilor optimi pentru fiecare set de caracteristici
print("\nSumar al Parametrilor Optimi:")
for nume, rezultat in rezultate.items():
    print(f"\n{nume}:")
    print("Parametri Optimi Affinity Propagation:", rezultat['parametri_ap'])
    print("Cel mai bun scor ARI:", rezultat['scor_ari'])
    print("Parametri Optimi Random Forest:", rezultat['parametri_padure'])
    print("Cel mai bun scor de acuratețe:", rezultat['scor_padure'])

# Număr de clustere (ajustați în funcție de setul de date)
nr_clustere = 2  # Pentru clustering binar

# K-Means cu caracteristici TF-IDF
print("Clustering folosind caracteristici TF-IDF:")
clustere_tfidf, ari_tfidf = clustering_kmeans_cu_ari(caracteristici_tfidf, nr_clustere, "TF-IDF", etichete)

# K-Means cu încorporări GloVe
print("\nClustering folosind încorporări GloVe:")
clustere_glove, ari_glove = clustering_kmeans_cu_ari(incorporari, nr_clustere, "GloVe", etichete)

# Comparație între rezultatele TF-IDF și GloVe
print(f"\nComparație scoruri ARI:")
print(f"ARI TF-IDF: {ari_tfidf}")
print(f"ARI GloVe: {ari_glove}")

# Funcție pentru vizualizarea rezultatelor
def vizualizeaza_clustere_kmeans(caracteristici, etichete_cluster, nume_caracteristici):
    plt.scatter(caracteristici[:, 0], caracteristici[:, 1], c=etichete_cluster, cmap='viridis', s=10)
    plt.title(f"Clustering K-Means cu {nume_caracteristici}")
    plt.xlabel("Caracteristica 1")
    plt.ylabel("Caracteristica 2")
    plt.show()

# Vizualizare pentru TF-IDF
pca_tfidf = PCA(n_components=2).fit_transform(caracteristici_tfidf)
vizualizeaza_clustere_kmeans(pca_tfidf, clustere_tfidf, "TF-IDF")

# Vizualizare pentru GloVe
pca_glove = PCA(n_components=2).fit_transform(incorporari)
vizualizeaza_clustere_kmeans(pca_glove, clustere_glove, "GloVe")
