'''
QUESTO SCRIPT HA LA FUNZIONE DI CONFRONTARE LA DIVERSA EFFICACIA DEI METODI DI RACCOMANDAZIONE NEAREST NEIGHBOR CLASSIFIER E BAYES CLASSIFIER
'''

import time
import nltk
import math
import pickle
import random
import multiprocessing as mp
import numpy as np
from tabulate import tabulate

'''
Crea il dizionario, Users, per salvare gli User. Le chiavi saranno gli UserID, a cui corrisponderà un DIZIONARIO. Le chiavi di questo
DIZIONARIO saranno "Liked", a cui corrisponderà un INSIEME di articoli che allo user piacciono, "Disliked", a cui corrisponderà un INSIEME
di articoli che allo user non piacciono, e "Test", a cui corrisponde un insieme di newsID che verranno studiate per la raccomandazione.
- Users[UserID]["Test"] è sottoinsieme di Users[UserID]["Liked"] | Users[UserID]["Disliked"].
'''
Users=dict()
users_dict = open('Users_sample.pkl', 'rb')
Users = pickle.load(users_dict)

'''
Creo il dizionario, NewsArchive, per salvare gli elementi degli articoli. Le chiavi saranno le NewsID, a cui corrisponderà un DIZIONARIO. Le chiavi di questo
DIZIONARIO saranno "Titolo","Genere","Sottogenere","Url" e "TF_IDF". Le prime corrispondono a stringhe, l'ultima è associata ad un dizionario termine-tf-idf score.
Una chiave che verrà definita in seguito è "Features", a cui corrisponderà l'insieme di termini che faranno da descrittori per l'articolo.
'''
NewsArchive=dict()
news_dict = open('News.pkl', 'rb')
NewsArchive = pickle.load(news_dict)


'''
PARAMETRI
'''

#Numero di features massime da tenere per ogni documento, scelte tramite tf-idf.
nFeat = 100
#Proporzione di features da tenere nel caso di supervised feature selection attraverso il metodo del X2.
retain = 0.5


#Numero di documenti da considerare per ciasun articolo nella Nearest-Neighbor-classification (numero di "vicini")
top_k = 20
#Frazione di cluster da creare in relazione all'insieme di partenza (0.5 -> Gli articoli vengono raggruppati nella metà dei cluster (es. # articoli = 100 # cluster = 50))
clusterFactor = 0.1
#Numero di iterazioni massime per il k-means clustering
nIter = 4
#Numero di documenti da considerare per ciascun articolo nella Nearest Neighbor Classification in caso di clustering (numero di "vicini")
top_k_cluster = 3



'''
UTILITY
'''

#Funzione che stampa uno User
def print_User (User):
    print("\nID dello User: {}".format(User))
    print("\n\tLiked: {}".format(Users[User]["Liked"]))
    print("\n\tDisliked: {}".format(Users[User]["Disliked"]))
    print("\n\tTest: {}".format(Users[User]["Test"]))

#Funzione che stampa tutti gli User
def print_Users():
    for User in Users:
        print_User(User)

#Funzione che stampa un articolo
def print_Article (ArticleID):
    print("\nID dell'articolo: {}".format(ArticleID))
    print("\n\tCategoria: {} /\tSottocategoria: {}".format(NewsArchive[ArticleID]["Category"],NewsArchive[ArticleID]["Subcategory"]))
    print("\n\tTitolo: {}".format(NewsArchive[ArticleID]["Title"]))
    print("\n\tUrl ->\t {}".format(NewsArchive[ArticleID]["Url"]))
    print("\nFeatures: {}".format(NewsArchive[ArticleID]["Features"]))

#Funzione che stampa tutte le News
def printNews():
    for ArticleID in NewsArchive:
        print_Article(ArticleID)

#Stampa i parametri specificati inizialmente
def print_parameters():
    print("\n -PARAMETERS- ")
    print("\n Number of features = {} / retained = {} %".format(nFeat,retain*100))
    print(" --- Nearest Neighbor Classifier : top_k = {} ({}) / clusterFactor = {}".format(top_k,top_k_cluster,clusterFactor))

#Stampa delle informazioni sui cluster
def print_clusters(Clusters):
    for cluster in Clusters:
        print("--- Cluster {}: {} items".format(cluster, len(Clusters[cluster]["Items"])))


'''
FEATURE SELECTION
'''

#Feature selection, per ogni articolo, dopo aver pesato per alpha.
def feature_Selection (alpha):

    print("\nSelezionando le features per alpha = {} ...".format(alpha))

    #Il peso non deve essere modificato per alpha == 0.
    if alpha != 0:
        for ArticleID in NewsArchive.keys():

            #Voglio ridurre il peso dei termini che non compaiono in titolo, categoria e sottocategoria. Crea una lista con le parole di titolo, categoria e sotto-categoria.
            heavy_terms = NewsArchive[ArticleID]["Title"].split()
            heavy_terms.append(NewsArchive[ArticleID]["Category"])
            heavy_terms.append(NewsArchive[ArticleID]["Subcategory"])

            for term in NewsArchive[ArticleID]["TF_IDF"]:
                #Il peso viene ridotto per alpha se la parola appartiene a corpo o abstract.
                if term not in heavy_terms:
                    NewsArchive[ArticleID]["TF_IDF"][term] = NewsArchive[ArticleID]["TF_IDF"][term]*(1-alpha)

    #Creo una nuova chiave, a cui corrisponde un insieme di termini, necessaria per ricordare le features del documento.
    for ArticleID in NewsArchive.keys():
        NewsArchive[ArticleID]["Features"] = set()
        count = 0

        #Ordina i termini in ordine decrescente di tf-idf score.
        for term in sorted(NewsArchive[ArticleID]["TF_IDF"],key=NewsArchive[ArticleID]["TF_IDF"].get,reverse=True):

            #Nel caso alpha = 1, non avrebbe senso tenere anche termini del corpo e dell'abstract, che a quel punto avrebbero punteggio zero (e sarebbero selezionati casualmente).
            if NewsArchive[ArticleID]["TF_IDF"][term] == 0:
                break
            if count < nFeat:
                NewsArchive[ArticleID]["Features"].add(term)
                count += 1
            else:
                break

    print("Features selezionate.\n")

#De-Selection. In feature selection gli score TF-IDF vengono modificati. Questa funzione riporta gli score ai valori originali ricaricando il dizionario News.pkl (dividere i termini per 1- alpha non
#funzionava correttamente). Questo è necessario per ripetere la raccomandazione per più configurazioni di features.
def feature_Deselection ():

    print("\nRiportando i pesi ai valori originali ...".format())

    temp_dict = open('News.pkl', 'rb')
    temp_dict = pickle.load(temp_dict)

    for article in NewsArchive:
        NewsArchive[article]["TF_IDF"] = temp_dict[article]["TF_IDF"]
        NewsArchive[article]["Features"] = set()

    print("Punteggi sistemati.\n")

#Specificato un articolo in input riduce il numero di features utilizzate per la sua rappresentazione tramite il metodo X2. nLiked e nDisliked sono le dimensioni del training set (piaciuti e non piaciuti),
#freq il dizionario delle frequenze delle singole features in un training set.
def reducer_X2(ArticleID,freq,nLiked,nDisliked):

    #REFERENCE PAG. 149 AGGARWAL

    #Numero di features da tenere (dipende dal numero di features e dal parametro specificato inizialmente):
    nFeats = int(len(NewsArchive[ArticleID]["Features"])*retain)

    #Dizionario in cui verranno salvati i valori X2 per ogni feature dell'articolo
    x2_scores = dict()

    for feature in NewsArchive[ArticleID]["Features"]:

        try:
            #Occorrenze negli articoli piaciuti
            o1 = freq[feature][1]

            #Non-occorrenze negli articoli piaciuti
            o2 = nLiked - o1

            #Occorrenze negli articoli non piaciuti
            o3 = freq[feature][0]

            #Non-occorrenze negli articoli non piaciuti
            o4 = nDisliked - o3

            #Punteggio X2 per il termine (formula (4.6) pag. 149)
            x2_scores[feature] = ((o1 + o2 + o3 + o4)*((o1*o4 - o2*o3)**2)) / ((o1 + o2)*(o3 + o4)*(o1 + o3)*(o2 + o4))

        except:

            #Nel caso in cui vi siano key errors, vuol dire che la feature non è presente nel training set (compare solo nel test set), ne tra i liked ne tra i disliked. Non sarà quindi influente nella
            #determinazione della probabilità.
            x2_scores[feature] = 0

    count = 0
    NewsArchive[ArticleID]["Features"] = set()

    for feature in sorted(x2_scores,key=x2_scores.get,reverse=True):
        if count >= nFeats:
            break
        count += 1
        NewsArchive[ArticleID]["Features"].add(feature)

#Questa funzione prende in input due insiemi di articoli, quelli piaciuti e non piaciuti, e restituisce un dizionario contenente le frequenze per le features che vi compaiono.
def get_freq (Liked, Disliked):

    #Dizionario che conterrà le frequenze delle features negli articoli con rating 1 e con rating 0.
    freq = dict()

    #Calcola le occorrenze delle features negli articoli piaciuti e non del TRAINING SET ("Liked" - "Test" e "Disliked" - "Test")
    for article in Liked:
        for feature in NewsArchive[article]["Features"]:
            try:
                freq[feature][1] += 1
            except:
                #Memorizza le occorrenze negli articoli graditi (1), ma anche non graditi (0)
                freq[feature] = {1:1,0:0}

    for article in Disliked:
        for feature in NewsArchive[article]["Features"]:
            try:
                freq[feature][0] += 1
            except:
                freq[feature] = {1:0,0:1}

    return freq

'''
FUNZIONI NEAREST NEIGHBOR CLASSIFIER
'''

#Calcola la distanza cosena tra due articoli usando le loro features
def cosine_distance (ArticleID1,ArticleID2):
    num = 0
    lenght_1 = 0
    lenght_2 = 0

    #Lunghezza del primo articolo
    for feature in NewsArchive[ArticleID1]["Features"]:
        lenght_1 += NewsArchive[ArticleID1]["TF_IDF"][feature]**2

    #Lunghezza del secondo articolo
    for feature in NewsArchive[ArticleID2]["Features"]:
        lenght_2 += NewsArchive[ArticleID2]["TF_IDF"][feature]**2


    features_in_common = NewsArchive[ArticleID1]["Features"] & NewsArchive[ArticleID2]["Features"]

    #Le features in comune servono per il numeratore della distanza coseno
    for feature in features_in_common:
        num += NewsArchive[ArticleID1]["TF_IDF"][feature]*NewsArchive[ArticleID2]["TF_IDF"][feature]

    #Restituisce il valore di similarità coseno calcolato
    return num/(math.sqrt(lenght_1)*math.sqrt(lenght_2))

#Distanza cosena da utilizzare per la similarità articolo-centroide.
def cosine_distance_kmeans (ArticleID, centroide):
    num = 0
    lenght_1 = 0
    lenght_2 = 0

    #Lunghezza dell'articolo
    for feature in NewsArchive[ArticleID]["Features"]:
        lenght_1 += NewsArchive[ArticleID]["TF_IDF"][feature]**2

    #Lunghezza del centroide
    for feature in centroide:
        lenght_2 += centroide[feature]**2

    features_in_common = NewsArchive[ArticleID]["Features"] & set(centroide.keys())

    #Le features in comune servono per il numeratore della distanza coseno
    for feature in features_in_common:
        num += NewsArchive[ArticleID]["TF_IDF"][feature]*centroide[feature]

    return num/(math.sqrt(lenght_1)*math.sqrt(lenght_2))

#Restituisce una previsione del rating (tra 0 o 1) per il documento (ArticleID) tramite confronto con il training set (Liked e Disliked).
def neighbor_prediction(Liked,Disliked,ArticleID):

    #Dizionario per memorizzare le distanze coseno calcolate tra ArticleID e gli articoli nel training set
    similarities = dict()

    #Distanze coseno con ogni articolo
    for Article in Liked | Disliked:
        similarities[Article] = cosine_distance(Article,ArticleID)

    #Valore che verrà restituito, il rating previsto per ArticleID
    rating = 0
    denom = 0
    count = 0

    #Scorre gli articoli dal più simile al meno simile
    for Article in sorted(similarities,key=similarities.get,reverse=True):
        if count < top_k:
            #print(similarities[Article])
            if Article in Liked:

                #Il rating previsto è la media ponderata usando le similarità. L'unico rating influente è 1 (l'altro è 0 se l'articolo è nei Disliked).
                rating += similarities[Article]

            #Al denominatore, essendo una media pesata, compare la somma dei pesi (le similarità)
            denom += similarities[Article]
            count += 1
        else:
            break

    try:
        #È possibile che il denominatore sia 0, nel caso in cui l'articolo non abbia alcuna feature in comune con i documenti del training set
        rating = rating/denom
    except:
        #Il rating in questo caso è pari a 0, no features in comune con nessuno degli articoli di training.
        rating = rating

    #Restituisce il rating previsto per ArticleID
    return round(rating,4)

#Nel caso di clustering. Restituisce una previsione del rating (tra 0 o 1) per il documento (ArticleID) tramite confronto con il training set (Liked e Disliked).
def neighbor_prediction_cluster(Liked,Disliked,ArticleID):

    #Dizionario per memorizzare le distanze coseno calcolate tra ArticleID e i cluster
    similarities = dict()

    #Inizio con i cluster di articoli piaciuti
    for cluster in Liked:
        similarities[cluster] = cosine_distance_kmeans (ArticleID , Liked[cluster]["Center"])

    #Calcola le similarità con i cluster di articoli non piaciuti
    for cluster in Disliked:
        similarities[cluster] = cosine_distance_kmeans (ArticleID , Disliked[cluster]["Center"])

    #Valore che verrà restituito, il rating previsto per ArticleID
    rating = 0
    denom = 0
    count = 0

    #Scorre gli articoli dal più simile al meno simile
    for cluster in sorted(similarities,key=similarities.get,reverse=True):
        if count < top_k_cluster:
            if cluster in Liked.keys():

                #Il rating previsto è la media ponderata usando le similarità. L'unico rating influente è 1 (l'altro è 0 se l'articolo è nei Disliked).
                rating += similarities[cluster]

            #Al denominatore, essendo una media pesata, compare la somma dei pesi (le similarità)
            denom += similarities[cluster]
            count += 1
        else:
            break

    try:
        rating = rating/denom
    except:
        #Il rating è 0 in questo caso, se il denominatore (somma delle similarità) è 0
        rating = rating

    #Restituisce il rating previsto per ArticleID
    return round(rating,4)

#Dato un insieme di articoli, ne calcola il centroide (un dizionario feature - punteggio).
def getCenter (items):

    #   PAG. 152 AGGARWAL

    #Dizionario che verrà restituito e che funzionerà da centroide per il cluster. Le chiavi sono le features che compaiono negli articoli dell'insieme,
    #i valori sono la somma degli score tf-idf per quei termini in riferimento agli articoli in cui compaiono.
    center = dict()


    #Crea il centroide
    for article in items:
        for feature in NewsArchive[article]["Features"]:
            try:
                center[feature] +=  NewsArchive[article]["TF_IDF"][feature]
            except:
                center[feature] = NewsArchive[article]["TF_IDF"][feature]

    return center

#Restituisce un insieme di centroidi iniziali per effettuare il clustering. Alternativa alla scelta casuale, si basa sul metodo k-means++.
def starting_clusters (insieme, nClusters):

    #centroide di partenza.
    first = random.sample(insieme,1)[0]
    active_center = first

    #Insieme degli articoli che vanno a costituire i centroidi iniziali.
    starting_centers = [first]

    #Dizionario articolo - lista [cluster più simile , distanza coseno].
    similarities = dict()
    #Inizializza il dizionario.
    for article in insieme:
        similarities[article] = [first,0]

    #Il primo centroide è già stato scelto casualmente
    i = 1

    while i < nClusters:

        #Similarità minore, necessario per identificare il successivo cluster.
        min_s = [first,1]

        #Per ogni articolo da studiare per determinare il successivo cluster:
        for article in insieme - set(starting_centers):
            #Calcola la distanza cosena dall'ultimo centroide trovato
            s = cosine_distance(article,active_center)
            #E' il nuovo centroide più vicino? Lo confronto con il più vicino
            if s > similarities[article][1]:
                #Il centroide più vicino è active_center.
                similarities[article] = [active_center , s]

        for article in insieme - set(starting_centers):

            #Cerca di individuare il punto più lontano dai centroidi già individuati.
            if similarities[article][1] < min_s[1]:
                #Prendo l'articolo più lontano da tutto.
                min_s = [article,similarities[article][1]]

        starting_centers.append(min_s[0])
        active_center = min_s[0]
        i += 1

        #Deve calcolare le similarità anche con l'ultimo center trovato.
        if i == nClusters:
            for article in insieme - set(starting_centers):
                #Calcola la distanza cosena dall'ultimo centroide trovato
                s = cosine_distance(article,active_center)
                #E' il nuovo centroide più vicino? Lo confronto con il più vicino
                if s > similarities[article][1]:
                    #Il centroide più vicino è active_center.
                    similarities[article] = [active_center , s]

    #Crea il dizionario dei cluster da restituire.
    Clusters = dict()

    #Per ogni articolo ho l'informazione sul centroide iniziale più vicino.
    for center in starting_centers:
        Clusters[center] = dict( Center=dict() , Items={center} )

    for article in insieme - set(starting_centers):
        #Inserisco l'articolo nel cluster più vicino
        Clusters[similarities[article][0]]["Items"].add(article)

    #print_clusters(Clusters)

    #Non vengono ancora calcolati i centroidi (i Center).

    return Clusters

#Crea dei cluster di articoli (specificati nell'insieme di input) utilizzando il metodo delle k-means e la distanza cosena.
def clustering (insieme):

    #Numero di cluster da creare. ClusterFactor è definita all'inizio, variabile globale.
    nCluster = int(len(insieme)*clusterFactor)

    Clusters = starting_clusters(insieme,nCluster)

    # A QUESTO PUNTO ABBIAMO UNA PRIMA PARTIZIONE DELL'INSIEME DI ARTICOLI IN nCluster CLUSTER, ottenuta con il metodo k-means ++.


    #Questo è un insieme che, ad ogni iterazione, indica quali cluster sono stati modificati (per cui quindi deve essere ricalcolato il centroide e le misure di distanza).
    Clusters_updated = set(Clusters.keys())

    #Misure di distanza tra ogni articolo e ogni centroide (utile per ridurre il numero di distanze da calcolare ad ogni iterazione dell'algoritmo).
    similarities = dict()

    for article in insieme:
        #Inizializza
        similarities[article] = dict()

    for i in range(nIter):

        #Conta il numero di spostamenti tra cluster nell'iterazione dell'algoritmo. Se non ce ne sono stati, l'algoritmo si interrompe.
        moves = 0

        #Dizionario dove, temporaneamente, viene salvata la destinazione di un articolo che deve essere spostato da un cluster (move_from) ad un altro (move_to).
        moves_register = dict()

        #Aggiorna i centroidi. Alla prima iterazione Clusters_updated contiene tutti i cluster.
        for cluster in Clusters_updated:
            Clusters[cluster]["Center"] = getCenter(Clusters[cluster]["Items"])


        #Per ogni articolo in ogni cluster vengono calcolate le similarità con i cluster che all'ultima iterazione sono stati modificati (con tutti i cluster se
        #alla prima). Se un articolo è più simile ad un cluster differente da quello di appartenenza, lo spostamento viene "pianificato" (tramite moves_register).
        for cluster in Clusters:
            for article in Clusters[cluster]["Items"]:

                #Le distanze devono essere ricalcolate solo per i cluster che sono stati modificati.
                for cluster_1 in Clusters_updated:
                    similarities[article][cluster_1] = cosine_distance_kmeans(article,Clusters[cluster_1]["Center"])

                most_similar_cluster = max(similarities[article],key=similarities[article].get)
                #Se il cluster più simile è diverso da quello in cui l'articolo si trova:
                if cluster != most_similar_cluster:
                    moves += 1
                    moves_register[article] = dict( Move_to = most_similar_cluster, Move_from = cluster)

        #Se non devono essere fatti spostamenti, l'algoritmo si interrompe.
        if moves == 0:
            print("K-means interrotto all'iterazione {}".format(i))
            #print_clusters(Clusters)
            #L'algoritmo si interrompe e vengono restituiti i cluster come sono.
            return Clusters
        else:
            Clusters_updated = set()
            for article in moves_register:
                Clusters[moves_register[article]["Move_to"]]["Items"].add(article)
                Clusters[moves_register[article]["Move_from"]]["Items"].remove(article)

                #Indica, per l'iterazione successiva, quali cluster sono stati modificati. Questo dovrebbe ridurre il numero di distanze che andranno calcolate ad ogni iterazione
                #dell'algoritmo.
                Clusters_updated.add(moves_register[article]["Move_to"])
                Clusters_updated.add(moves_register[article]["Move_from"])


    #Restituisce il dizionario di cluster
    print("K-means ha raggiunto il numero massimo di iterazioni {}".format(nIter))
    return Clusters

#Argomento l'ID di uno User. Ordina i documenti del suo test set in ordine decrescente di rating previsto (tra 0 e 1), e sull'ordinamento calcola gli indici NDCG, che vengono restituiti. reply -> Fare clustering o meno.
#x2_selection -> Se fare una riduzione delle features utilizzando il metodo del X2.
def recommend_NNC (UserID , reply, x2_selection):

    #Dizionario degli articoli - rating previsto
    ratings = dict()

    if x2_selection == "Y":
        #Applicare il metodo del X2 per ridurre il numero di features con cui rappresentare gli articoli porta a modificare la configurazione delle features,
        #che deve essere riportata a quella originale per poter ripetere il procedimento sullo user successivo. Il seguente dizionario servirà a mantenere
        #l'informazione sulla configurazione di features iniziale.
        init_features = dict()

        #Dizionario per ricavare le quantità O1, O2, O3 e O4
        freq = get_freq(Users[UserID]["Liked"] - Users[UserID]["Test"], Users[UserID]["Disliked"] - Users[UserID]["Test"])

        for article in Users[UserID]["Liked"] | Users[UserID]["Disliked"]:
            init_features[article] = NewsArchive[article]["Features"]
            reducer_X2(article, freq, len(Users[UserID]["Liked"] - Users[UserID]["Test"]), len(Users[UserID]["Disliked"] - Users[UserID]["Test"]))

    #Se il cluster deve essere fatto:
    if reply == "Y":

        #Cluset_liked è un dizionario che contiene l'informazione sui cluster di articoli graditi nel training set.
        clusters_liked = clustering(Users[UserID]["Liked"] - Users[UserID]["Test"])
        #Cluset_Disliked è un dizionario che contiene l'informazione sui cluster di articoli non graditi nel training set.
        clusters_disliked = clustering(Users[UserID]["Disliked"] - Users[UserID]["Test"])

        for ArticleID in Users[UserID]["Test"]:
            ratings[ArticleID] = neighbor_prediction_cluster(clusters_liked,clusters_disliked,ArticleID)

    #Se invece non serve:
    if reply == "N":

        #Per ogni articolo del test set viene previsto il rating
        for ArticleID in Users[UserID]["Test"]:
            ratings[ArticleID] = neighbor_prediction(Users[UserID]["Liked"] - Users[UserID]["Test"],Users[UserID]["Disliked"] - Users[UserID]["Test"],ArticleID)

    #Riporta le features alla configurazione iniziale, se sono state ridotte.
    if x2_selection == "Y":
        for article in Users[UserID]["Liked"] | Users[UserID]["Disliked"]:
             NewsArchive[article]["Features"] = init_features[article]

    #Lista che contiene i valori di DCG@5 [0], DCG@10 [1], DCG@20 [2], calcolati sull'ordinamento proposto
    DCG = [0,0,0]
    #Lista che contiene i valori di IDCG@5 [0], IDCG@10 [1], IDCG@20 [2], calcolati sull'ordinamento ideale
    IDCG = [0,0,0]

    #Articoli rilevanti nel training set (quelli con rating 1, piaciuti dallo user)
    articoli_rilevanti = Users[UserID]["Test"] - Users[UserID]["Disliked"]

    n = list(range(1,len(articoli_rilevanti) + 1))

    for i in n:
        if i == 1:
            IDCG[0] += 1
            IDCG[1] += 1
            IDCG[2] += 1
        elif i <= 5 and i > 1:
            #Logaritmo della posizione in base 2
            IDCG[0] += 1/math.log(i,2)
            IDCG[1] += 1/math.log(i,2)
            IDCG[2] += 1/math.log(i,2)
        elif i > 5 and i <=10:
            IDCG[1] += 1/math.log(i,2)
            IDCG[2] += 1/math.log(i,2)
        elif i > 10 and i <=20:
            IDCG[2] += 1/math.log(i,2)
        else:
            break

    #Indice per la posizione nell'ordine
    i = 1

    #Adesso per il calcolo della componente DCG per l'ordinamento proposto
    for Article in sorted(ratings,key=ratings.get,reverse=True):
        #print("Predicted rating: {}".format(ratings[Article]))
        #Per considerare solo le prime 20 posizioni
        if i > 20:
            break

        #Se l'articolo non è rilevante il valore di rilevanza è 0, e quindi non si considera
        #Se l'articolo è rilevante:
        if Article in Users[UserID]["Liked"]:
            if i == 1:
                DCG[0] += 1
                DCG[1] += 1
                DCG[2] += 1
            elif i <= 5 and i > 1:
                #Logaritmo della posizione in base 2
                DCG[0] += 1/math.log(i,2)
                DCG[1] += 1/math.log(i,2)
                DCG[2] += 1/math.log(i,2)
            elif i > 5 and i <=10:
                DCG[1] += 1/math.log(i,2)
                DCG[2] += 1/math.log(i,2)
            elif i > 10 and i <=20:
                DCG[2] += 1/math.log(i,2)
        i += 1

    #Restituisce la tupla con i valori di NDCG@5, NDCG@10, NDCG@20
    return  DCG[0]/IDCG[0], DCG[1]/IDCG[1], DCG[2]/IDCG[2]

#Restituisce i risultati della raccomandazione tramite Nearest Neighbor Classifier per una serie di valori di alpha proposti (può essere anche un solo valore).
def test_NNC_for_alpha (alphas):

    #Lista che verrà restituita contenente i risultati della raccomandazione sugli user.
    nn_NDCG_results = []

    print("\n[ NEAREST NEIGHBOR CLASSIFIER ]")

    reply = ""
    while reply != "Y" and reply != "N":
        reply = input("\nFare clustering per il Nearest Neighbor Classifier (Y/N)?")

    x2_selection = ""
    while x2_selection != "Y" and x2_selection != "N":
        x2_selection = input("\nSelezionare ulteriormente le features con il metodo del X2 (Y/N)?")

    time_s = 0

    for alpha in alphas:

        #Viene fatta (o rifatta) la feature selection per gli articoli
        feature_Selection(alpha)

        start_time = time.time()
        count = 0
        nn_NDCG = []

        for User in Users.keys():
            count += 1
            print("[{}] - ({}%) Recommending to {}".format(alpha,round(count/len(Users.keys())*100,2),User))
            nn_NDCG.append(recommend_NNC(User,reply,x2_selection))

        mean_NDCG = [0,0,0]

        for result in nn_NDCG:
            mean_NDCG[0] += result[0]
            mean_NDCG[1] += result[1]
            mean_NDCG[2] += result[2]

        mean_NDCG[0] = mean_NDCG[0]/len(Users.keys())
        mean_NDCG[1] = mean_NDCG[1]/len(Users.keys())
        mean_NDCG[2] = mean_NDCG[2]/len(Users.keys())

        nn_NDCG_results.append([alpha,mean_NDCG[0],mean_NDCG[1],mean_NDCG[2]])

        end_time = time.time()
        time_s += end_time - start_time

        #I punteggi tf-idf vengono modificati in feature selection. Devono essere riportati ai valori iniziali per procedere con un altro alpha.
        feature_Deselection()

    minutes_time = time_s/60
    #Restituisce una lista, in una posizione i risultati rispetto ad un alpha.
    return nn_NDCG_results, reply, minutes_time, x2_selection


'''
FUNZIONI BAYES CLASSIFIER
'''

#Restituisce una stima della probabilità che il rating dell'articolo in input sia 1 tramite confronto con il training set (Liked e Disliked) di UserID.
def bayes_estimation(ArticleID,UserID,freq):


    #IN MERITO ALLE FORMULE UTILIZZATE FARE RIFERIMENTO ALLE PAG. 153-155 DELL'AGGARWAL.


    #Di article ID abbiamo le features, che insieme alle chiavi di freq vanno a costituire il vocabolario da utilizzare.

    #Parametro per il laplacian smoothing
    beta = 0.001

    #Numero di articoli nel training set dello user
    nTrain = len((Users[UserID]["Liked"] | Users[UserID]["Disliked"]) - Users[UserID]["Test"])

    #Numero di articoli con rating 1 (Liked)
    nTrain_1 = len(Users[UserID]["Liked"] - Users[UserID]["Test"])

    #Numero di articoli con rating 0 (Disliked)
    nTrain_0 = len(Users[UserID]["Disliked"] - Users[UserID]["Test"])

    #Probabilità base che il rating sia 1.
    p_1 = (nTrain_1 + beta) / (nTrain + 2*beta)

    #Probabilità base che il rating sia 0.
    p_0 = (nTrain_0 + beta) / (nTrain + 2*beta)

    #Queste probabilità verranno aggiornate per arrivare ad una p_1 definitiva da utilizzare per l'ordinamento

    #Iniziando dalle features utilizzate per rappresentare il documento
    for feature in NewsArchive[ArticleID]["Features"]:
        try:

            #Probabilità Occorrenza della feature tra gli articoli con rating 1.
            prob_occ_1 = (freq[feature][1] + beta) / (nTrain_1 + 2*beta)
            p_1 = p_1 * prob_occ_1

            #Probabilità occorrenza della feature tra gli articoli con rating 0.
            prob_occ_0 = (freq[feature][0] + beta) / (nTrain_0 + 2*beta)
            p_0 = p_0 * prob_occ_0

        except:

            #Nel caso in cui la feature non compaia nel training set.
            prob_occ_1 = beta / (nTrain_1 + 2*beta)
            p_1 = p_1 * prob_occ_1
            prob_occ_0 = beta / (nTrain_0 + 2*beta)
            p_0 = p_0 * prob_occ_0


    #Adesso studiamo le occorrenze della features che non compaiono nell'articolo
    for feature in set(freq.keys()) - NewsArchive[ArticleID]["Features"]:

        #Sicuramente queste feature compaiono nel dizionario.
        #nTrain - freq[feature][1] è il numero di articoli con rating 1 in cui la feature non compare
        prob_occ_1 = (nTrain_1 - freq[feature][1] + beta) / (nTrain_1 + 2*beta)
        p_1 = p_1 * prob_occ_1
        prob_occ_0 = (nTrain_0 - freq[feature][0] + beta) / (nTrain_0 + 2*beta)
        p_0 = p_0 * prob_occ_0

    K = p_1 + p_0

    try:
        #print("{}".format(K))
        p_1 = p_1 / K
        return p_1

    except:
        #return 0 perché l'eccezione è che K=0 per approssimazione. Se k=0, allora p_1 e p_0 sono uguali (per approx.) a 0. Usare 0.5, considerando che p_1 e p_0 sarebbero uguali, peggiora i risultati.
        return 0

#L'argomento è l'ID di uno User. Ordina i documenti del suo test set in ordine decrescente di probabilità che il rating sia 1, e sull'ordinamento calcola gli indici NDCG, che vengono restituiti.
#x2_selection -> Se fare una riduzione delle features utilizzando il metodo del X2.
def recommend_BC (UserID,x2_selection):

    #Dizionario che conterrà le frequenze delle features negli articoli del training set.
    freq = dict()

    #Dizionario che conterrà le stime delle probabilità che il rating sia 1 per ogni articolo del test set.
    estimates = dict()

    #Frequenze delle features negli articoli graditi e non.
    freq = get_freq(Users[UserID]["Liked"] - Users[UserID]["Test"], Users[UserID]["Disliked"] - Users[UserID]["Test"])

    if x2_selection =="Y":

        #Applicare il metodo del X2 per ridurre il numero di features con cui rappresentare gli articoli porta a modificare la configurazione delle features,
        #che deve essere riportata a quella originale per poter ripetere il procedimento sullo user successivo. Il seguente dizionario servirà a mantenere
        #l'informazione sulla configurazione di features iniziale.
        init_features = dict()

        for article in Users[UserID]["Liked"] | Users[UserID]["Disliked"]:
            init_features[article] = NewsArchive[article]["Features"]
            reducer_X2(article, freq, len(Users[UserID]["Liked"] - Users[UserID]["Test"]), len(Users[UserID]["Disliked"] - Users[UserID]["Test"]))

        #E' necessario ricalcolare le frequenze ora che il numero di features si è ridotto.
        freq = get_freq(Users[UserID]["Liked"] - Users[UserID]["Test"], Users[UserID]["Disliked"] - Users[UserID]["Test"])

    #Ho le informazioni che mi servono riguardo le occorrenze di parole
    for article in Users[UserID]["Test"]:
        estimates[article] = bayes_estimation(article,UserID,freq)


    if x2_selection =="Y":
        #Riporta le features alla configurazione iniziale, se fossero state ridotte.
        for article in Users[UserID]["Liked"] | Users[UserID]["Disliked"]:
            NewsArchive[article]["Features"] = init_features[article]

    #Lista che contiene i valori di DCG@5 [0], DCG@10 [1], DCG@20 [2], calcolati sull'ordinamento proposto
    DCG = [0,0,0]
    #Lista che contiene i valori di IDCG@5 [0], IDCG@10 [1], IDCG@20 [2], calcolati sull'ordinamento ideale. Nel nostro caso, essendovi almeno 40 articoli rilevanti nel test
    #set per costruzione, i valori di IDCG sono delle costanti. Li calcoliamo ugualmente per poter modificare @k utilizzato, se necessario.
    IDCG = [0,0,0]

    #Articoli rilevanti nel training set (quelli con rating 1, piaciuti dallo user)
    articoli_rilevanti = Users[UserID]["Test"] - Users[UserID]["Disliked"]

    n = list(range(1,len(articoli_rilevanti) + 1))

    for i in n:
        if i == 1:
            IDCG[0] += 1
            IDCG[1] += 1
            IDCG[2] += 1
        elif i <= 5 and i > 1:
            #Logaritmo della posizione in base 2
            IDCG[0] += 1/math.log(i,2)
            IDCG[1] += 1/math.log(i,2)
            IDCG[2] += 1/math.log(i,2)
        elif i > 5 and i <=10:
            IDCG[1] += 1/math.log(i,2)
            IDCG[2] += 1/math.log(i,2)
        elif i > 10 and i <=20:
            IDCG[2] += 1/math.log(i,2)
        else:
            break

    #Indice per la posizione nell'ordine
    i = 1

    #Adesso per il calcolo della componente DCG per l'ordinamento proposto
    for Article in sorted(estimates,key=estimates.get,reverse=True):

        #print("Estimated probability: {}".format(estimates[Article]))
        #Per considerare solo le prime 20 posizioni
        if i > 20:
            break

        #Se l'articolo è rilevante (è contenuto nella sezione "Liked" che, ricordando, contiene gli articoli piaciuti sia del test set che del training set):
        if Article in Users[UserID]["Liked"]:
            if i == 1:
                DCG[0] += 1
                DCG[1] += 1
                DCG[2] += 1
            elif i <= 5 and i > 1:
                #Logaritmo della posizione in base 2
                DCG[0] += 1/math.log(i,2)
                DCG[1] += 1/math.log(i,2)
                DCG[2] += 1/math.log(i,2)
            elif i > 5 and i <=10:
                DCG[1] += 1/math.log(i,2)
                DCG[2] += 1/math.log(i,2)
            elif i > 10 and i <=20:
                DCG[2] += 1/math.log(i,2)
        i += 1

    #Restituisce la tupla con i valori di NDCG@5, NDCG@10, NDCG@20
    return  DCG[0]/IDCG[0], DCG[1]/IDCG[1], DCG[2]/IDCG[2]

#Restituisce i risultati della raccomandazione tramite Bayes Classifier per una serie di valori di alpha proposti (può essere anche un solo valore).
def test_BC_for_alpha(alphas):
    #Risultati, che vengono restituiti.
    b_NDCG_results = []

    print("\n[ BAYES CLASSIFIER ]")

    x2_selection = ""
    while x2_selection != "Y" and x2_selection != "N":
        x2_selection = input("\nSelezionare ulteriormente le features con il metodo del X2 (Y/N)?")

    time_s = 0

    for alpha in alphas:
        #Vengono selezionate le features degli articoli utilizzando gli alpha proposti
        feature_Selection(alpha)

        start_time = time.time()
        count = 0
        b_NDCG = []

        for User in Users.keys():
            count += 1
            print("[{}] - ({}%) Recommending to {}".format(alpha,round(count/len(Users.keys())*100,2),User))
            b_NDCG.append(recommend_BC(User, x2_selection))

        mean_NDCG = [0,0,0]

        for result in b_NDCG:
            mean_NDCG[0] += result[0]
            mean_NDCG[1] += result[1]
            mean_NDCG[2] += result[2]

        mean_NDCG[0] = mean_NDCG[0]/len(Users.keys())
        mean_NDCG[1] = mean_NDCG[1]/len(Users.keys())
        mean_NDCG[2] = mean_NDCG[2]/len(Users.keys())

        b_NDCG_results.append([alpha,mean_NDCG[0],mean_NDCG[1],mean_NDCG[2]])

        end_time = time.time()
        time_s += end_time - start_time

        feature_Deselection()

    minutes_time = time_s/60
    #Restituisce una lista, in una posizione i risultati rispetto ad un alpha.
    return b_NDCG_results, minutes_time, x2_selection


'''
MAIN
'''

def main():


    #I valori che vogliamo usare per la pesatura dei tf-idf score nella feature selection.
    alpha_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #alpha_values = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
    #alpha_values = [0]

    print("\n[ RECOMMENDING FOR ALPHA -> {} ]".format(alpha_values))

    #Per memorizzare i tempi di esecuzione.
    nn_time = 0
    b_time = 0

    do_nn = ""
    while do_nn != "Y" and do_nn != "N":
        do_nn = input("\nNearest Neighbor Classification (Y/N)?")

    do_b = ""
    while do_b != "Y" and do_b != "N":
        do_b = input("\nBayes Classification (Y/N)?")


    if do_nn == "Y":
        #Lista per salvare i risultati delle raccomandazioni tramite Nearest Neighbor Classifier rispetto ai diversi valori di alpha, e altre informazioni
        nn_NDCG_results, reply, nn_time, x2_selection_nn = test_NNC_for_alpha(alpha_values)

    if do_b == "Y":
        #Lista per salvare i risultati delle raccomandazioni tramite Bayes Classifier rispetto ai diversi valori di alpha, e altre informazioni
        b_NDCG_results, b_time, x2_selection_b = test_BC_for_alpha(alpha_values)

    print("\n[  RISULTATI  ] -> [ {} minuti ]".format(round(nn_time + b_time,3)))

    print_parameters()

    if do_nn == "Y":
        print("\n[ RISULTATI NEAREST NEIGHBOR CLASSIFIER (time per rep = {}) (Clustering: {}) (Reduction: {})]\n".format(round(nn_time/len(alpha_values),3),reply,x2_selection_nn))
        print(tabulate(nn_NDCG_results, headers=["Alpha","NDCG@5", "NDCG@10", "NDCG@20"]))

    if do_b =="Y":
        print("\n[ RISULTATI BAYES CLASSIFIER (time per rep = {}) (Reduction: {})]\n".format(round(b_time/len(alpha_values),3),x2_selection_b))
        print(tabulate(b_NDCG_results, headers=["Alpha","NDCG@5", "NDCG@10", "NDCG@20"]))

    print("\nL'ESECUZIONE E' TERMINATA. \n")


if __name__=="__main__":

    #cd C:/Users/User/PYPROGETTO/RACCOMANDAZIONE/RECOMMENDER

    #python Recommender_System.py

    main()

else:

    "\n MODULO \"Recommender_System\" IMPORTATO CORRETTAMENTE!"
