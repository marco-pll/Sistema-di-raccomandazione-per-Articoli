'''
Questo script ha la funzione di estrarre dai file di utenti e news un campione di user. Dal file utenti verrà estratto un certo numero di utenti, che sarà salvato in un dizionario.
Le news con cui questi utenti hanno interagito verranno lette, preprocessate e salvate nel dizionario NewsArchive. Questi due dizionari saranno poi salvati nei file "Users_sample.pkl"
e "News.pkl".
'''
import random
import time
import nltk
import pickle
import requests
import argparse
from bs4 import BeautifulSoup
import multiprocessing as mp
import math
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import PorterStemmer

#Lo stemmer da utilizzare per lo stemming
stemmer=PorterStemmer()

#Numerosità campione
nUsers = 200

#Numero di news cliccate e non cliccate, nel training set, perché un individuo possa entrare nel campione (almeno nNews cliccate e almeno nNews non cliccate)
nNews = 100

"""
Dizionario in cui tutti gli user vengono caricati. Da qui saranno poi campionati. Associata ad ogni chiave, UserID, è associato un dizionario, contenente le chiavi "Liked", "Disliked"
e "Test". A "Liked" è associato un insieme (un set) contenente gli ID delle news che lo user ha cliccato, a "Disliked" è associato l'insieme di ID di news che
lo user ha ignorato. Tutte le news, sia quelle utilizzate per il training che per il test, compaiono in questi due insiemi. A "Test" è associato l'insieme che specifica quali news
verranno utilizzate come test.
"""
Users=dict()

"""
Dizionario in cui gli articoli vengono caricati. Le chiavi sono gli ID degli articoli, a cui è associato un dizionario. Questi dizionari contengono le chiavi "Title", "Category","Subcategory",
"Abstract","Body","Url", a cui sono associate delle stringhe, e poi "TF_IDF". Questa chiave è associata ad un dizionario feature - tf_idf score.
"""
NewsArchive=dict()

'''
Creo il dizionario, DF, per calcolare i valori di document frequency per le parole dei documenti del training set. Le chiavi sono le parole, i valori il numero di documenti
in cui la parola compare.
'''
DF=dict()


#Funzione che, dato l'url del sito, ritorna tutto il testo degli elementi di titolo e paragrafo
def get_Body(url,count):
    '''
    Funzione che, dato l'url del sito, ritorna tutto il testo dagli elementi di titolo e paragrafo

    - url: url del sito da cui estrarre i dati
    - count: per tenere traccia del progresso
    - return: lista di tutti i titoli e paragrafi del sito
    '''
    print(str(round(count,2)) + " %" + "\tAccessing site: " + url)
    #Prende la pagina dall'url e la trasforma in caratteri leggibili
    page = requests.get(url).content.decode("utf8")
    #Trasforma la pagina tramite BeautifulSoup per analizzarla
    soup = BeautifulSoup(page, "html.parser")
    output_list = []
    #Cercha ogni tipo di titolo (h1 NO perchè contiene il titolo dell'articolo) e ogni paragrafo nel sito e lo aggiunge al testo
    for element in soup.findAll(["h2", "h3", "p"]):
        output_list.append(element.text)

    return output_list

#Funzione per creare un file "filename" e salvare il dizionario
def save(obj, filename):
    with open(filename, 'wb') as o:  # Overwrites any existing file.
        pickle.dump(obj, o, pickle.HIGHEST_PROTOCOL)

#Rimuove la punteggiatura nelle stringhe che compongono la lista. Viene restituita una lista senza punteggiature, senza parole ad una lettera (poco indicative) e senza parole con l'apostrofo.
def remove_punctuation (stringList):
    output_List=[]
    for position in range(len(stringList)):
        #Condizione che mi permette di rimuovere la punteggiatura + le parole ad una singola lettera, che non danno informazioni aggiuntive
        if len(stringList[position]) > 1:
            if "\'" not in stringList[position]:
                #Mi permette di rimuovere le parole con l'apostrofo, che generalmente sono quelle risultanti dalla tokenizzazione di didn't ('t), they've ('ve), ('s)....
                output_List.append(stringList[position])

    return output_List

#Rimuove le stopwords presenti nella lista di stringhe
def remove_stopwords (stringList):
    output_List=[]
    for position in range(len(stringList)):
        if stringList[position] not in stopwords.words("english"):
            output_List.append(stringList[position])
    return output_List

#Fa lo stemming delle stringhe nelle posizioni della lista
def stemming (stringList):
    for position in range(len(stringList)):
        #Stemming tramite porter stemmer
        stringList[position]=stemmer.stem(stringList[position])

#Pre-processa una stringa: rimuove la punteggiatura e le stopwords, e poi fa stemming
def preProcess (string):
    #Fa una divisione utilizzando i whitespace (di default) e crea una lista. Tutte le parole vengono convertite in lowercase.
    stringSplit = word_tokenize(string.lower())

    #Rimuove la punteggiatura
    stringSplit = remove_punctuation(stringSplit)

    #Elimina le stop words dalla lista
    stringSplit = remove_stopwords(stringSplit)

    stemming(stringSplit)
    preProcessedString=""

    for word in stringSplit:
        preProcessedString = preProcessedString + " " + word
    return preProcessedString

#Funzione che ottiene il corpo per un articolo. La prima posizione della tupla contiene l'ID dell'articolo, la seconda l'URL, la terza un valore per tenere traccia del progresso.
def article_Body (tuple):
    content = get_Body(tuple[1],tuple[2])
    for i in range(len(content)):
        content[i] = content[i].replace("\r\n"," ")
        content[i] = content[i].replace("\n"," ")
        content[i] = content[i].replace("\t"," ")

    #Ricompongo in una stringa il contenuto dell'articolo
    body=""
    for piece in content:
        body = body + " " + piece

    return tuple[0], body

#Funzione che, dato un articolo, preprocessa titolo, abstract e Body. Nella tupla c'è l'ID dell'articolo [0], il titolo[1], l'abstract [2] e il corpo [3]. In [4] c'è un valore per tenere traccia del progresso.
def article_PreProcess (tuple):
    print("{} %\tPre-Processing: {}".format(round(tuple[4],2),tuple[0]))
    return tuple[0], preProcess(tuple[1]), preProcess(tuple[2]), preProcess(tuple[3])


'''
E' possibile che alcuni User, nella lista di Like e Dislike, abbiano elementi in comune. Questo è dovuto al fatto che, essendovi più accessi di ogni User
alla Homepage, in alcune può aver cliccato News precedentemente ignorate (e quindi inserite nella lista dei dislikes). Le news in comune vengono tolte dai
dislike.
'''
def usersCleaner():
    print("\nTOGLIENDO DALLE LISTE DEI DISLIKE I LIKE...")
    count=0
    for User in Users:
        count = count + int(len(Users[User]["Liked"] & Users[User]["Disliked"]))
        Users[User]["Disliked"]=Users[User]["Disliked"] - Users[User]["Liked"]
    print("\tNelle liste di like e dislike c'erano {} elementi in comune".format(count))

#Prende una lista di ID con label (es. "N12345-1") e toglie la stringa specificata come secondo argomento (es. "-1")
def cleanNewsID(S,String):
    for i in range(len(S)):
        S[i]=S[i].replace(String,"")

#Prende il comportamento degli user dai file "behaviors" di training e di validation. L'informazione viene utilizzata per riempire il dizionario "Users".
def getUsers(filePath):
    print("\nSTO LEGGENDO IL FILE DEGLI USER ({})...".format(filePath))
    start_time = time.time()
    Behaviors=open(filePath,"r")
    line=Behaviors.readline()
    count = 0

    while line != "":
        count += 1
        splitLine=line.split("\t")
        UserID=splitLine[1]
        History=splitLine[3].split(" ")
        Current_impression=splitLine[4].replace("\n","")
        Current_impression=Current_impression.split(" ")

        #Le news che lo user ha cliccato nell'accesso corrente hanno "-1" alla fine. Assunte gradite. Le non cliccate hanno "-0". Assunte non gradite.
        Liked_in_impression=[s for s in Current_impression if "-1" in s]
        #Viene tolto il "-1" e viene ottenuta una lista di ID di articoli
        cleanNewsID(Liked_in_impression,"-1")
        Disliked_in_impression=[s for s in Current_impression if "-0" in s]
        cleanNewsID(Disliked_in_impression,"-0")

        if UserID not in Users.keys():
            Users[UserID] = dict( Liked = set() , Disliked = set() )
        for newsID in History+Liked_in_impression:
            Users[UserID]["Liked"].add(newsID)
        for newsID in Disliked_in_impression:
            Users[UserID]["Disliked"].add(newsID)

        line=Behaviors.readline()

    end_time = time.time()
    Behaviors.close()
    time_spent = end_time - start_time
    print("\nIL FILE E' STATO LETTO ({} impression, {} minuti".format(count, time_spent/60))

#Restituisce i news ID che ci servono effettivamente, relativi agli articoli con cui gli user del campione hanno interagito.
def getNewsSample(sample_Users):
    sample_News = set()
    for user in sample_Users:
        for ID in Users[user]["Liked"]:
            sample_News.add(ID)
        for ID in Users[user]["Disliked"]:
            sample_News.add(ID)
    return sample_News

#Carica le news nel dizionario NewsArchive. Solo quelle effettivamente necessarie (i cui iD compaiono in sample_News).
def get_News (sample_News,news_path):
    print("\n STO LEGGENDO IL FILE DELLE NEWS ({}) ...".format(news_path))
    start_time = time.time()
    news_file=open(news_path,encoding="utf-8")
    line=news_file.readline()

    while line != "":
        splitLine=line.split("\t")
        NewsID=splitLine[0]

        #Controlla che l'articolo sia effettivamente necessario per il campione e che non sia già presente in NewsArchive, visto che abbiamo due file di news.
        if NewsID in sample_News and NewsID not in NewsArchive.keys():
            NewsCategory=splitLine[1]
            NewsSubcategory=splitLine[2]
            NewsTitle=splitLine[3]
            NewsAbstract=splitLine[4]
            NewsUrl=splitLine[5]

            #Creo l'articolo nel dizionario
            NewsArchive[NewsID]=dict(Category=NewsCategory,Subcategory=NewsSubcategory,Title=NewsTitle,Abstract=NewsAbstract,Url=NewsUrl)

        line=news_file.readline()

    end_time = time.time()
    news_file.close()
    time_spent = end_time - start_time
    print("\nLE NEWS ({}) SONO STATE LETTE ( MINUTI: {}).".format(news_path,time_spent/60))

#Ritorna un insieme di user che costituirà il campione di riferimento per la raccomandazione
def getUsersSample ():
    user_sample = set()

    #Considero solo gli user che hanno interagito con un certo numero di articoli nel training set
    for user in Users:
        if len(Users[user]["Liked"]) >= nNews and len(Users[user]["Disliked"]) >= nNews:
            user_sample.add(user)

    return set(random.sample(user_sample,nUsers))

#Funzione che permette poi il calcolo della document frequency di tutte le parole nei documenti. Le coppie parola - "insieme di ID" verranno usate per contare in quanti documenti la parola compare.
def getDF ():
    for Article in NewsArchive:

        #Inserisce la categoria
        try:
            DF[NewsArchive[Article]["Category"]].add(Article)
        except:
            DF[NewsArchive[Article]["Category"]]={Article}

        #Inserisce la sottocategoria
        try:
            DF[NewsArchive[Article]["Subcategory"]].add(Article)
        except:
            DF[NewsArchive[Article]["Subcategory"]]={Article}


        #Inserisce le parole del titolo, dell'abstract e del body
        words = NewsArchive[Article]["Title"] + " " + NewsArchive[Article]["Abstract"] + " " + NewsArchive[Article]["Body"]

        #Split usando i whitespace
        for word in words.split():
            try:
                DF[word].add(Article)
            except:
                DF[word] = {Article}

    #Una volta fatto, non ho veramente bisogno degli insiemi, ma sono sufficienti i numeri
    for term in DF.keys():
        DF[term] = len(DF[term])

#Funzione che stampa alcune informazioni sulla document frequency.
def printDF ():
    n=len(DF.keys())
    print("\nNumero di parole uniche nei documenti caricati: {}".format(n))
    random_words = random.sample(DF.keys(),20)
    print("SOME RANDOM WORDS:")
    for word in random_words:
        print("{}: {}".format(word,DF[word]))

#Calcola gli score TF_IDF per ogni parola di tutti gli articoli
def getTF_IDFScores ():
    nDocuments = len(NewsArchive)
    start_time = time.time()
    for ArticleID in NewsArchive.keys():
        NewsArchive[ArticleID]["TF_IDFScores"]=dict()
        nWords = 0
        string = NewsArchive[ArticleID]["Abstract"] + " " + NewsArchive[ArticleID]["Body"]

        #split per i whitespace
        for word in string.split():
            try:
                NewsArchive[ArticleID]["TF_IDFScores"][word] = NewsArchive[ArticleID]["TF_IDFScores"][word] + 1
            except:
                NewsArchive[ArticleID]["TF_IDFScores"][word] = 1
            nWords = nWords + 1

        #La stessa cosa, ma per le parole della categoria, sottocategoria e titolo
        cat_sub_tit = NewsArchive[ArticleID]["Title"] + " " + NewsArchive[ArticleID]["Category"] + " " + NewsArchive[ArticleID]["Subcategory"]
        for word in cat_sub_tit.split():
            try:
                NewsArchive[ArticleID]["TF_IDFScores"][word] = NewsArchive[ArticleID]["TF_IDFScores"][word] + 1
            except:
                NewsArchive[ArticleID]["TF_IDFScores"][word] = 1
            nWords = nWords + 1

        #Per ogni parola calcola il TF_IDF score
        for word in NewsArchive[ArticleID]["TF_IDFScores"].keys():
            NewsArchive[ArticleID]["TF_IDFScores"][word] = (NewsArchive[ArticleID]["TF_IDFScores"][word]/nWords)*math.log(nDocuments/DF[word])

    end_time = time.time()
    return end_time - start_time

#Funzione per costruire il test set degli user nel campione
def get_Test (sample_Users):

    for User in sample_Users:

        Users[User]["Test"] = set()

        #Prendo il 40% delle news piaciute e le aggiungo al test set
        Users[User]["Test"] = Users[User]["Test"].union(set(random.sample(Users[User]["Liked"],int(len(Users[User]["Liked"])*0.4))))

        #Prendo il 40% delle news non piaciute e le aggiungo al test set
        Users[User]["Test"] = Users[User]["Test"].union(set(random.sample(Users[User]["Disliked"],int(len(Users[User]["Disliked"])*0.4))))

        #Questa è tutta l'informazione di cui ho bisogno. In questo modo posso risalire anche ai documenti nel training set.

#Funzione che permette di ottenere il contenuto degli articoli facendo multiprocessing
def multiProcBody (sample_News):
    #La lista di input da utilizzare per il multiprocessing attraverso Pool.
    list = []
    count = 0
    nDocuments = len(sample_News)
    for ArticleID in sample_News:
        count +=1
        list.append((ArticleID, NewsArchive[ArticleID]["Url"], (count/nDocuments)*100))

    #Utilizza tutti i processori
    nProc = mp.cpu_count()

    start_time = time.time()

    #Ottiene il contenuto di tutti gli articoli
    p = mp.Pool(processes=nProc)
    contents = p.map(article_Body, list, chunksize=1)

    end_time = time.time()

    for i in range(len(contents)):
        NewsArchive[contents[i][0]]["Body"] = contents[i][1]

    return end_time - start_time

#Preprocessa il testo dell'articolo facendo multiprocessing
def multiProcPreP (sample_News):
    #Lista di input
    list = []
    count = 0
    nDocuments = len(sample_News)
    for ArticleID in sample_News:
        count += 1
        #E' necessario preprocessare solo Titolo, Abstract e Corpo.
        list.append((ArticleID, NewsArchive[ArticleID]["Title"],NewsArchive[ArticleID]["Abstract"],NewsArchive[ArticleID]["Body"], (count/nDocuments)*100))

    #Utilizza tutti i processori
    nProc = mp.cpu_count()
    start_time = time.time()

    #Preprocessa titolo, abstract e corpo
    p = mp.Pool(processes=nProc)
    preP = p.map(article_PreProcess, list, chunksize=1)

    end_time = time.time()

    #Sostituisco tutto con la versione pre-processata
    for i in range(len(preP)):
        NewsArchive[preP[i][0]]["Title"] = preP[i][1]
        NewsArchive[preP[i][0]]["Abstract"] = preP[i][2]
        NewsArchive[preP[i][0]]["Body"] = preP[i][3]

    return end_time - start_time

#Funzione che fa tutto. train e validation non fanno riferimento a training e validation set, ma ai file MIND utilizzati.
def sampler (users_train_path, news_train_path, users_val_path, news_val_path):

    #Prende le informazioni sul comportamento degli users
    getUsers(users_train_path)
    getUsers(users_val_path)

    usersCleaner()

    #Estrae, tra gli user con almeno nNews cliccate e nNews non cliccate, un campione di nUsers.
    sample_Users = getUsersSample()

    #Individuo gli iD degli Articoli di cui ho bisogno.
    sample_News = getNewsSample(sample_Users)

    print("\n USERS: {} / {} ARTICOLI ...".format(len(sample_Users),len(sample_News)))

    #Prende gli articoli di cui ho bisogno dai file di news
    get_News(sample_News,news_train_path)
    get_News(sample_News,news_val_path)

    #Memorizzo il tempo in secondi per il web scraping
    time_ws = multiProcBody(sample_News)

    #Memorizzo il tempo in secondi per il pre-processing
    time_pp = multiProcPreP(sample_News)

    getDF()
    time_tf = getTF_IDFScores()

    #Creo una versione di NewsArchive più semplice, con un ridotto numero di informazioni, da usare poi in Recommender_System.py
    NewsArchiveNew = dict()

    for ArticleID in NewsArchive.keys():

        '''
        In questo dizionario devono essere presenti i punteggi tf_idf per ogni parola dell'articolo, e un qualche modo di separare le parole di titolo, categoria
        e sotta-catagoria (il cui peso sarà aumentato) da quelle di corpo e abstract. Tecnicamente sarebbe sufficiente avere due chiavi, una associata a dizionari
        di parole-score, e una associata agli insiemi di parole il cui peso deve essere aumentato. Tuttavia mantenere le chiavi title, category e subcategory è più
        comodo per rappresentare le informazioni (se eventualmente ci fosse la necessità di stampare informazioni sugli articoli, queste saranno più chiare). Vengono quindi
        tagliati corpo e abstract. Url viene mantenuta per indagare situazioni straordinarie.
        '''

        NewsArchiveNew[ArticleID] = dict(Title=NewsArchive[ArticleID]["Title"],Category=NewsArchive[ArticleID]["Category"],Subcategory=NewsArchive[ArticleID]["Subcategory"],Url=NewsArchive[ArticleID]["Url"],TF_IDF=NewsArchive[ArticleID]["TF_IDFScores"])

    #Salvo il dizionario delle news che mi servono, per riutilizzarlo nella raccomandazione (Recommender_System.py)
    save(NewsArchiveNew,"News.pkl")

    get_Test(sample_Users)

    #Non mi servono tutti gli user caricati per la raccomandazione
    UsersNew = dict()
    for User in sample_Users:
        UsersNew[User] = Users[User]

    #Salvo il dizionario degli user che mi servono, per riutilizzarlo nella raccomandazione (Recommender_System.py)
    save(UsersNew,"Users_sample.pkl")

    print("\n ARTICOLI ({}), USERS ({}) -> WEB SCRAPING: {} minuti / PREPROCESSING: {} minuti / TF-IDF: {} minuti".format(len(sample_News),len(sample_Users),time_ws/60,time_pp/60,time_tf/60))

    print("\n[SAMPLER HA FINITO].\n")

#main
def main ():

    parser = argparse.ArgumentParser(description = "Campionamento, webscraping e preprocessing.")

    parser.add_argument("--UTrain", help = "percorso al file behaviors di MIND Small proposto come training set (non è il training set che verrà utilizzato)", required = True)
    parser.add_argument("--UVal", help = "percorso al file behaviors di MIND Small proposto come validation set (non è il validation set che verrà utilizzato)", required = True)
    parser.add_argument("--NTrain", help = "percorso al file news di MIND Small proposto come training set (non è il training set che verrà utilizzato)", required = True)
    parser.add_argument("--NVal", help = "percorso al file news di MIND Small proposto come validation set (non è il validation set che verrà utilizzato)", required = True)

    args = vars(parser.parse_args())

    #Il file behaviors.tsv del training set
    users_train_path = args["UTrain"]

    #Il file news.tsv del training set MIND.
    news_train_path = args["NTrain"]

    #Il file behaviors.tsv del validation set.
    users_val_path = args["UVal"]

    #Il file news.tsv del validation set MIND.
    news_val_path = args["NVal"]

    #Fa tutto: estrae quello che serve e lo salva in file o dizionari
    sampler(users_train_path,news_train_path,users_val_path,news_val_path)




if __name__=="__main__":

    #cd C:/Users/User/PYPROGETTO/RACCOMANDAZIONE/SAMPLING

    #python Sampler.py --UTrain behaviors_train.tsv --NTrain news_train.tsv --UVal behaviors_val.tsv --NVal news_val.tsv

    main()
else:
    print("\n MODULO \"Sampler\" IMPORTATO CORRETTAMENTE!")
