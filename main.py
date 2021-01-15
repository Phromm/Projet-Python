import tkinter as tk
import tkinter.font as tkFont
import tkinter.messagebox
from Corpus import Corpus,Document
import datetime as dt
import praw
import urllib.request
import xmltodict 
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


#dimensions pour les menus
WIDTH_menu = 600
HEIGHT_menu = 300

#variables pour stocker le mot clé et la taille des corpus
global mot_cle
mot_cle = ""
global taille_corpus
taille_corpus = 0

#instances de stockage des corpus
corpus_reddit = Corpus("")
corpus_arxiv = Corpus("")

#tableau pour stocker les fréquences de mots
freq = {}

#instanciation pour le nettoyage des données
wn = nltk.WordNetLemmatizer()

#initialisation du dictionnaire employé pour filtrer les mots superflus
stopwords = nltk.corpus.stopwords.words('english')



#initialisation des corpus reddit et arxiv
def start():
    global mot_cle
    mot_cle = get_key()
    taille_corpus = get_size()

    if(get_key() and get_size()):
        corpus_reddit = Corpus("Corpus Reddit : " + mot_cle)
        corpus_arxiv = Corpus("Corpus Arxiv : " + mot_cle)
        reddit(mot_cle,taille_corpus,corpus_reddit)
        arxiv(mot_cle,taille_corpus,corpus_arxiv)
        show_menu()


def get_key():
    mot_cle = keyWordField.get("1.0","end-1c")
    if(mot_cle == ""):
        tk.messagebox.showerror(title="Erreur",message="Aucun mot clé renseigné")
    else:
        return mot_cle
    
    
def get_size():

    taille_corpus = sizeCorpus.get("1.0","end-1c")
    if(taille_corpus == ""):
        tk.messagebox.showerror(title="Erreur",message="Aucune taille de corpus renseignée")
    elif(int(taille_corpus) <= 0):
        tk.messagebox.showerror(title="Erreur",message="La taille du corpus doit être supérieur à zero")
    else:
        return int(taille_corpus)


def reddit(motCle, taille, corpus):

    reddit = praw.Reddit(client_id='5ibikq38IHf3Vg', client_secret='tbHuYsz-3xJN2cVKrtzmdCq01WY', user_agent='Iwan')

    global df_Reddit
    hot_posts = reddit.subreddit(motCle).hot(limit=taille)

    topics_dict = { "title":[], "score":[], "id":[], "url":[], "comms_num": [], "created": [], "summary":[] }

    for submission in hot_posts:
        topics_dict["title"].append(submission.title)
        topics_dict["score"].append(submission.score)
        topics_dict["id"].append(submission.id)
        topics_dict["url"].append(submission.url)
        topics_dict["comms_num"].append(submission.num_comments)
        topics_dict["created"].append(submission.created)
        topics_dict["summary"].append(submission.selftext)

    df_Reddit = pd.DataFrame(topics_dict)
    df_Reddit = set_up_reddit(df_Reddit)

    for post in hot_posts:
        datet = dt.datetime.fromtimestamp(post.created)
        txt = post.title + ". "+ post.selftext
        txt = txt.replace('\n', ' ')
        txt = txt.replace('\r', ' ')
        doc = Document(datet,
                    post.title,
                    "None",
                    txt,
                    post.url)
        corpus.add_doc(doc)
    print("Reddit : OK")

def arxiv(motCle, taille, corpus):
    url = 'http://export.arxiv.org/api/query?search_query=all:'+motCle+'&start=0&max_results='+str(taille)
    data =  urllib.request.urlopen(url).read().decode()
    docs = xmltodict.parse(data)['feed']['entry']

    column_names = docs.pop(0)
    global df_Arxiv
    df_Arxiv = pd.DataFrame(docs, columns=column_names)
    df_Arxiv = set_up_arxiv(df_Arxiv)
    

    for i in docs:
        datet = dt.datetime.strptime(i['published'], '%Y-%m-%dT%H:%M:%SZ')
        try:
            author = [aut['name'] for aut in i['author']][0]
        except:
            author = i['author']['name']
        txt = i['title']+ ". " + i['summary']
        txt = txt.replace('\n', ' ')
        txt = txt.replace('\r', ' ')
        doc = Document(datet,
                    i['title'],
                    author,
                    txt,
                    i['id']
                    )
        corpus.add_doc(doc)
    print("Arxiv : OK")


#mise en forme d'une date à partir d'un nombre de secondes (timestamp)
def get_date(created):
    return dt.datetime.fromtimestamp(created)

#décompose une chaine de caractere en mots
def tokenize(txt):

    tokens = re.split('\W+',txt)
    return tokens

#retire les mots superflus d'un chaine de caractere
def remove_stopwords(txt_tokenized):

    txt_clean = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean

#permet d'obtenir le radical des mots
def lemmatization(token_txt):

    text = [wn.lemmatize(word) for word in token_txt]
    return text

#compte le nombre d'occurence de chaque mot
def count_freq(word):

    for w in word:
        if w in list(freq.keys()):
            freq[w] += 1
        else:
            freq[w] = 1

def set_up_arxiv(df):

    #décomposition des articles
    df['title_tokenized'] = df ['title'].apply(lambda x: tokenize(x.lower()))
    df['summary_tokenized'] = df ['summary'].apply(lambda x: tokenize(x.lower()))

    #retirer les mots superflus
    df['title_no_sw'] = df['title_tokenized'].apply(lambda x: remove_stopwords(x))
    df['summary_no_sw'] = df['summary_tokenized'].apply(lambda x: remove_stopwords(x))

    #simplification des termes
    df ['title_lemmatized'] = df['title_no_sw'].apply(lambda x: lemmatization(x))
    df ['summary_lemmatized'] = df['summary_no_sw'].apply(lambda x: lemmatization(x))

    #on se sépare de certaines données inutiles pour la suite
    df = df.drop(columns=['author', 'link','arxiv:primary_category','category','updated'])

    df['published'] = pd.to_datetime(df['published']).dt.date
    df=df.set_index('published')
    df=df.sort_index()


    return df

def set_up_reddit(df):

    _timestamp = df["created"].apply(get_date)
    df = df.assign(timestamp = _timestamp)
    #on ne garde que la date et pas l'heure
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    
    #décomposition des articles
    df['title_tokenized'] = df ['title'].apply(lambda x: tokenize(x.lower()))
    df['summary_tokenized'] = df ['summary'].apply(lambda x: tokenize(x.lower()))

    #retirer les mots superflus
    df['title_no_sw'] = df['title_tokenized'].apply(lambda x: remove_stopwords(x))
    df['summary_no_sw'] = df['summary_tokenized'].apply(lambda x: remove_stopwords(x))

    #simplification des termes
    df ['title_lemmatized'] = df['title_no_sw'].apply(lambda x: lemmatization(x))
    df ['summary_lemmatized'] = df['summary_no_sw'].apply(lambda x: lemmatization(x))

    df=df.set_index('timestamp')
    df=df.sort_index()

    return df


#retour au menu en fonction de "où" on se trouve
def show_menu():

    clear_widgets()

    #agrandit la fenetre pour voir les graphiques
    update_geometry(int(0.8*root.winfo_screenwidth()),int(0.8*root.winfo_screenheight()))

    restart.place(x=500,y=350)
    frequence_bouton.place(x=800,y=350)
    quitter.place(x=10,y=15)

def frequence_arxiv():

    #remplissage du tableau freq avec le contenu des articles
    df_Arxiv['summary_no_sw'].apply(count_freq)

    #mise en forme des données de frequence dans un dataFrame
    reslt = pd.DataFrame.from_dict(freq, orient='index')
    reslt = reslt.sort_values([0],ascending = [False])
    test = reslt.reset_index( level=None, drop=False, inplace=False, col_level=0, col_fill='')
    test.rename(columns = {'index':'Word', 0:'freq'}, inplace = True) 
    WordFrequencies_Arxiv = pd.DataFrame(test)

    #on s'interesse au 15 mots les plus fréquents
    topWords = WordFrequencies_Arxiv.head(15)
    return topWords


def frequence_reddit():

    #remplissage du tableau freq avec le contenu des articles
    df_Reddit['summary_no_sw'].apply(count_freq)

    #mise en forme des données de frequence dans un dataFrame
    reslt = pd.DataFrame.from_dict(freq, orient='index')
    reslt = reslt.sort_values([0],ascending = [False])
    test = reslt.reset_index( level=None, drop=False, inplace=False, col_level=0, col_fill='')
    test.rename(columns = {'index':'Word', 0:'freq'}, inplace = True) 
    WordFrequencies_Reddit = pd.DataFrame(test)

    #on s'interesse au 15 mots les plus fréquents
    topWords = WordFrequencies_Reddit.head(15)
    return topWords

def decoupage_reddit__by_month():
    months = {}

    
    #découpage des données par mois

    startdate = pd.to_datetime("2021-01-01").date()
    enddate = pd.to_datetime("2021-01-31").date()
    df_Reddit_Janvier =df_Reddit.loc[startdate:enddate]
    months["janvier"] = df_Reddit_Janvier

    startdate = pd.to_datetime("2020-02-01").date()
    enddate = pd.to_datetime("2020-02-29").date()
    df_Reddit_Fevrier =df_Reddit.loc[startdate:enddate]
    months["fevrier"] = df_Reddit_Fevrier

    startdate = pd.to_datetime("2020-03-01").date()
    enddate = pd.to_datetime("2020-03-31").date()
    df_Reddit_Mars =df_Reddit.loc[startdate:enddate]
    months["mars"] = df_Reddit_Mars
    
    startdate = pd.to_datetime("2020-04-01").date()
    enddate = pd.to_datetime("2020-04-30").date()
    df_Reddit_Avril =df_Reddit.loc[startdate:enddate]
    months["avril"] = df_Reddit_Avril
   
    startdate = pd.to_datetime("2020-05-01").date()
    enddate = pd.to_datetime("2020-05-31").date()
    df_Reddit_Mai =df_Reddit.loc[startdate:enddate]
    months["mai"] = df_Reddit_Mai
    
    startdate = pd.to_datetime("2020-06-01").date()
    enddate = pd.to_datetime("2020-06-30").date()
    df_Reddit_Juin =df_Reddit.loc[startdate:enddate]
    months["juin"] = df_Reddit_Juin
   
    startdate = pd.to_datetime("2020-07-01").date()
    enddate = pd.to_datetime("2020-07-31").date()
    df_Reddit_Juillet =df_Reddit.loc[startdate:enddate]
    months["juillet"] = df_Reddit_Juillet
    
    startdate = pd.to_datetime("2020-08-01").date()
    enddate = pd.to_datetime("2020-08-31").date()
    df_Reddit_Aout =df_Reddit.loc[startdate:enddate]
    months["aout"] = df_Reddit_Aout
    
    startdate = pd.to_datetime("2020-09-01").date()
    enddate = pd.to_datetime("2020-09-30").date()
    df_Reddit_Septembre =df_Reddit.loc[startdate:enddate]
    months["septembre"] = df_Reddit_Septembre
    
    startdate = pd.to_datetime("2020-10-01").date()
    enddate = pd.to_datetime("2020-10-31").date()
    df_Reddit_Octobre =df_Reddit.loc[startdate:enddate]
    months["octobre"] = df_Reddit_Octobre
    
    startdate = pd.to_datetime("2020-11-01").date()
    enddate = pd.to_datetime("2020-11-30").date()
    df_Reddit_Novembre =df_Reddit.loc[startdate:enddate]
    months["novembre"] = df_Reddit_Novembre
    
    startdate = pd.to_datetime("2020-12-01").date()
    enddate = pd.to_datetime("2020-12-31").date()
    df_Reddit_Decembre =df_Reddit.loc[startdate:enddate]
    months["decembre"] = df_Reddit_Decembre
    
    

    return months

def frequence_reddit_by_month():
    months = decoupage_reddit__by_month()
    freq_by_month = {}
    k = 0
    for m,df in months.items():
        k = k + 1
        df.title_no_sw.apply(count_freq)
        reslt = pd.DataFrame.from_dict(freq, orient='index')
        reslt = reslt.sort_values([0],ascending = [False])
        test = reslt.reset_index( level=None, drop=False, inplace=False, col_level=0, col_fill='')
        test.rename(columns = {'index':'Word', 0:'freq'}, inplace = True) 
        freq_by_month[m] = pd.DataFrame(test).head(15)
        freq_by_month[m]['Date'] = str(k)

    X = pd.concat([freq_by_month["janvier"], freq_by_month["fevrier"]])
    X2 = pd.concat([X, freq_by_month["mars"]])
    X3 = pd.concat([X2, freq_by_month["avril"]])
    X4 = pd.concat([X3, freq_by_month["mai"]])
    X5 = pd.concat([X4, freq_by_month["juin"]])
    X6 = pd.concat([X5, freq_by_month["juillet"]])
    X7 = pd.concat([X6, freq_by_month["aout"]])
    X8 = pd.concat([X7, freq_by_month["septembre"]])
    X9 = pd.concat([X8, freq_by_month["octobre"]])
    X10 = pd.concat([X9, freq_by_month["novembre"]])
    word_frequencies_reddit_by_month = pd.concat([X10, freq_by_month["decembre"]])
    word_frequencies_reddit_by_month=word_frequencies_reddit_by_month.sort_values(['Date'])

    return word_frequencies_reddit_by_month

def show_frequence_over_time_reddit():
    clear_widgets()

    mois = ["Janvier","Fevrier","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Decembre"]
    y_months = np.arange(len(mois))
    keyWord = search_word_reddit.get("1.0","end-1c")
    data_reddit = frequence_reddit_by_month()


    res_reddit = data_reddit.loc[data_reddit['Word']==keyWord,:].sort_values(['Date'])

    fig_reddit = Figure(figsize = (15, 8))
    graphe_reddit = fig_reddit.add_subplot(111)
    graphe_reddit.plot(res_reddit['Date'],res_reddit['freq'])
    graphe_reddit.set_xticks(y_months)
    graphe_reddit.set_xticklabels(mois, fontsize=5)
    graphe_reddit.set_ylabel("Occurences")
    graphe_reddit.set_title("Apparition du mot "+'"'+mot_cle.upper()+'"'+" au cours du temps - Reddit")
    canvas_reddit = FigureCanvasTkAgg(fig_reddit, master = root)   
    canvas_reddit.draw()

    canvas_reddit.get_tk_widget().place(x=20,y=60)
    menu.place(x=30,y=30)

def decoupage_arxiv__by_month():
    months = {}

    
    #découpage des données par mois

    startdate = pd.to_datetime("2021-01-01").date()
    enddate = pd.to_datetime("2021-01-31").date()
    df_Arxiv_Janvier =df_Arxiv.loc[startdate:enddate]
    months["janvier"] = df_Arxiv_Janvier

    startdate = pd.to_datetime("2020-02-01").date()
    enddate = pd.to_datetime("2020-02-29").date()
    df_Arxiv_Fevrier =df_Arxiv.loc[startdate:enddate]
    months["fevrier"] = df_Arxiv_Fevrier

    startdate = pd.to_datetime("2020-03-01").date()
    enddate = pd.to_datetime("2020-03-31").date()
    df_Arxiv_Mars =df_Arxiv.loc[startdate:enddate]
    months["mars"] = df_Arxiv_Mars
    
    startdate = pd.to_datetime("2020-04-01").date()
    enddate = pd.to_datetime("2020-04-30").date()
    df_Arxiv_Avril =df_Arxiv.loc[startdate:enddate]
    months["avril"] = df_Arxiv_Avril
   
    startdate = pd.to_datetime("2020-05-01").date()
    enddate = pd.to_datetime("2020-05-31").date()
    df_Arxiv_Mai =df_Arxiv.loc[startdate:enddate]
    months["mai"] = df_Arxiv_Mai
    
    startdate = pd.to_datetime("2020-06-01").date()
    enddate = pd.to_datetime("2020-06-30").date()
    df_Arxiv_Juin =df_Arxiv.loc[startdate:enddate]
    months["juin"] = df_Arxiv_Juin
   
    startdate = pd.to_datetime("2020-07-01").date()
    enddate = pd.to_datetime("2020-07-31").date()
    df_Arxiv_Juillet =df_Arxiv.loc[startdate:enddate]
    months["juillet"] = df_Arxiv_Juillet
    
    startdate = pd.to_datetime("2020-08-01").date()
    enddate = pd.to_datetime("2020-08-31").date()
    df_Arxiv_Aout =df_Arxiv.loc[startdate:enddate]
    months["aout"] = df_Arxiv_Aout
    
    startdate = pd.to_datetime("2020-09-01").date()
    enddate = pd.to_datetime("2020-09-30").date()
    df_Arxiv_Septembre =df_Arxiv.loc[startdate:enddate]
    months["septembre"] = df_Arxiv_Septembre
    
    startdate = pd.to_datetime("2020-10-01").date()
    enddate = pd.to_datetime("2020-10-31").date()
    df_Arxiv_Octobre =df_Arxiv.loc[startdate:enddate]
    months["octobre"] = df_Arxiv_Octobre
    
    startdate = pd.to_datetime("2020-11-01").date()
    enddate = pd.to_datetime("2020-11-30").date()
    df_Arxiv_Novembre =df_Arxiv.loc[startdate:enddate]
    months["novembre"] = df_Arxiv_Novembre
    
    startdate = pd.to_datetime("2020-12-01").date()
    enddate = pd.to_datetime("2020-12-31").date()
    df_Arxiv_Decembre =df_Arxiv.loc[startdate:enddate]
    months["decembre"] = df_Arxiv_Decembre
    
    

    return months

def frequence_arxiv_by_month():
    months = decoupage_arxiv__by_month()
    freq_by_month = {}
    k = 0
    for m,df in months.items():
        k = k + 1
        df.title_no_sw.apply(count_freq)
        reslt = pd.DataFrame.from_dict(freq, orient='index')
        reslt = reslt.sort_values([0],ascending = [False])
        test = reslt.reset_index( level=None, drop=False, inplace=False, col_level=0, col_fill='')
        test.rename(columns = {'index':'Word', 0:'freq'}, inplace = True) 
        freq_by_month[m] = pd.DataFrame(test).head(15)
        freq_by_month[m]['Date'] = str(k)

    X = pd.concat([freq_by_month["janvier"], freq_by_month["fevrier"]])
    X2 = pd.concat([X, freq_by_month["mars"]])
    X3 = pd.concat([X2, freq_by_month["avril"]])
    X4 = pd.concat([X3, freq_by_month["mai"]])
    X5 = pd.concat([X4, freq_by_month["juin"]])
    X6 = pd.concat([X5, freq_by_month["juillet"]])
    X7 = pd.concat([X6, freq_by_month["aout"]])
    X8 = pd.concat([X7, freq_by_month["septembre"]])
    X9 = pd.concat([X8, freq_by_month["octobre"]])
    X10 = pd.concat([X9, freq_by_month["novembre"]])
    word_frequencies_arxiv_by_month = pd.concat([X10, freq_by_month["decembre"]])
    word_frequencies_arxiv_by_month=word_frequencies_arxiv_by_month.sort_values(['Date'])

    return word_frequencies_arxiv_by_month

def show_frequence_over_time_arxiv():
    clear_widgets()

    mois = ["Janvier","Fevrier","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Decembre"]
    y_months = np.arange(len(mois))
    keyWord = search_word_arxiv.get("1.0","end-1c")
    data_arxiv = frequence_arxiv_by_month()


    res_arxiv = data_arxiv.loc[data_arxiv['Word']==keyWord,:].sort_values(['Date'])

    fig_arxiv = Figure(figsize = (15, 8))
    graphe_arxiv = fig_arxiv.add_subplot(111)
    graphe_arxiv.plot(res_arxiv['Date'],res_arxiv['freq'])
    graphe_arxiv.set_xticks(y_months)
    graphe_arxiv.set_xticklabels(mois, fontsize=5)
    graphe_arxiv.set_ylabel("Occurences")
    graphe_arxiv.set_title("Apparition du mot "+'"'+mot_cle.upper()+'"'+" au cours du temps - Arxiv")
    canvas_arxiv = FigureCanvasTkAgg(fig_arxiv, master = root)   
    canvas_arxiv.draw()

    canvas_arxiv.get_tk_widget().place(x=20,y=60)
    menu.place(x=30,y=30)
    

def show_frequence():
    restart.place_forget()
    frequence_bouton.place_forget()
    quitter.place_forget()
    menu.place(x=30,y=30)
    search_button_arxiv.place(x=200,y=30)
    search_word_arxiv.place(x=400,y=30)
    search_button_reddit.place(x=600,y=30)
    search_word_reddit.place(x=800,y=30)

    #on s'interesse au 15 mots les plus fréquents
    topWords_arxiv = frequence_arxiv()
    topWords_reddit = frequence_reddit()
    

    #traçage d'un graphe pour visualiser le resultat (Arxiv)
    y_arxiv = np.arange(len(list(topWords_arxiv['Word'])))
    fig_arxiv = Figure(figsize = (15, 4))
    graphe_arxiv = fig_arxiv.add_subplot(111)
    graphe_arxiv.bar(y_arxiv, topWords_arxiv['freq'], align='center', alpha=0.5)
    graphe_arxiv.set_xticks(y_arxiv)
    graphe_arxiv.set_xticklabels(list(topWords_arxiv['Word']))
    graphe_arxiv.set_ylabel("Occurences")
    graphe_arxiv.set_title("Mots les plus fréquents pour "+'"'+mot_cle.upper()+'"'+" - Arxiv")
    canvas_arxiv = FigureCanvasTkAgg(fig_arxiv, master = root)   
    canvas_arxiv.draw()
    canvas_arxiv.get_tk_widget().place(x=20,y=65)

    #traçage d'un graphe pour visualiser le resultat (Arxiv)
    y_reddit = np.arange(len(list(topWords_reddit['Word'])))
    fig_reddit = Figure(figsize = (15, 4))
    graphe_reddit = fig_reddit.add_subplot(111)
    graphe_reddit.bar(y_reddit, topWords_reddit['freq'], align='center', alpha=0.5)
    graphe_reddit.set_xticks(y_reddit)
    graphe_reddit.set_xticklabels(list(topWords_reddit['Word']))
    graphe_reddit.set_ylabel("Occurences")
    graphe_reddit.set_title("Mots les plus fréquents pour "+'"'+mot_cle.upper()+'"'+" - Reddit")
    canvas_reddit = FigureCanvasTkAgg(fig_reddit, master = root)   
    canvas_reddit.draw()
    canvas_reddit.get_tk_widget().place(x=20,y=450)

    
    
#change la taille de la fenetre
def update_geometry(width,height):

    root.geometry(str(width)+"x"+str(height))

def clear_widgets():
    for widget in root.winfo_children():
        widget.place_forget()

#retour au menu de depart
def reboot():
    clear_widgets()
    keyWordField.delete("1.0","end-1c")
    sizeCorpus.delete("1.0","end-1c")
    update_geometry(WIDTH_menu,HEIGHT_menu)
    keyWordField.place(x=(WIDTH_menu/2)-100,y=100)
    sizeCorpus.place(x=(WIDTH_menu/2)-100,y=130)
    linfos.place(x=(WIDTH_menu/2)-200,y=60)
    lkey.place(x=(WIDTH_menu/2)-175,y=96)
    lsize.place(x=(WIDTH_menu/2)-235,y=126)
    submit.place(x=(WIDTH_menu/2)-40,y=170)
    quitter.place(x=10,y=270)


#fenetre principale
root = tk.Tk()
#taille de la fenetre
update_geometry(WIDTH_menu,HEIGHT_menu)
root.title("Projet Python")
#choix d'une couleur de fond
root.configure(bg="#000823")
#choisir si la fenetre est redimensionnable
root.resizable(width=0, height=0)


#champ pour entrer un mot clé
keyWordField = tk.Text(root,height=1, width=20, font=("Arial", 10, "normal"))
keyWordField.place(x=(WIDTH_menu/2)-100,y=100)

#champ pour entrer la taille des corpus
sizeCorpus = tk.Text(root,height=1, width=5)
sizeCorpus.place(x=(WIDTH_menu/2)-100,y=130)

#champ pour entrer un mot à analyser - Arxiv
search_word_arxiv = tk.Text(root,height=1, width = 20,font=("Arial", 10, "normal"))

#champ pour entrer un mot à analyser - Reddit
search_word_reddit = tk.Text(root,height=1, width = 20,font=("Arial", 10, "normal"))


#label informatif sur le champ du mot clé
font1 = tkFont.Font(family="Calibri",size="15",weight="bold")
linfos = tk.Label(root,text="Entrez un mot clé et la taille des corpus souhaité",font=font1,bg="#000823",fg="white")
linfos.place(x=(WIDTH_menu/2)-200,y=60)

#label gauche du champ clé
lkey = tk.Label(root,text="Mot clé : ",font=("Calibri","12"),bg="#000823",fg="white")
lkey.place(x=(WIDTH_menu/2)-175,y=96)

#label gauche du champ taille
lsize = tk.Label(root,text="Taille des corpus : ",font=("Calibri","12"),bg="#000823",fg="white")
lsize.place(x=(WIDTH_menu/2)-235,y=126)

#bouton pour confirmer le mot clé entré
submit = tk.Button(root,text="Confirmer", command= lambda:start(),bg="green3",fg="black", activebackground="green4", activeforeground="white", bd=0,font=("Bahnschrift", 12, "normal"))
submit.place(x=(WIDTH_menu/2)-40,y=170)

#bouton pour lancer l'analyse au cours du temps - Arxiv
search_button_arxiv = tk.Button(root,text="Analyser un terme Arxiv", command= lambda :show_frequence_over_time_arxiv(),bg="green3",fg="black", activebackground="green4", activeforeground="white", bd=0,font=("Bahnschrift", 10, "normal"),width=21)

#bouton pour lancer l'analyse au cours du temps - Reddit
search_button_reddit = tk.Button(root,text="Analyser un terme Reddit", command= lambda :show_frequence_over_time_reddit(),bg="green3",fg="black", activebackground="green4", activeforeground="white", bd=0,font=("Bahnschrift", 10, "normal"),width=21)

#bouton pour redémarrer le programme
restart = tk.Button(root,text="Changer de corpus",padx=20,bg="green3",fg="black", activebackground="green4", activeforeground="white", bd=1,font=("Bahnschrift", 12, "normal"),width=17,height=5,command= lambda :reboot())


#bouton pour afficher les mots les plus fréquents
frequence_bouton = tk.Button(root,text="Termes les plus frequents ",bg="green3",fg="black", activebackground="green4", activeforeground="white", bd=1,font=("Bahnschrift", 12, "normal"),command= lambda:show_frequence(),width=21,height=5)

#bouton pour quitter le programme
quitter = tk.Button(root,text="Quitter", command= lambda: root.destroy(),bg="green3",fg="black", activebackground="green4", activeforeground="white", bd=0,font=("Bahnschrift", 12, "normal"))
quitter.place(x=10,y=270)


#bouton retour au menu
menu = tk.Button(root,text="Retour",bg="green3",fg="black", activebackground="green4", activeforeground="white", bd=0,font=("Bahnschrift", 10, "normal"),command= lambda :show_menu())



#execution
root.mainloop()
