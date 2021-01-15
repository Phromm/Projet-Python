#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/05/2020

@author: julien and antoine
"""

################################## Déclaration des classes ##################################

import datetime as dt
import re
import pickle
import numpy as np

class Corpus():
    
    def __init__(self,name):
        self.name = name
        self.collection = {}
        self.authors = {}
        self.id2doc = {}
        self.id2aut = {}
        self.ndoc = 0
        self.naut = 0
        self.concatenate = False
        self.chaine = ""
        self.body = ""
            
    def add_doc(self, doc):
        
        self.collection[self.ndoc] = doc
        self.id2doc[self.ndoc] = doc.get_title()
        self.ndoc += 1
        aut_name = doc.get_author()
        aut = self.get_aut2id(aut_name)
        if aut is not None:
            self.authors[aut].add(doc)
        else:
            self.add_aut(aut_name,doc)
            
    def add_aut(self, aut_name,doc):
        
        aut_temp = Author(aut_name)
        aut_temp.add(doc)
        
        self.authors[self.naut] = aut_temp
        self.id2aut[self.naut] = aut_name
        
        self.naut += 1

    def get_aut2id(self, author_name):
        aut2id = {v: k for k, v in self.id2aut.items()}
        heidi = aut2id.get(author_name)
        return heidi

    def get_doc(self, i):
        return self.collection[i]
    
    def get_coll(self):
        return self.collection

    def __str__(self):
        return "Corpus: " + self.name + ", Number of docs: "+ str(self.ndoc)+ ", Number of authors: "+ str(self.naut)
    
    def __repr__(self):
        return self.name

    def sort_title(self,nreturn=None):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_title())][:(nreturn)]

    def sort_date(self,nreturn):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_date(), reverse=True)][:(nreturn)]
    
    def save(self,file):
            pickle.dump(self, open(file, "wb" ))

    def search(self, expre):
        docs = []
        if (not self.concatenate):
            for x,y in self.get_coll().items():
                docs.append(y.get_text())
            chaine = ". ".join(docs)
        results = re.findall(expre, chaine)
        print(results)
        
    def merge_docs(self):
        for x,y in self.collection.items():
            self.body += y.title + y.text
            
    #retourne le nombre de document contenant le mot recherché
    def query_count(self,query):
        sum = 0
        for x,y in self.collection.items():
            sum += y.search_query(query)
        return sum
    
    #score IDF ()
    def IDF(self,query):
       return np.log((self.ndoc - self.query_count(query) + 0.5) / (self.query_count(query) + 0.5) + 1)
   
    #taille moyenne des documents du corpus
    def avg_length(self):
        sum_len = 0
        for x,y in self.collection.items():
            sum_len += len(y.title) + len(y.text)
        
        return sum_len / self.ndoc
    
    #score BM25 d'un terme en considérant le corpus comme un gros document
    
    def BM25(self,query):
       k1 = 1.5
       #calcul de la fréquence de query dans le corpus
       self.merge_docs()
       frequence = (len(re.findall(query,self.body)) / len(self.body.split()))
       
       return (frequence * (k1+1)) / ((frequence + k1) * 0.25*(len(self.body)/self.avg_length()))

class Author():
    def __init__(self,name):
        self.name = name
        self.production = {}
        self.ndoc = 0
        
    def add(self, doc):     
        self.production[self.ndoc] = doc
        self.ndoc += 1

    def __str__(self):
        return "Auteur: " + self.name + ", Number of docs: "+ str(self.ndoc)
    def __repr__(self):
        return self.name
    


class Document():
    
    # constructor
    def __init__(self, date, title, author, text, url):
        self.date = date
        self.title = title
        self.author = author
        self.text = text
        self.url = url
    
    # getters
    
    def get_author(self):
        return self.author

    def get_title(self):
        return self.title
    
    def get_date(self):
        return self.date
    
    #def get_source(self):
        #return self.source
        
    def get_text(self):
        return self.text

    def __str__(self):
        return "Document " + self.getType() + " : " + self.title
    
    def __repr__(self):
        return self.title
    
    def getType(self):
        pass
    
    #test la présence d'un terme dans un document
    def search_query(self,query):
        if(re.search(query,self.title+self.text) != None):
            return 1
        return 0
    
    def query_frequency(self,query):
        contenu = self.title + self.text
        return (len(re.findall(query,contenu)) / len(contenu.split()))


###################### TESTS ########################   
"""
corpus = Corpus("Corona")


reddit = praw.Reddit(client_id='5ibikq38IHf3Vg', client_secret='tbHuYsz-3xJN2cVKrtzmdCq01WY', user_agent='Iwan')
hot_posts = reddit.subreddit('Coronavirus').hot(limit=100)
for post in hot_posts:
    datet = dt.datetime.fromtimestamp(post.created)
    txt = post.title + ". "+ post.selftext
    txt = txt.replace('\n', ' ')
    txt = txt.replace('\r', ' ')
    doc = Document(datet,
                   post.title,
                   post.author_fullname,
                   txt,
                   post.url)
    corpus.add_doc(doc)

url = 'http://export.arxiv.org/api/query?search_query=all:covid&start=0&max_results=100'
data =  urllib.request.urlopen(url).read().decode()
docs = xmltodict.parse(data)['feed']['entry']

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



#tests pour le calcul des scores IDF et BM25
print(corpus.IDF("covid"))
print(corpus.BM25("covid"))
"""