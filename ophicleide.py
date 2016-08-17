import argparse
import datetime
import json
import sys
import uuid
from urllib2 import urlopen

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.feature import Word2Vec

import flask as f
from flask import render_template, request, redirect, url_for

import re

def cleanstr(s):
    noPunctuation = re.sub("[^a-z ]", " ", s.lower())
    collapsedWhitespace = re.sub("(^ )|( $)", "", re.sub("  *", " ", noPunctuation))
    return collapsedWhitespace

def url2rdd(sc, url):
    response = urlopen(url)
    rdd = sc.parallelize(response.read().split("\r\n\r\n"))
    rdd.map(lambda l: l.replace("\r\n", " ").split(" "))
    return rdd.map(lambda l: cleanstr(l).split(" "))

def train(sc, urls):
    w2v = Word2Vec()
    rdds = reduce(lambda a, b: a.union(b), [url2rdd(sc, url) for url in urls.split("\n")])
    return w2v.fit(rdd)

def trainOne(sc, url):
    w2v = Word2Vec()
    return w2v.fit(url2rdd(sc, url))

app = f.Flask(__name__)
models = {}

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/view/<modelname>")
def view(modelname):
    from random import randint
    global models
    model = models[modelname]
    keys = model.getVectors().keys()
    words = ", ".join(list(set([keys[randint(0, len(keys))] for i in range(10)])))
    return render_template('view.html', modelname=modelname, words=words)

@app.route("/new", methods=["POST",])
def newModel():
    global sc, models
    #    return "training with %r" % request.form['source']
    model = trainOne(sc, request.form['source'])
    models[request.form['modelName']] = model
    return redirect(url_for('view', modelname=request.form['modelName']))

@app.route("/query/", methods=["GET"])
def query():
    global models
    model = models[request.args.get('modelName')]

    synonyms = [term for (term, x) in model.findSynonyms(request.args.get('term'), 5)]

    return render_template('query.html', term=request.args.get('term'), modelname=request.args.get('modelName'), synonyms=synonyms)
    
sc = None
models = {}

def main():
    global sc
    # app.debug = True
    if sc is None:
        sconf = SparkConf().setAppName("ophicleide")
        sc = SparkContext(conf=sconf)
    
    app.run(host='0.0.0.0', port=9050)


if __name__ == '__main__':
    main()

