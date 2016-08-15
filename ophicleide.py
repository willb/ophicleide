import argparse
import datetime
import json
import sys
import uuid
from urllib2 import urlopen

import pymongo
from pyspark import SparkConf
from pyspark import SparkContext

from pyspark.mllib.feature import Word2Vec


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

def train(sc, url):
    w2v = Word2Vec()
    rdd = url2rdd(sc, url)
    return w2v.fit(rdd)
