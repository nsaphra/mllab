import sys
import os
import getopt
import cPickle
import operator
import random

from math import log
from math import exp

from time import *
from types import *
from operator import *
from collections import *
from unittest import *

from eval import purity
from word_tokenize import tokenize

# Sum two values in log space
def log_sum( x, y ):

    # If one value is much smaller than the other, keep the larger value.
    if (x < (y - log(1e200))):
        return y
    if (y < (x - log(1e200))):
        return x
    
    diff = x - y
    
    # TODO: check if difference is too large
    # print 'diff=',diff
    # print 'exp(diff)=',exp(diff)

    # Otherwise return the sum
    return y + log(1.0 + exp(diff))

# Given: N docs, K clusters, niter iterations of EM
# Return: Viterbi cluster assignment for each document, parameters
def mixturemodel_em( docs, K, niter, alpha ):
    
    # TODO: Randomly initialize the parameters
    params = []

    # EM loop for niter iterations
    for it in xrange(niter):
        
        print 'iteration',it

        # TODO: E-step. Compute expectations

        # TODO: M-step. Maximize parameters given expected counts

        # TODO: Check for convergence
        
        # ======================== END OF EM LOOP ===========================

    # TODO: Find the Viterbi cluster assignment for evaluation
    clusters = []

    return (clusters, params)
            
def process_dir( docs, dirname, filenames ):
    for filename in filenames:
        path = os.path.join(dirname,filename)
        if os.path.isfile(path):
            docs.append(path)

def get_files( path ):
    files = []
    os.path.walk( path, process_dir, files )
    return files

def clean_token( token ):
    token = token.lstrip('>,.')
    token = token.rstrip('.,!?')
    return token.lower()

def labels_from_files( files ):
    return [ os.path.dirname(f) for f in files ]

def filter_words( docs, N ):

    stats = defaultdict(int)
    for doc in docs:
        for x in doc: stats[x] += 1

    singletons = set([ word for word in stats if stats[word] == 1 ]) #use filter
    sorted_stats = sorted(stats.iteritems(), key=operator.itemgetter(1))
    frequent = set([ item[0] for item in sorted_stats[-N:] ])
#    print frequent

    badwords = frequent.union(singletons)
        
    newdocs = []
    for doc in docs:
        newdoc = filter(lambda x: not x in badwords, doc)
        if len(newdoc) > 0:
            newdocs.append( newdoc )

    return newdocs


def read_doc( path ):
    doc = open(path).read()
    tokens = []
    for token in tokenize(doc).split():
        if token.isalpha():
            tokens.append(clean_token(token))
    return tokens

def load_docs( files ):
    docs = []
    for file in files:
        docs.append(read_doc(file))
    return docs

def pickled_load_docs( path ):
    
    files=[]
    docs=[]
    docsfile = os.path.split(path)[1] + ".pickle"
    if os.path.exists(docsfile):
        print "loading pickled data...",
        sys.stdout.flush()
        FILE=open(docsfile)
        files, docs = cPickle.load(FILE)
        FILE.close()
    else:
        # Get the list of files and load them
        files = get_files(path)
        print len(files), "files found, loading...",
        sys.stdout.flush()
        docs = load_docs(files)

        # Run some extra filters on the documents
        N=len(docs)
        docs = filter_words( docs, 20 )
        print len(docs),'docs after filtering words ( from',N,')'

        # If this is not true, need to update labels for evaluation
        assert len(docs) == N

        # Write the pickled version to a file
        FILE = open(docsfile,'w')
        cPickle.dump((files, docs),FILE)
        FILE.close()
        print "done"

    return (files, docs)

class Usage(Exception):
    def __init__(self,msg):
        self.msg = msg

def main( argv=None ):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], 'h', ['help'])
        except getopt.error, msg:
            raise Usage(msg)


        # Load the data in memory
        files, docs = pickled_load_docs( args[0] )

        # Set the number of clusters
        K = int( args[1] )

        assert len(files) == len(docs)

        # Run clustering
        alpha = 1.0
        clustering, params = mixturemodel_em( docs,   # list of documents
                                              K,      # number of clusters
                                              10,     # maximum number of iterations
                                              alpha ) # Dirichlet parameter

        
        # Evaluate the clustering
        labels = labels_from_files( files )

        # Compute purity for a random ("guessing") baseline
        print '---Baseline---'
        baseline = [ random.randint(1,K) for l in labels ]
        p = purity(labels, baseline)
        print 'purity=',p

        # Compute purity for the EM clustering
        print '---Mixture model (EM)---'
        stats = defaultdict(int)
        for label in clustering: stats[label] += 1
        print stats

        # TODO: Uncomment this when mixturemodel_em returns something
        #p = purity(labels, clustering)
        #print 'purity =',p
        
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())