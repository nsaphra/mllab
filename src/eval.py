import math
import operator

from types import *
from collections import *

def HC(C,N):
    assert type(C) is ListType
    entropy = 0.0
    for i in xrange(len(C)):
        tot = float(C[i])
        if tot != 0:
            entropy += (tot / N) * math.log(tot / N)
    return -entropy

def HK(K,N):
    assert type(K) is ListType
    entropy = 0.0
    for i in xrange(len(K)):
        tot = float(K[i])
        if tot != 0:
            entropy += (tot / N) * math.log(tot / N)
    return -entropy

def HCK_HKC(A, C, K, N):

    assert type(A) is ListType
    assert type(A[0]) is ListType
    assert type(C) is ListType
    assert type(K) is ListType

    hck = 0.0
    hkc = 0.0
    for i in xrange(len(K)):
        for j in xrange(len(C)):
            a_c_k = float(A[j][i])
            if K[i] != 0 and a_c_k != 0.0:
                hck += (a_c_k / N) * math.log(a_c_k / K[i])
            if C[j] != 0 and a_c_k != 0.0:
                hkc += (a_c_k / N) * math.log(a_c_k / C[j])
    hck *= -1.0
    hkc *= -1.0
    return (hck, hkc)

def nvi(gold, pred):

    assert len(gold) == len(pred)

    N = len(gold)

    goldmap = dict()
    predmap = dict()

    for i in range(N):
        if not gold[i] in goldmap:
            goldmap[gold[i]] = len(goldmap)
        if not pred[i] in predmap:
            predmap[pred[i]] = len(predmap)

    ngold = len(goldmap)
    npred = len(predmap)

    A = [ [0] * npred ] * ngold

    print A

    C = [ 0 ] * ngold
    K = [ 0 ] * npred

    for i in xrange(N):
        j = goldmap[gold[i]]
        k = predmap[pred[i]]
        A[j][k] += 1
        C[j] += 1
        K[k] += 1

    print A

    hc = HC(C,N)
    hk = HK(K,N)
    hck, hkc = HCK_HKC(A,C,K,N)
    
    print hc
    print hk
    print hck, hkc

    # VI measure
    VI = hck + hkc
    print 'VI=', VI

    # V measure
    h = hc if (hc == 0.0) else 1.0 - hck/hc
    c = hk if (hk == 0.0) else 1.0 - hkc/hk
    V = 2*h*c/(h+c)
    print 'V=',V
    
    # NVI measure
    NVI = hk if (hc == 0.0) else (hck+hkc)/hc
    NVIK = hc if (hk == 0.0) else (hck+hkc)/hk
    print 'NVI=',NVI
    print 'NVIK=',NVIK

    return NVI

def purity(gold,pred):

    assert len(gold) == len(pred)

    # How many clusters have been hypothesized
    clusters = set()
    for label in pred:
        clusters.add(label)

    # For each hypthesized cluster, find the majority true cluster
    ncorrect = 0
    for k in clusters:
        stats = defaultdict(int)
        # Find the majority *true* cluster
        for (index, value) in enumerate(pred):
            if value == k:
                stats[gold[index]] += 1
        sorted_stats = sorted(stats.iteritems(), key=operator.itemgetter(1))
        print 'majority for k=',k,'is',sorted_stats[-1][0]
        ncorrect += sorted_stats[-1][1]

    return float(ncorrect) / len(pred)
        

# Unit test

if __name__ == "__main__":
    
    gold = [1,1,1,2,2,2]
    pred1 = [1,1,1,2,2,2] #right
    pred2 = [2,2,2,1,1,1] #also right
    pred3 = [1,2,1,1,2,1] #wrong
    pred4 = [1,2,2,2,1,1] #wronger
    pred5 = [1,1,2,3,3,3] #cluster mismatch (more)
    pred6 = [1,1,1,1,1,1] #cluster mismatch (less)

    print 'right'
    nvi(gold,pred1)
    print 'also right'
    nvi(gold,pred2)
    print 'wrong'
    nvi(gold,pred3)
    print 'wronger'
    nvi(gold,pred4)
    print 'num cluster mismatch (more)'
    nvi(gold,pred5)
    print 'num cluster mismatch (less)'
    nvi(gold,pred6)
    
