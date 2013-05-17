import sys
import re
from math import log10, log
import pickle

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
      
    f.close()
    return (queries, features) 

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features):
    rankedQueries = {}
    for query in queries.keys():
      results = queries[query]
      #features[query][x].setdefault('body_hits', {}).values() returns the list of body_hits for all query terms
      #present in the document, empty if nothing is there. We sum over the length of the body_hits array for all
      #query terms and sort results in decreasing order of this number
      rankedQueries[query] = sorted(results, 
                                    key = lambda x: sum([len(i) for i in 
                                    features[query][x].setdefault('body_hits', {}).values()]), reverse = True)

    return rankedQueries

def load_doc_freq(docFreqFile):
    term_doc_freq = {}
    with open(docFreqFile, 'rb') as ff:
        term_doc_freq = pickle.load(ff)
    return term_doc_freq

def vector_from_text(items, content):
    vec = [0] * len(items)
    content = content.split()
    for idx in range(len(items)):
        vec[idx] = content.count(items[idx])
    return vec

def vector_product(vec1, vec2):
    assert(len(vec1) == len(vec2))
    return [ (vec1[i] * vec2[i]) for i in range(len(vec1)) ]

def vector_dot_product(vec1, vec2):
    return sum(vector_product(vec1, vec2))

def vector_sum(vec1, vec2):
    assert(len(vec1) == len(vec2))
    return [ (vec1[i] + vec2[i]) for i in range(len(vec1)) ]

def vector_scale(vec, alpha):
    return [ (float(alpha) * float(u)) for u in vec ]

def sublinear_scale(vec):
    rvec = []
    for u in vec:
        if u == 0:
            rvec.append(0)
        else:
            rvec.append(1 + log(u))
    return rvec

def vector_doc_freq(items, doc_freq):
    vec = [0] * len(items)
    for idx in range(len(items)):
        vec[idx] = log10(doc_freq[items[idx]])
    return vec

def weight_average(vecList, normalizer):
    #weights = [1.0, 0.3, 0.1, 0.3, 2.0]
    weights = [1.0, 0.3, 0.1, 0.3, 2.0]
    rvec = [0] * len(vecList[0])
    for idx in range(len(vecList)):
        #rvec = vector_sum(rvec, vector_scale(sublinear_scale(vecList[idx]), float(weights[idx])/float(normalizer)))
        rvec = vector_sum(rvec, vector_scale(vecList[idx], float(weights[idx])/float(normalizer)))
    return rvec
        
def task1(queries, features, doc_freq):
    rankedQueries = {}
    for query in queries.keys():
        # Query item and query vector
        qitem = list(set(query.split()))
        qvec = sublinear_scale(vector_from_text(qitem, query))
        #qvec = vector_from_text(qitem, query)
        idf = vector_doc_freq(qitem, doc_freq)
        qvec = vector_product(qvec, idf)
        
        # Calculate vectors and scores
        results = queries[query]
        feat = {}
        for x in results:
            # title
            title = features[query][x]['title']
            title_vec = vector_from_text(qitem, title)
            # url
            url = re.sub(r'\W+', ' ', x)
            url_vec = vector_from_text(qitem, url)
            # header
            header_vec = [0] * len(qitem)
            if 'header' in features[query][x]:
                header_arr = features[query][x]['header']
                for header in header_arr:
                    header_vec = vector_sum(header_vec, vector_from_text(qitem, header))
            # body
            body_vec = [0] * len(qitem)
            if 'body_hits' in features[query][x]:
                body = features[query][x]['body_hits']
                body_vec = [len(body.setdefault(item, [])) for item in qitem]
            # achors
            anchor_vec = [0] * len(qitem)
            if 'anchors' in features[query][x]:
                anchor = features[query][x]['anchors']
                for key in anchor:
                    anchor_vec = vector_sum(anchor_vec, [anchor[key] * u for u in vector_from_text(qitem, key)])
            # length normalization
            norm = features[query][x]['body_length'] + 400
            dvec = weight_average([title_vec, header_vec, url_vec, body_vec, anchor_vec], norm)
            feat[x] = vector_dot_product(qvec, dvec)
        rankedQueries[query] = [u[0] for u in sorted(feat.items(), key=lambda x:x[1], reverse=True)]
    return rankedQueries

def task2(queries, features, doc_freq):
    rankedQueries = {}
    for query in queries.keys():
        # Query item and query vector
        qitem = list(set(query.split()))
        qvec = sublinear_scale(vector_from_text(qitem, query))
        idf = vector_doc_freq(qitem, doc_freq)
        qvec = vector_product(qvec, idf)
        
        # Calculate vectors and scores
        results = queries[query]
        feat = {}
        for x in results:
            # title
            title = features[query][x]['title']
            title_vec = vector_from_text(qitem, title)
            # url
            url = re.sub(r'\W+', ' ', x)
            url_vec = vector_from_text(qitem, url)
            # header
            header_vec = [0] * len(qitem)
            if 'header' in features[query][x]:
                header_arr = features[query][x]['header']
                for header in header_arr:
                    header_vec = vector_sum(header_vec, vector_from_text(qitem, header))
            # body
            body_vec = [0] * len(qitem)
            if 'body_hits' in features[query][x]:
                body = features[query][x]['body_hits']
                body_vec = [len(body.setdefault(item, [])) for item in qitem]
            # achors
            anchor_vec = [0] * len(qitem)
            if 'anchors' in features[query][x]:
                anchor = features[query][x]['anchors']
                for key in anchor:
                    anchor_vec = vector_sum(anchor_vec, [anchor[key] * u for u in vector_from_text(qitem, key)])
            # length normalization
            norm = features[query][x]['body_length'] + 400
            dvec = weight_average([title_vec, header_vec, url_vec, body_vec, anchor_vec], norm)
            feat[x] = vector_dot_product(qvec, dvec)
        rankedQueries[query] = [u[0] for u in sorted(feat.items(), key=lambda x:x[1], reverse=True)]
    return rankedQueries
    
#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries):
    for query in queries:
      print("query: " + query)
      for res in queries[query]:
        print("  url: " + res)

def printToFileRankedResults(queries, outputFile):
    with open(outputFile, 'w') as ff:
        for query in queries:
            ff.write("query: " + query + "\n")
            for res in queries[query]:
                ff.write("  url: " + res + "\n")

#inparams
#  featureFile: file containing query and url features
def main(featureFile):
    #output file name
    outputFile = "ranked.txt" #Please don't change this!

    #populate map with features from file
    (queries, features) = extractFeatures(featureFile)

    #load document frequency
    term_doc_freq = load_doc_freq("term_doc_freq")

    #calling baseline ranking system, replace with yours
    #rankedQueries = baseline(queries, features)
    rankedQueries = task1(queries, features, term_doc_freq)
    print >> sys.stderr, sum([len(rankedQueries[u]) for u in rankedQueries])
    
    #print ranked results to file
    #printRankedResults(rankedQueries)
    printToFileRankedResults(rankedQueries, outputFile)
       
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    #main(sys.argv[1])
    main("queryDocTrainData")


