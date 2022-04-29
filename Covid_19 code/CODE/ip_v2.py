#Influence Passivity (IP) Algorithm

import networkx as nx
import pickle
import operator

def normalize(dictionary):
    norm = sum((dictionary[p] for p in dictionary))
    return {k: dictionary[k] / float(norm) for k in dictionary}

def set_acc_rejec_rates(digraph):

# Input graph G=(N,E,W). 
#W={wij on edge(i,j)=(influence exerted by i on j/total influence i attempted to exert on j)} 
#Acceptance rate uij=(wij/sigma(wkj)) for all (k,j) in E
#Rejection rate vji=((1-wji)/sigma(1-wjk)) for all (j,k) in E

    node_set = digraph.nodes()
    countr=0
    #print "node_set in set_acc_rejec_rates= ", str(list(node_set))
    for i in node_set:
        outedges=digraph.out_edges([i]) #list of (i,j)
        for edge in outedges:
            wij=digraph[edge[0]][edge[1]]['weight'] 
            inedges=digraph.in_edges(edge[1]) #list of (k,j)
            sigma_wkj=0
            for kj in inedges:
                sigma_wkj=sigma_wkj+digraph[kj[0]][kj[1]]['weight']
            if(sigma_wkj!=0):
                uij=wij/float(sigma_wkj)
                #print ("uij is %f " %uij)
                # nx.set_edge_attributes(digraph,'acc_rate',{edge:uij})
                nx.set_edge_attributes(digraph,{edge:uij},'acc_rate')
                #print str(nx.get_edge_attributes(digraph, 'acc_rate'))

        inedges=digraph.in_edges([i]) #list of (j,i)
        for edge in inedges:
            wji=digraph[edge[0]][edge[1]]['weight'] 
            outedges=digraph.out_edges(edge[0]) #list of (j,k)
            sigma_1mwjk=0
            for jk in outedges:
                sigma_1mwjk=sigma_1mwjk+(1-digraph[jk[0]][jk[1]]['weight'])
            if(sigma_1mwjk!=0):
                vji=(1-wji)/(float(sigma_1mwjk))
                #print ("vji is %f " %vji)
                nx.set_edge_attributes(digraph,{edge:vji}, 'rejec_rate')

    #print "acc_rate= ", str(nx.get_edge_attributes(digraph, 'acc_rate'))
    #print "rej_rate= ", str(nx.get_edge_attributes(digraph, 'rej_rate'))
    return list(digraph.nodes())
    

def IP(digraph,max_iterations, min_delta=0.00001):

#Input = weighted directed graph saved in Networkx Digraph format with edge weights stored in edge attribute 'weight'
#Output = dictionaries of influence and passivity scores for each graph node

    node_set=set_acc_rejec_rates(digraph)
    passiv = dict.fromkeys(node_set, 1)  #initialize
    inf = dict.fromkeys(node_set, 1)

    print ("starting iterations")
    d=nx.get_edge_attributes(digraph,'rejec_rate')
    d2=nx.get_edge_attributes(digraph,'acc_rate')
    for m in range(max_iterations):
        for p in node_set:
            passiv[p]= sum((inf.get(q,0)*d.get((q,p),0)) for q in digraph.predecessors(p))
        passiv = normalize(passiv)
        
        print ("done with passiv for iteration %d" %m)
        
        old_inf = dict()
        for p in node_set:
            old_inf[p] = inf[p]
            inf[p] = sum((passiv.get(r, 0)*d2.get((p,r),0)) for r in digraph.successors(p))
        inf = normalize(inf)

        print ("done with inf for iteration %d" %m)

        delta = sum((abs(old_inf[k] - inf[k]) for k in inf))
        print ("delta for iteration %d is %f" %(m,delta))
        if delta <= min_delta:
            return (inf, passiv)
    
    return (inf, passiv)
