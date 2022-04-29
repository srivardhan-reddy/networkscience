#Trust computation

from ip_v2 import *     #IP
import networkx as nx
import pickle
from collections import defaultdict
import operator
from networkx.algorithms import bipartite
import math
from collections import Counter
import json
# from scipy import stats
import csv
# from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
from networkx.algorithms.bipartite import biadjacency_matrix

#######################################

#Set input parameters before running code

MAX_ITER=35     #no. of iterations for IP algorithm
NUM_TOPICS=17   #no. of topics for user-topic bipartite graph
SIGMA=0.2       #topic node in-degree in bipartite graph at which topic is considered "popular". Set this according to the dataset


#######################################


#######################################

#Set Input files and output files

IP_GRAPH_INP='sample_data/DIGRAPH_743_10k.gpickle'                                    #directed weighted graph of users saved in Networkx Digraph format, to compute user influence and passivity (input to IP) 
IP_INF_OUT='overall_influence_covid_output.txt'                           #influence output of IP written here
IP_PASSIV_OUT='overall_passivity_covid_output.txt'                        #passivity output of IP written here

BIP_GRAPH_INP='sample_data/BIP_GRAPH_INP_743_10k_low_deg_removed.adjlist'                              #bipartite graph of users and topics saved in Networkx Graph format
TRUST_PARAM_VALUES='trust_covid_paramvalues.csv'

TOPIC_USERS_DICT_FILE='sample_data/TOPIC_USERS_DICT_FILE_743_10k_low_deg_removed.txt'                        #Stored in pickle format. Python dictionary of length NUM_TOPICS, with keys as topic names/ids and values as list of users posting on that topic. 
#E.g. topic_users_dict[0] = ['1869', '3496', '1260', '23514', '1786'] where 0 is topic id and '1869' etc are user ids.
TOPIC_USERS_SENTIMENT_DICT_FILE='sample_data/TOPIC_USERS_SENTIMENT_DICT_FILE_743_10k_low_deg_removed.txt'    #Stored in pickle format. Python dictionary of length NUM_TOPICS, with keys as topic names/ids and values as dictionary with 3 keys: pos, neg and neut. Values of internal dictionary with keys pos, neg, neut are list of users who posted on a given topic and whose text contains a given sentiment.
#E.g. topic_users_sentiment_dict[0] = {'neut': ['1075', '2284', '1343'], 'pos': ['819', '762', '40'], 'neg':['160', '97']} where 0 is topic id, '1075' etc are user ids for positive, negative and neutral sentiments.

#Parameter weight values for the three trust components. May require tuning.
A=0.6
B=0.1
C=0.3

TRUST_OUTPUT='trust_values_covid_0.6_0.1_0.3_0.2_output.txt'   #output file containing trustor, trustee, trust value

#######################################



### running IP algorithm
# DG = pickle.load(open(IP_GRAPH_INP, 'rb'))         #loading a weighted DiGraph from a file
DG = pickle.load(open(IP_GRAPH_INP, 'rb'))
#DG = max(nx.weakly_connected_component_subgraphs(DG2),key=len)  #largest connected component
# print (DG.nodes())

edge_list=DG.edges()
for e in edge_list:
    DG[e[0]][e[1]]['acc_rate']=0
    DG[e[0]][e[1]]['rejec_rate']=0


inf_score, passiv_score = IP(DG, MAX_ITER) 

#write output to file
sorted_inf = sorted(inf_score.items(), key=operator.itemgetter(1))
sorted_inf.reverse()
with open(IP_INF_OUT,'w') as f:
    for i in sorted_inf:
        if i[1]!=0:
            f.write("%s : %f\n" %(i[0], i[1]))

sorted_passiv = sorted(passiv_score.items(), key=operator.itemgetter(1))
sorted_passiv.reverse()
with open(IP_PASSIV_OUT,'w') as f:
    for p in sorted_passiv:
        if p[1]!=0:
            f.write("%s : %f\n" %(p[0], p[1]))


print ("computed influence passivity")


### Bipartite graph: compute Trust 

# BG_topic = pickle.load(open(BIP_GRAPH_INP, 'rb'))
# print (BG_topic.nodes(data='True'))
# DG2 = max(nx.strongly_connected_components(BG_topic),key=len)
# BG_topic = nx.read_adjlist(BIP_GRAPH_INP, create_using=nx.DiGraph(), delimiter=' ')
# print (BG_topic.nodes(data='True'))
# print (DG2.nodes(data='True'))
# # u = [n for n in BG_topic.nodes if BG_topic.nodes[n]['bipartite'] == 0]
# topics,users=bipartite.sets(BG_topic)
# users=list(users)
# topics=list(topics)

# print (users)
# print (topics)


#####    Trust(u,v) = A.Influence(u,v) + B.JaccSim(u,v) + C.(1-sum(D(shared-topics))/num(shared-topics))
#####    D(i)={ 2/(1+e^((-degree(i)^sigma) + 2^sigma) - 1 }
#####    use sigma to define from which in-degree value a topic is considered popular. E.g. sigma=0.8 => deg>=10 is popular. sigma=0.2 => deg>=1000 is popular.


inf={}
with open(IP_INF_OUT, 'r') as f:
   for line in f:
      inf[line.split()[0]]=line.split()[2]    

# print ("influence : ", inf)

topic_users_dict = pickle.load(open(TOPIC_USERS_DICT_FILE, 'rb'))
# print (topic_users_dict)
topics = []
users = []
items = topic_users_dict.items()
for item in items:
    topics.append(item[0]), users.append(item[1])

# print ("t_u_dict", topic_users_dict)
# print ("users", users)


#computing values of the three components of trust
def get_trust_components():
    with open(TRUST_PARAM_VALUES, 'w') as fp:
        writer=csv.writer(fp,delimiter=',')
        writer.writerow(["trustor", "trustee", "influence", "jaccard similarity", "topic popularity"])
        sigma=SIGMA   
        count=0
        for node in users:
            for k in range(len(node)):
                count=count+1
                for node2 in inf:                
                    if node[k] == int(node2) : continue 
                    try:   influ=float(inf[node2])
                    except KeyError:   influ=0
                    sharedtopics=[]
                    nu=set()
                    nv=set()
                    for i in topic_users_dict:
                        len1 = len(topic_users_dict[i])!=0
                        len2 = len(node)!=0 

                        if len1 and len2:
                            if node[k] in topic_users_dict[i] and int(node2) in topic_users_dict[i]: 
                                sharedtopics.append(i)
                            if node[k] in topic_users_dict[i]: 
                                nu.update(topic_users_dict[i])
                            if int(node2) in topic_users_dict[i]:
                                nv.update(topic_users_dict[i])
                    intersect=len(nu&nv)
                    union=len(nu|nv)
                    if (union != 0):
                        jaccsim=intersect/float(union)
                        # print (jaccsim)
                        #D(i)={ 2/(1+e^((-degree(i)^sigma) + 2^sigma) - 1 }
                        #C.(1-sum(D(shared-topics))/num(shared-topics))
                        D=[]
                        for i in sharedtopics:
                            deg_i=len(set(topic_users_dict[i]))
                            D.append((float(2)/float(1+math.exp(-pow(deg_i, sigma)+pow(2,sigma))))-1)
                        try:   shar_top=1-(sum(D)/float(len(sharedtopics)))
                        except ZeroDivisionError:  shar_top=0
                        if influ != 0 and jaccsim != 0 and shar_top != 0:
                            writer.writerow([str(node[k]), str(node2), influ, jaccsim, shar_top])
           
      
#adding degree discounting
def create_deg_disc():
    
    BG_topic = nx.read_adjlist(BIP_GRAPH_INP, create_using=nx.DiGraph())
    # print (BG_topic.nodes(data='True'))
    final_users = tuple(tuple(each) for each in users)
    final_user = list(set(final_users))
    # final_user_f = list(set(final_users))
    
    A=biadjacency_matrix(BG_topic,final_user).toarray()    #create adjacency matrix of networkx bipartite graph

    Di = [len(topic_users_dict[i]) for i in topic_users_dict]    #indegree matrix of each topic
    Di=np.array(Di)
    Do = [len(list(BG_topic.edges(u))) for u in users]     #number of topics (outdegree) each user comments on, users is the list of users got from bipartite graph
    Do=np.array(Do)
         
    deg_disc=[]
    df=pd.read_csv(TRUST_PARAM_VALUES)    
    for ui,uj in zip(df['trustor'], df['trustee']):
        try:   i=users.index(ui)
        except ValueError:
           s=0
           deg_disc.append(s)
           continue
        try:   j=users.index(uj)
        except ValueError:
           s=0
           deg_disc.append(s)
           continue
        s=0
        for k in topic_users_dict:
            s=s+((A[i][k]*A[j][k])/math.sqrt(Di[k]))
        s=s/(math.sqrt(Do[i])*math.sqrt(Do[j])) 
        print (s)
        deg_disc.append(s)

    return deg_disc  


#adding sentiment 
with open(TOPIC_USERS_SENTIMENT_DICT_FILE,'rb') as f: topic_users_sentiment_dict=pickle.load(f)

def add_sentiment():
    shar_top_sent=[]         	
    sigma=SIGMA

    df=pd.read_csv(TRUST_PARAM_VALUES)    
    for ui,uj in zip(df['trustor'], df['trustee']):
        sharedtopics_sent=[]
        for i in range(0,len(topic_users_sentiment_dict)):
           if str(ui) in topic_users_sentiment_dict[i]['pos'] and str(uj) in topic_users_sentiment_dict[i]['pos']: sharedtopics_sent.append(i)
           elif str(ui) in topic_users_sentiment_dict[i]['neg'] and str(uj) in topic_users_sentiment_dict[i]['neg']: sharedtopics_sent.append(i)
           elif str(ui) in topic_users_sentiment_dict[i]['neut'] and str(uj) in topic_users_sentiment_dict[i]['neut']: sharedtopics_sent.append(i)
        D=[]
        for i in sharedtopics_sent:
            deg_i=len(set(topic_users_dict[i]))
            D.append((float(2)/float(1+math.exp(-pow(deg_i, sigma)+pow(2,sigma))))-1)
        try:
               shar_top=1-(sum(D)/float(len(sharedtopics_sent)))
        except ZeroDivisionError:   shar_top=0
        shar_top_sent.append(shar_top)

    return shar_top_sent


get_trust_components()
deg_disc=create_deg_disc()
shar_top_sent=add_sentiment()

print ("computed all trust components")


df=pd.read_csv(TRUST_PARAM_VALUES)    
df['degree discounted']=deg_disc
df['topic_popularity_sentiment']=shar_top_sent
cols=list(df.columns)
cols.remove('trustor')
cols.remove('trustee')
for col in cols:
   col_zscore=col+'_zscore'
   df[col_zscore]=(df[col]-df[col].mean())/df[col].std(ddof=0)

cols_to_keep=['trustor', 'trustee', 'influence', 'jaccard similarity', 'topic popularity', 'influence_zscore', 'jaccard similarity_zscore', 'topic popularity_zscore', 'degree discounted', 'degree discounted_zscore', 'topic_popularity_sentiment', 'topic_popularity_sentiment_zscore']
df[cols_to_keep].to_csv(TRUST_PARAM_VALUES, mode='w', index=False)


#column indices from file TRUST_PARAM_VALUES
inf_col=5
jaccsim_col=6
topicpopu_col=11
trustor_col=0
trustee_col=1

#Compute actual trust values
with open(TRUST_PARAM_VALUES, 'rb') as fp:
    reader = csv.reader(fp, delimiter=',')
    reader.next()
    trust_dict=defaultdict(float)
    for row in reader:
        trust=(A*float(row[inf_col]))+(B*float(row[jaccsim_col]))+(C*float(row[topicpopu_col])) 
        if trust>0:   trust_dict[(str(row[trustor_col]), str(row[trustee_col]))]=trust

    #normalize
    norm_values=[]
    mini=min(trust_dict.values())
    maxi=max(trust_dict.values())
    for i in trust_dict.values():  norm_values.append((i-mini)/(maxi-mini))
    norm_trust_dict={}
    for i in range(len(trust_dict)):
       norm_trust_dict[trust_dict.keys()[i]]=norm_values[i]


#write to file
with open(TRUST_OUTPUT, 'w') as f:  
    f.write('Trustor\tTrustee\tTrust value\n') 
    for k, v in norm_trust_dict.iteritems():
       f.write(k[0]+'\t'+k[1]+'\t'+str(v)+'\n')
