### Matthew Virgin
### Dr. Chaofan Chen
### COS 482
### 24 April 2023

#### Homework 3, Task 2

from pyspark import SparkConf, SparkContext

## findspark fixes "python not found" error
## solution from https://stackoverflow.com/questions/74851410/python-was-not-found-when-running-pyspark-in-vsc
import findspark
findspark.init()

conf = SparkConf().setMaster("local").setAppName("pageRank")
sc = SparkContext(conf=conf)

## create rdd of a list of edges (edges represented as tuples)
edges = sc.textFile("spark_input.txt").map(lambda line: 
                                           tuple(map(int, line.strip().split())))
                        
## only the destinations on the right of the input file
rightEdges = sc.textFile("spark_input.txt").map(lambda line: (int(line.strip().split()[1])))

## only the destinations of the left of the input file
leftEdges = sc.textFile("spark_input.txt").map(lambda line: (int(line.strip().split()[0])))

## nodes without outgoing edges
lonely_nodes = rightEdges.subtract(leftEdges)

## get all the nodes in the graph
nodes = edges.flatMap(lambda edge: edge).distinct()

## add edge from lonely nodes to every other node in the graph
if not lonely_nodes.isEmpty():      # if nodes w/o outgoing edges exist
    edges = edges.union(lonely_nodes.cartesian(nodes).filter(lambda x: x[0] != x[1]))

## initialize all ranks to 1
pRanks = nodes.map(lambda node: (node, 1.0))

## chance to stay on the page
chance_stay = 0.05
## chance to randomly follow a link
chance_link = 0.85
## chance to randomly jump
chance_jump = .10

## computes contribution of node to neighbors given a tuple that contains
## a node, its neighbors, and its page rank
## returns list of tuples, each containing ID of neighboring node + contribution
def contribs(someTuple):
    node = someTuple[0]
    neighbors = someTuple[1][0]
    curr_pRank = someTuple[1][1]
    num_neighbors = len(neighbors)

    contribution = curr_pRank / num_neighbors
    result = []
    for neighbor_id in neighbors:
        result.append((neighbor_id, contribution))
    
    return result

## iterations
k = 10
everyNode = nodes.collect()
for i in range(k):
    ## group each node with the list of nodes it can travel to by an edge
    ## i.e group each node with the list of its neighbors
    nodeAndDestinations = edges.groupByKey().mapValues(lambda node: list(node))

    ## join with ranks so that it takes form (node, ([destinations], node's page rank))
    nodeDestRanks = nodeAndDestinations.join(pRanks)

    ## get contributions - list of tuples - first ele is node, 2nd is contribution to
    ## and sum them for each vertex by key
    contributions = nodeDestRanks.flatMap(lambda x: contribs(x)).reduceByKey(lambda x, y: x + y)

    ## get all node IDs in contributions
    contIDs = contributions.map(lambda x: x[0]).collect()

    ## update ranks
    new_ranks = []
    for node in everyNode:
        ## add rank of 0 if node not in contributions
        if node not in contIDs:
            new_ranks.append((node, 0))
    pRanks = contributions.union(sc.parallelize(new_ranks)).map(lambda x: (x[0], 0.15 + 0.85 * x[1]))

pRanks_final = pRanks.map(lambda x: (x[0], x[1]/len(everyNode)))  

## output to txt file
filename = "output.txt"
with open(filename, "w") as f:
    # Iterate over the RDD and write each element to the file
    for elem in pRanks_final.collect():
        line = "{} <{}>\n".format(elem[0], elem[1])
        f.write(line)

sc.stop()
## Note: for some reason the terminal does not end and I need to press Ctrl+C
## on my own every time