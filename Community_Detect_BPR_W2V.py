import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array

class Node:
    def __init__(self, nodeNo):
        self.nodeNo = nodeNo
        self.deg = 0
        self.path = None # Path (list of indices) from the root to the node (leaf)
        self.code = None # Huffman encoding
        self.neighbor={}

class Graph:
    def __init__(self, fi, min_count):
        nodes = []
        node_hash = {}
        pair_count=0
        fi = open(fi, 'r')

        for line in fi:
            if line.startswith('#'):
                continue
            nodepair = line.split()
            a=nodepair[0]
            b=nodepair[1]
            for node in nodepair:
                if node not in node_hash:
                    node_hash[node] = len(nodes) 
                    nodes.append(Node(node))
                #assert nodes[node_hash[node]].nodeNo == node, 'Wrong node_hash index'
                nodes[node_hash[node]].deg += 1
            nodes[node_hash[a]].neighbor[node_hash[b]]=1
            nodes[node_hash[b]].neighbor[node_hash[a]]=1
            pair_count += 1
            if pair_count % 10000 == 0:
                sys.stdout.write("\rReading edges %d" % pair_count)
                sys.stdout.flush()

        fi.close()
        sys.stdout.write("%s reading completed\n" % fi)
        
        self.nodes=nodes
        self.node_hash=node_hash  #node_hash[nodeNo] = position in self.nodes[]
        self.node_count=len(node_hash)
        self.pair_count=pair_count

        print 'Total nodes in training file: %d' % self.node_count 
        print 'Total pair in training file: %d' % self.pair_count

    def __getitem__(self, i):
        return self.nodes[i]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, key):
        return key in self.node_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(Node('<unk>'))
        unk_hash = 0
        
        count_unk = 0
        for node in self.nodes:
            if node.deg < min_count:
                count_unk += 1
                tmp[unk_hash].count += node.deg
            else:
                tmp.append(node)

        tmp.sort(key=lambda node : node.deg, reverse=True)

        # Update node_hash
        node_hash = {}
        for i, node in enumerate(tmp):
            node_hash[node.nodeNo] = i

        self.nodes = tmp
        self.node_hash = node_hash

        print
        print 'Unknown nodes size:', count_unk

    def indices(self, nodes):
        return [self.node_hash[node] if node in self else self.node_hash['<unk>'] for node in nodes]

    def encode_huffman(self):
        # Build a Huffman tree
        node_size = len(self)
        count = [t.deg for t in self] + [1e15] * (node_size - 1)
        parent = [0] * (2 * node_size - 2)
        binary = [0] * (2 * node_size - 2)
        
        pos1 = node_size - 1
        pos2 = node_size

        for i in xrange(node_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[node_size + i] = count[min1] + count[min2]
            parent[min1] = node_size + i
            parent[min2] = node_size + i
            binary[min2] = 1

        # Assign binary code and path pointers to each node
        root_idx = 2 * node_size - 2
        for i, token in enumerate(self):
            path = [] # List of indices from the leaf to the root
            code = [] # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= node_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - node_size for j in path[::-1]]
            token.code = code[::-1]

class UnigramTable:
    """
    A list of indices of node in the node set following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, nodes):
        node_size = len(nodes)
        power = 0.75
        norm = sum([math.pow(t.deg, power) for t in nodes]) # Normalizing constant

        table_size = 1e8 # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print 'Filling unigram table'
        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(nodes):
            p += float(math.pow(unigram.deg, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z>10:
        return 1
    if z<-10:
        return 0.00000001
    return 1 / (1 + math.exp(-z))

def init_net(dim, node_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [0/dim, 1/dim]
    tmp = np.random.uniform(low=0.0/dim, high=1.0/dim, size=(node_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)
   
   # Init syn1 with zeros
    tmp = np.zeros(shape=(node_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)

def train_process(pid):
    initial_loss = tr_error(graph, syn0)
    last_error = initial_loss
    sys.stdout.write( "Initial loss: %f\n" %(initial_loss))
    #sys.stdout.flush()
    num_iter = 20
    for iterator in range(num_iter):
        start = len(graph.nodes) / num_processes * pid
        end = len(graph.nodes) if pid == num_processes - 1 else len(graph.nodes) / num_processes * (pid + 1)
        print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)
        alpha = starting_alpha
        sample_count = 0
        
        while start< end:
            sent=[start]+[i for i in graph.nodes[start].neighbor.keys()]
            for sent_pos, token in enumerate(sent):
##                if sample_count % 20000 == 0:
##                    
##                    # Recalculate alpha
##                    alpha = starting_alpha * (1 - float(global_sample_count.value) / graph.pair_count)
##                    if alpha < starting_alpha * 0.001: alpha = starting_alpha * 0.001
##
##                    # Print progress info
##                    sys.stdout.write("\rAlpha: %f, sample count %d " % (alpha, global_sample_count.value))
##                    sys.stdout.flush()

                # Randomize window size, where win is the max window size
                #current_win = np.random.randint(low=1, high=win+1)
                #context_start = max(sent_pos - current_win, 0)
                #context_end = min(sent_pos + current_win + 1, len(sent))
                #context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?
                context = sent
                #print 'context',(context)
                # CBOW
                if cbow:
                    # Compute neu1
                    neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                    assert len(neu1) == dim, 'neu1 and dim do not agree'

                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(graph[token].path, graph[token].code)
                    for target, label in classifiers:
                       # print 'CBOW',target,label
                        z = np.dot(neu1, syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * syn1[target] # Error to backpropagate to syn0
                        syn1[target] += g * neu1  # Update syn1
                        syn1[target] = syn1[target].clip(min=0)

                    # Update syn0
                    for context_word in context:
                        syn0[context_word] += neu1e
                        syn0[context_word] = syn0[context_word].clip(min=0)

                # Skip-gram
                else:
                    for context_word in context:
                        # Init neu1e with zeros
                        neu1e = np.zeros(dim)

                        # Compute neu1e and update syn1
                        if neg > 0:
                            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                        else:
                            classifiers = zip(graph[token].path, graph[token].code)
                        for target, label in classifiers:
                            #print 'SG',target,label
                            z = np.dot(syn0[context_word], syn1[target])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * syn1[target]              # Error to backpropagate to syn0
                            syn1[target] += g * syn0[context_word] # Update syn1
                            syn1[target] = syn1[target].clip(min=0)

                        # Update syn0
                        syn0[context_word] += neu1e
                        syn0[context_word] = syn0[context_word].clip(min=0)
                
                # pairwise ranking estimation
                if len(graph.nodes[token].neighbor.keys())!=0:
                    neighbor_item=np.random.choice(graph.nodes[token].neighbor.keys())
                    non_neighbor_item=np.random.randint(graph.node_count)
                    while non_neighbor_item in graph.nodes[token].neighbor.keys():
                        non_neighbor_item=np.random.randint(graph.node_count)
                
                    p_e=np.dot(syn0[token],syn0[neighbor_item])-np.dot(syn0[token],syn0[non_neighbor_item])
                    bpr_e=1-sigmoid(p_e)
                    syn0[token]+=beta*bpr_e*(syn0[neighbor_item]-syn0[non_neighbor_item])
                    syn0[token]=syn0[token].clip(min=0)
                    syn0[neighbor_item]+=beta*bpr_e*(syn0[token])
                    syn0[neighbor_item]=syn0[neighbor_item].clip(min=0)
                    syn0[non_neighbor_item]+=beta*bpr_e*(-syn0[token])
                    syn0[non_neighbor_item]=syn0[non_neighbor_item].clip(min=0)

                sample_count += 1
                #global_sample_count.value += 1 #one sgd step is done
                
            start+=1 

        current_error=tr_error(graph, syn0)
        sys.stdout.write("Worker %d Iteration %d, loss: %f\n" %(pid, iterator, current_error))
        #sys.stdout.flush()
        if -initial_loss*0.0001 < current_error - last_error < initial_loss*0.0001:
            break
        last_error = current_error
##        # Print progress info
##        sys.stdout.write("\rAlpha: %f, sample count %d " %
##                         (alpha, global_sample_count.value))
##        sys.stdout.flush()
    fi.close()

def save(graph, syn0, fo, binary):
    print 'Saving model to', fo
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(graph, syn0):
            fo.write('%s ' % token.nodeNo)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(graph, syn0):
            nodeNo = token.nodeNo
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (nodeNo, vector_str))

    fo.close()

def __init_process(*args):
    global graph, syn0, syn1, table, cbow, neg, dim, starting_alpha,beta
    global win, num_processes, global_sample_count, last_error, current_error, fi
    
    graph, syn0_tmp,  syn1_tmp, table, cbow, neg, dim, starting_alpha, beta, win, num_processes, global_sample_count, last_error, current_error = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)

def train(fi, fo, cbow, neg, dim, alpha, beta, win, min_count, num_processes, binary):
    # Read train file to init graph
    graph = Graph(fi,  min_count)
    
    # Init net
    syn0, syn1 = init_net(dim, len(graph))
    global_sample_count = Value('i', 0)
    last_error = Value('f',99999)
    current_error = Value('f', 9999.0)
    table = None
    if neg > 0:
        print 'Initializing unigram table'
        table = UnigramTable(graph)
    else:
        print 'Initializing Huffman tree'
        graph.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(graph, syn0, syn1, table, cbow, neg, dim, alpha, beta,
                          win, num_processes, global_sample_count, last_error, current_error, fi))
#    print 'Global',global_sample_count.value
    
    pool.map(train_process, range(num_processes))
    t1 = time.time()
    print
    print 'Completed training. Training took', (t1 - t0) / 60, 'minutes'

    # Save model to file
    save(graph, syn0, fo, binary)

def tr_error(graph, syn0):
    sample_size = int(100*graph.node_count**0.5)
    ranking_loss = 0;
    for i in range(sample_size):
        source = np.random.randint(graph.node_count)
        while len(graph.nodes[source].neighbor.keys())==0:
            source = np.random.randint(graph.node_count)
        neighbor_item=np.random.choice(graph.nodes[source].neighbor.keys())
        non_neighbor_item=np.random.randint(graph.node_count)
        while non_neighbor_item in graph.nodes[source].neighbor.keys():
            non_neighbor_item=np.random.randint(graph.node_count)
        x = np.dot(syn0[source],syn0[neighbor_item])-np.dot(syn0[source],syn0[non_neighbor_item])

        ranking_loss += -math.log(sigmoid(x))

    return ranking_loss
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=0, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-beta', help='Starting beta', dest='beta', default=5, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int) 
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    #TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()

    train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.beta, args.win,
          args.min_count, args.num_processes, bool(args.binary))
