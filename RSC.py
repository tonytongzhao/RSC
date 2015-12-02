import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array

class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding
        self.user_set={}
        self.neighbor={}
        self.combination={}

class Vocab:
    def __init__(self, fi, pair, comb, min_count,percentage):
        vocab_items = []
        user_hash={}
        vocab_hash = {}
        user_avg={}
        item_avg={}
        rating_mean=0
        rating_count = 0
        pair_count=0
        comb_count=0
        fi = open(fi, 'r')

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
#        for token in ['<bol>', '<eol>']:
#            vocab_hash[token] = len(vocab_items)
#            vocab_items.append(VocabItem(token))

        for line in fi:
            tokens = line.split()
            user=tokens[0]
            token=tokens[1]
            rating=float(tokens[2])
            #if user not in user_hash.keys():
            #    user_hash[user]=len(user_hash.keys())
            user_hash[user]=user_hash.get(user, int(len(user_hash)))
            if token not in vocab_hash:
                vocab_hash[token] = len(vocab_items)
                vocab_items.append(VocabItem(token))
            user=user_hash[user]
            token=vocab_hash[token]
            user_avg[user]=user_avg.get(user,[])
            user_avg[user].append(rating)
            item_avg[token]=item_avg.get(token, [])
            item_avg[token].append(rating)
                #assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
            vocab_items[token].count += 1
            vocab_items[token].user_set[user]=rating
            rating_count += 1
            rating_mean+=rating
            if rating_count % 10000 == 0:
                sys.stdout.write("\rReading ratings %d" % rating_count)
                sys.stdout.flush()
            #if rating_count>10000:
            #    break
            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
#            vocab_items[vocab_hash['<bol>']].count += 1
#            vocab_items[vocab_hash['<eol>']].count += 1
#            word_count += 2
        fi.close()
        sys.stdout.write("%s reading completed\n" % fi)
        for user in user_avg:
            user_avg[user].append(np.average(user_avg[user]))
        for item in item_avg:
            item_avg[item].append(np.average(item_avg[item]))
        pair=open(pair,'r')
        for line in pair:
            tokens=line.split()
            item1=tokens[0]
            item2=tokens[1]
            #if item1 not in vocab_hash or item2 not in vocab_hash:
            #    continue
            if item1 not in vocab_hash:
                vocab_hash[item1]=len(vocab_items)
                vocab_items.append(VocabItem(item1))

            if item2 not in vocab_hash:
                vocab_hash[item2]=len(vocab_items)
                vocab_items.append(VocabItem(item2))
            item1=vocab_hash[item1]
            item2=vocab_hash[item2]
            pair_count+=1
            vocab_items[item1].neighbor[item2]=1
            vocab_items[item2].neighbor[item1]=1
            if pair_count % 10000 == 0:
                sys.stdout.write("\rReading pairs %d" % pair_count)
                sys.stdout.flush()
            #if pair_count>10000:
            #    break
        pair.close()
        sys.stdout.write("%s reading completed\n" % pair)

        comb=open(comb,'r')
        for line in comb:
            tokens=line.split()
            item1=tokens[0]
            item2=tokens[1]
            #if item1 not in vocab_hash or item2 not in vocab_hash:
            #    continue
            if item1 not in vocab_hash:
                vocab_hash[item1]=len(vocab_items)
                vocab_items.append(VocabItem(item1))

            if item2 not in vocab_hash:
                vocab_hash[item2]=len(vocab_items)
                vocab_items.append(VocabItem(item2))
            item1=vocab_hash[item1]
            item2=vocab_hash[item2]
            comb_count+=1
            vocab_items[item1].combination[item2]=1
            vocab_items[item2].combination[item1]=1
            if comb_count % 10000 == 0:
                sys.stdout.write("\rReading combination %d" % comb_count)
                sys.stdout.flush()
            #if comb_count>19999:
            #    break
        comb.close()
        sys.stdout.write("%s reading completed\n" % comb)

        #self.bytes = fi.tell()
        self.vocab_items = vocab_items         # List of VocabItem objects
        self.vocab_hash = vocab_hash           # Mapping from each token to its index in vocab
        self.user_avg=user_avg
        self.item_avg=item_avg
        self.rating_mean=rating_mean/rating_count
        self.rating_count = rating_count           # Total number of words in train file
        self.user_count=len(user_hash.keys())
        self.item_count=len(vocab_hash)
        self.pair_count=pair_count
        self.comb_count=comb_count
        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file

#        self.__sort(min_count)
        print self.rating_count
        print self.rating_count*percentage
        self.split(percentage)
        #assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print 'Total user in training file: %d' % self.user_count
        print 'Total item in training file: %d' % self.item_count 
        print 'Total rating in file: %d' % self.rating_count
#        print 'Total raiting in testing set: %d' % self.rating_count-int(self.rating_count*percentage)
        print 'Total pair in training file: %d' % self.pair_count
        print 'Total combination in training file: %d' % self.comb_count
        #print 'Vocab size: %d' % len(self)

    def split(self, percentage):
        cur_test=0
        test_case=(1-percentage)*self.rating_count
        print 'Test case: ', test_case
        #print test_case
        for item in self.vocab_items:
            if len(item.user_set.keys())<5:
                continue
            if cur_test>=test_case:
                break
            for user in item.user_set.keys():
                if cur_test<test_case and np.random.random()>percentage:
                    cur_test+=1
                    item.user_set[user]+=10
            


    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0
        
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print
        print 'Unknown vocab size:', count_unk

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

    def encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self)
        count = [t.count for t in self] + [1e15] * (vocab_size - 1)
        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)
        
        pos1 = vocab_size - 1
        pos2 = vocab_size

        for i in xrange(vocab_size - 1):
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

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1

        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self):
            path = [] # List of indices from the leaf to the root
            code = [] # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constant

        table_size = 1e8 # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print 'Filling unigram table'
        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

def init_net(dim, vocab_size, user_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=0.1, high=0.22, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)
   
    # init syn_user with random number from a uniform distributino
    tmp=np.random.uniform(low=0.1, high=0.22, size=(user_size,dim))
    syn_user=np.ctypeslib.as_ctypes(tmp)
    syn_user=Array(syn_user._type_,syn_user, lock=False)
   
   # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn_user, syn1)

def train_process(pid):
    # Set fi to point to the right chunk of training file
    start = len(vocab.vocab_items) / num_processes * pid
    end = len(vocab.vocab_items) if pid == num_processes - 1 else len(vocab.vocab_items) / num_processes * (pid + 1)
    #fi.seek(start)
    #print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)
    current_item=start
    alpha = starting_alpha
    iter_num=0
    total_iter=1
    word_count = 0
    last_word_count = 0
    orig_start=start
    tmp_rmse,tmp_mae=99999.99,99999.99
    while start< end:
        #sys.stdout.write("%d iter\n" % (iter_num))
        if iter_num<total_iter and start==end-1:
            iter_num+=1
            start=orig_start
        #global_word_count.value+=()
        #line = fi.readline().strip()
        # Skip blank lines
        # if not line:
        #    continue
        #print line.split()
        # Init sent, a list of indices of words in line
        #sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])
        if len(vocab.vocab_items[start].combination.keys())==0:
            start+=1
            continue
        #word_count+=1
        if  word_count % 2000 <= 10:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count
                # Recalculate alpha
                #alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.rating_count)
                #if alpha < starting_alpha * 0.001: alpha = starting_alpha * 0.001
#                sys.stdout.write("\rProcessing %d" %(global_word_count.value))
#                sys.stdout.flush()
        if global_word_count.value%10000==0:
            tmp_mse, tmp_mae=tr_error(vocab, syn0,syn_user)
            if current_error.value>tmp_mse:
                current_error.value=float(tmp_mse)
            if last_error.value>tmp_mae:
                last_error.value=float(tmp_mae)
#            if current_error==float(tmp_mse):
            sys.stdout.write( "\rProcessing %d, MSE: %f, MAE: %f" %(global_word_count.value, tmp_mse, tmp_mae))
            sys.stdout.flush()
    #if current_error.value>last_error.value:
        #    break
        #last_error.value=current_error.value
        sent=[start]+[i for i in vocab.vocab_items[start].combination.keys()]
        for sent_pos, token in enumerate(sent):
                            # Print progress info
                #sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)\n" %
                #                 (alpha, global_word_count.value, vocab.rating_count,
                #                  float(global_word_count.value) / vocab.rating_count * 100))
                #sys.stdout.flush()

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?
            #print 'context',(context)
            #print 'word', (sent[sent_pos])
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
                    classifiers = zip(vocab[token].path, vocab[token].code)
                for target, label in classifiers:
                   # print 'CBOW',target,label
                    z = np.dot(neu1, syn1[target])
                    p = sigmoid(z)
                    g = alpha * beta*(label - p)
                    neu1e += g * syn1[target] # Error to backpropagate to syn0
                    syn1[target] += g * neu1  # Update syn1

                # Update syn0
                for context_word in context:
                    syn0[context_word] += neu1e

            # Skip-gram
            else:
                for context_word in context:
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        #print 'SG',target,label
                        z = np.dot(syn0[context_word], syn1[target])
                        p = sigmoid(z)
                        g = alpha *beta* (label - p)
                        neu1e += g * syn1[target]              # Error to backpropagate to syn0
                        syn1[target] += g * syn0[context_word] # Update syn1

                    # Update syn0
                    syn0[context_word] += neu1e
            # rating estimation
            if len(vocab.vocab_items[token].user_set.keys())!=0:
                for c_user in vocab.vocab_items[token].user_set.keys():
                    #c_user=np.random.choice(vocab.vocab_items[token].user_set.keys())
                    if vocab.vocab_items[token].user_set[c_user]>8:
                        continue
                        #c_user=np.random.choice(vocab.vocab_items[token].user_set.keys())
                    word_count+=1
                    syn0[token]+=beta*(vocab.vocab_items[token].user_set[c_user]+vocab.rating_mean-vocab.user_avg[c_user][-1]-vocab.item_avg[token][-1]-np.dot(syn0[token], syn_user[c_user]))*syn_user[c_user]
                    syn_user[c_user]+=beta*(vocab.vocab_items[token].user_set[c_user]+vocab.rating_mean-vocab.user_avg[c_user][-1]-vocab.item_avg[token][-1]-np.dot(syn0[token], syn_user[c_user]))*syn0[token]
                #global_word_count.value+=1
             #   word_count+=1
            # pairwise ranking estimation
            
            if len(vocab.vocab_items[token].neighbor.keys())!=0:
                neighbor_item=np.random.choice(vocab.vocab_items[token].neighbor.keys())
                non_neighbor_item=np.random.randint(vocab.item_count)
                while non_neighbor_item in vocab.vocab_items[token].neighbor.keys():
                    non_neighbor_item=np.random.randint(vocab.item_count)
                p_e=np.dot(syn0[token],syn0[neighbor_item])-np.dot(syn0[token],syn0[non_neighbor_item])
                bpr_e=-np.exp(-p_e)/(1+np.exp(-p_e))
                syn0[token]+=alpha*beta*bpr_e*(syn0[neighbor_item]-syn0[non_neighbor_item])
                syn0[neighbor_item]+=alpha*beta*bpr_e*(syn0[token])
                syn0[non_neighbor_item]+=alpha*beta*bpr_e*(-syn0[token])
            
        start+=1
        #word_count+=1
        #print start    
    # Print progress info
      #  global_word_count.value += (word_count - last_word_count)
#    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
#                     (alpha, global_word_count.value, vocab.rating_count,
#                      float(global_word_count.value)/vocab.rating_count * 100))
#    sys.stdout.flush()
    #fi.close()

def save(vocab, syn0, fo, binary):
    print 'Saving model to', fo
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

  #  fo.close()

def __init_process(*args):
    global vocab, syn0, syn_user, syn1, table, cbow, neg, dim, starting_alpha,beta
    global win, num_processes, global_word_count, last_error, current_error,fi
    
    vocab, syn0_tmp, syn_user_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, beta,win, num_processes, global_word_count,last_error, current_error = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)
        syn_user=np.ctypeslib.as_array(syn_user_tmp)

def train(fi, pair, comb, split, fo, cbow, neg, dim, alpha, beta, win, min_count, num_processes, binary):
    # Read train file to init vocab
    vocab = Vocab(fi, pair, comb, min_count, split)
    user_size=vocab.user_count
    
    # Init net
    syn0, syn_user, syn1 = init_net(dim, len(vocab), user_size)
#    print 'Item latent representation ', np.shape(syn0)
#    print 'User latent representation ', np.shape(syn_user)
    global_word_count = Value('i', 0)
    last_error=Value('f',99999)
    current_error=Value('f',9999.0)
    table = None
    if neg > 0:
        print 'Initializing unigram table'
        table = UnigramTable(vocab)
    else:
        print 'Initializing Huffman tree'
        vocab.encode_huffman()
    print 'Begin Training'
    # Begin training using num_processes workers
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn_user, syn1, table, cbow, neg, dim, alpha,beta,
                          win, num_processes, global_word_count, last_error, current_error, fi))
#    print 'Global',global_word_count.value
    
    pool.map(train_process, range(num_processes))
    t1 = time.time()
    print
    print 'Completed training. Training took', (t1 - t0) / 60, 'minutes'

    # Save model to file
    save(vocab, syn0, fo, binary)
    prediction(vocab, syn0, syn_user)

def tr_error(vocab, syn0, syn_user):
    test_case=0
    mae=0
    rmse=0
    for item in xrange(len(vocab.vocab_items)):
        vocab_item=vocab.vocab_items[item]
        if len(vocab_item.user_set.keys())==0:
            continue
        for user in vocab_item.user_set.keys():
            if vocab_item.user_set[user]>8:
          #      print item, user
          #      print len(syn0[item])
          #      print len(syn_user[user])
                test_case+=1
                pred=-vocab.rating_mean+vocab.user_avg[user][-1]+vocab.item_avg[item][-1]
                for i in xrange(len(syn0[item])):
                    pred+=syn0[item][i]*syn_user[user][i]
                gr_r=vocab_item.user_set[user]-10
    #            print gr_r, pred
                mae+=abs(gr_r-pred)
                rmse+=np.power(gr_r-pred,2)
    
   #print MAE: " %(mae/test_case))
    return rmse/test_case, mae/test_case

def prediction(vocab, syn0, syn_user):
    test_case=0
    mae=0
    rmse=0
    for item in xrange(len(vocab.vocab_items)):
        vocab_item=vocab.vocab_items[item]
        if len(vocab_item.user_set.keys())==0:
            continue
        for user in vocab_item.user_set.keys():
            if vocab_item.user_set[user]>8:
          #      print item, user
          #      print len(syn0[item])
          #      print len(syn_user[user])
                test_case+=1
                pred=-vocab.rating_mean+vocab.user_avg[user][-1]+vocab.item_avg[item][-1]
                for i in xrange(len(syn0[item])):
                    pred+=syn0[item][i]*syn_user[user][i]
                gr_r=vocab_item.user_set[user]-10
    #            print gr_r, pred
                mae+=abs(gr_r-pred)
                rmse+=np.power(gr_r-pred,2)
    print 'MSE: ',rmse/test_case
    print 'MAE: ',mae/test_case
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-pair',help='Pairwise Ranking file', dest='pair',required=True)
    parser.add_argument('-comb',help='Combination file', dest='comb',required=True)
    parser.add_argument('-split', help='Split for testing', dest='split', required=True, type=float,default=0.8)
    parser.add_argument('-model', help='Output model file', dest='fo', required=True)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=0, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.0025, type=float)
    parser.add_argument('-beta', help='Starting beta', dest='beta', default=50, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int) 
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    #TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()

    train(args.fi, args.pair,args.comb,args.split, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha,args.beta, args.win,
          args.min_count, args.num_processes, bool(args.binary))
