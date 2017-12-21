import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import random
import os
start_token = " "

with open("news.txt",encoding='cp1251') as f:
    names = f.read()[:-1].split('\n')
    names = [start_token+name for name in names]
    
"""print ('n samples = ',len(names))
for x in names[::1000]:
    print (x)
"""
    
"""from sklearn.cross_validation import train_test_split
names_train, names_test = train_test_split(names, test_size=0.05, random_state=42)"""
MAX_LENGTH = max(map(len,names))

#import learn_bpe
#learn_bpe.main(open("news.txt",encoding='cp1251'), open("./bpes_sort.txt",'w',encoding='cp1251'), 1000)
from apply_bpe import BPE #!!!
bpe = BPE( open("./bpes_sort.txt",'r',encoding='cp1251'), '`')

tokens=[' ']
f=open("tokens.txt",'r',encoding='cp1251')
for i in f:
    tokens.append(i.replace('\n',""))
f.close()

tokens = sorted(set(tokens))
tokens = list(tokens)
n_tokens = len(tokens)
print ('n_tokens = ',n_tokens)

token_to_id = {token:ix for ix,token in enumerate(tokens)}

def to_matrix(names,max_len=None,pad=token_to_id[' '],dtype='int32'):
    """Casts a list of names into rnn-digestable matrix"""
    names = [[' '] + bpe.segment(names[i]).split() for i in range(len(names))]
    max_len = max_len or max(map(len,names)) + 1
    names_ix = np.zeros([len(names),max_len],dtype) + pad
    
    #print(max_len)
    
    for i in range(len(names)):
        name_ix = [token_to_id.get(i, -1) for i in names[i]]
        #print(names_ix[i,:len(name_ix)],name_ix)
        #for j in range(len(name_ix)):
            #names_ix[i,j] = name_ix[j]
        #print(max_len-len(name_ix))
        names_ix[i,:len(name_ix)] = name_ix
       

    return names_ix.T
    
import theano
import theano.tensor as T
theano.config.floatX = 'float32'
import lasagne
import lasagne.layers as L

n_tokens = len(tokens)
rnn_num_units = 256
embedding_size = 64

from agentnet.memory import LSTMCell

def log_softmax(a, axis=-1):
    return a - T.log(T.exp(a).sum(axis=axis, keepdims=True))

prev_token = L.InputLayer([None])
prev_rnn = L.InputLayer([None, rnn_num_units])
prev_rnn1 = L.InputLayer([None, rnn_num_units])

# convert character id into embedding

prev_token_emb = L.EmbeddingLayer(prev_token, n_tokens, embedding_size)

# concatenate x embedding and previous h state
#rnn_input = L.ConcatLayer([prev_token_emb, prev_rnn])

# compute next state given x_and_h

#new_rnn = L.DenseLayer(rnn_input, rnn_num_units, nonlinearity=T.tanh)


(new_rnn,new_rnn1) = LSTMCell(prev_rnn,prev_rnn1,prev_token_emb) #GRUCell(prev_rnn1,[new_rnn])

# get probabilities for language model P(x_next|h_next)
next_token_logits = L.DenseLayer(new_rnn1, n_tokens, nonlinearity=None) #L.ConcatLayer([new_rnn,new_rnn1])

next_token_probs = L.NonlinearityLayer(next_token_logits, T.nnet.softmax)
next_token_logprobs = L.NonlinearityLayer(next_token_logits, log_softmax)

input_sequence = T.imatrix("input tokens [time, batch]")
batch_size = input_sequence.shape[1]

predicted_probas = []
h0 = T.zeros([batch_size,rnn_num_units]) #initial hidden state
h1 = T.zeros([batch_size,rnn_num_units])
probas0 = T.zeros([batch_size, n_tokens])

state0 = [h0,h1, probas0]

def rnn_one_step(x_t, h_t, h1_t, prev_probas):
    h_next, h1_next, next_logprobs = L.get_output([new_rnn,new_rnn1, next_token_logprobs],
                           {
                               #send x_t and h_t to the appropriate output
                               prev_token: x_t,
                               prev_rnn: h_t,
                               prev_rnn1: h1_t
                           })
    
    return h_next, h1_next, next_logprobs

(h_seq, h1_seq, predicted_logprobas), upd = theano.scan(rnn_one_step, 
                                        outputs_info=state0, sequences=input_sequence)


x_t = T.ivector('previous tokens')
h_t = theano.shared(np.zeros([1,rnn_num_units],'float32'))
h1_t = theano.shared(np.zeros([1,rnn_num_units],'float32'))

h_next,h1_next,next_logprobs = rnn_one_step(x_t,h_t,h1_t,probas0)
temp = theano.shared(np.float32(1))
next_probs=T.nnet.softmax(next_logprobs/temp)

update_rnn = theano.function([x_t], next_probs,
                           updates={h_t : h_next,
                                   h1_t:h1_next},
                               allow_input_downcast=True)

def test(samples):
    return test_step(to_matrix(samples))

def generate_sample(seed_phrase='',max_length=MAX_LENGTH):
    '''
    The function generates text given a phrase of length at least SEQ_LENGTH.
        
    parameters:
        The phrase is set using the variable seed_phrase
        The optional input "N" is used to set the number of characters of text to predict.     
    '''
    x_sequence = [token_to_id[token] for token in [' '] + bpe.segment(seed_phrase).split()]
    
    h_t.set_value(np.zeros([1,rnn_num_units],'float32'))
    h1_t.set_value(np.zeros([1,rnn_num_units],'float32'))
    
    #feed the seed phrase, if any
    for ix in x_sequence[:-1]:
         _ = update_rnn([ix])
    
    #start generating
    for _ in range(max_length-len(seed_phrase)):
        x_probs = update_rnn([x_sequence[-1]])
        x_sequence.append(np.random.choice(n_tokens,p=x_probs[0]))
        
    return ' '.join([tokens[ix] for ix in x_sequence]).replace('` ', '')

def save():
    np.savez("weights_LSTM_bpe.npz",*L.get_all_param_values([new_rnn,next_token_probs]))
    
def load():
    with np.load('weights_LSTM_bpe.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    L.set_all_param_values([new_rnn,next_token_probs],param_values)
    
load()

def upgrade(a):
    loss_test=test(names_test)
    for i in tqdm_notebook(range(a)):
        batch = to_matrix(sample(names_train,32))
        loss_i = train_step(batch)
        #batch = to_matrix(names_test[0],max_len=MAX_LENGTH)


        history.append(loss_i)
        test_history.append(loss_test)
        if (i+1)%100==0:
            loss_test=test(names_test)
            save()
            clear_output(True)
            plt.plot(history,label='loss')
            plt.plot(test_history,label='test')
            plt.legend()
            plt.show()
            for _ in range(10):
                print(generate_sample())

#assert np.mean(history[:10]) > np.mean(history[-10:]), "RNN didn't converge."

import ipywidgets as widgets
from IPython.display import display

result = widgets.Textarea()
seed=widgets.Text()
temp_slider=widgets.FloatSlider(min=0.001,max=2)
btn = widgets.Button(description='Press me')

def display_result(x):
    if not seed.value.startswith(' '):
        result.value=generate_sample(' '+seed.value)
    else:
        result.value=generate_sample(seed.value)
def temp_change(x):
    global temp
    temp.set_value(temp_slider.value)

temp_slider.on_trait_change(temp_change)
btn.on_click(display_result)

def drawAll():
    display(btn,seed,temp_slider,result)