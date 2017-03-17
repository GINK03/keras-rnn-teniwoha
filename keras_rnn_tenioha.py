from __future__ import print_function
from keras.models               import Sequential, load_model
from keras.layers               import Dense, Activation
from keras.layers               import LSTM, GRU, SimpleRNN
from keras.optimizers           import RMSprop, Adam
from keras.utils.data_utils     import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.noise         import GaussianDropout as GD
import numpy as np
import random
import sys
import tensorflow               as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
import glob
import json
import pickle
import msgpack
import msgpack_numpy as mn
mn.patch()
import MeCab
import plyvel
from itertools import cycle as Cycle

def build_model(maxlen=None, out_dim=None, in_dim=256):
  print('Build model...')
  model = Sequential()
  model.add(GRU(128*5, return_sequences=True, input_shape=(maxlen, in_dim)))
  model.add(BN())
  model.add(GN(0.2))
  model.add(GRU(128*5, return_sequences=False, input_shape=(maxlen, in_dim)))
  #model.add(BN())
  model.add(GN(0.2))
  model.add(Dense(out_dim))
  model.add(Activation('linear'))
  model.add(Activation('sigmoid'))
  optimizer = Adam()
  model.compile(loss='binary_crossentropy', optimizer=optimizer) 
  return model

def sample(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def dynast(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  return np.argmax(preds)

TENIWOHA = set( list( filter(lambda x:x!="", """が,の,を,に,へ,と,で,や,の,に,と,や,か,は,も,ば,と,が,し,て,か,な,ぞ,わ,よ""".split(',') ) ) )
TENIWOHA_IDX = { x:i for i,x in enumerate( filter(lambda x:x!="", """が,の,を,に,へ,と,で,や,の,に,と,や,か,は,も,ば,と,が,し,て,か,な,ぞ,わ,よ""".split(',') ) )  }
def preexe():
  print(TENIWOHA)
  dataset = []
  term_vec = pickle.loads(open('term_vec.pkl', 'rb').read())
  with open('dump.news.wakati', 'r') as f:
    for line in f:
      terms = line.split()
      for cur in range(10, len(terms) - 10, 1):
        if terms[cur] in TENIWOHA:
           try:
             head = list(map(lambda x:term_vec[x], terms[cur-10:cur]))
             ans  = terms[cur]
             tail = list(map(lambda x:term_vec[x], terms[cur+1:cur+10]))
             dataset.append( (head, ans, tail) )
           except KeyError as e:
             pass
  print("all data set is %d"%len(dataset))
  open('dataset.pkl', 'wb').write(pickle.dumps(dataset))

def train():
  print("importing data from algebra...")
  datasets = pickle.loads(open('dataset.pkl', 'rb').read())
  model    = build_model(maxlen=len(datasets), in_dim=256, out_dim=len(TENIWOHA))
  sentences = []
  answers   = []
  for dbi, series in enumerate(datasets[:2048]):
    head, ans, tail = series 
    head.extend(tail)
    sentences.append(np.array(head))
    answers.append(TENIWOHA_IDX[ans])
  print('nb sequences:', len(sentences))

  print('Vectorization...')
  X = np.zeros((len(sentences), len(sentences[0]), 256), dtype=np.bool)
  y = np.zeros((len(sentences), len(TENIWOHA_IDX)), dtype=np.bool)
  for i, sentence in enumerate(sentences):
    for t, vec in enumerate(sentence):
      #print(vec)
      #print(len(vec))
      X[i, t, :] = vec
    y[i, :] = answers[i]
  sys.exit() 
  for iteration in range(1, 4):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)
    MODEL_NAME = "./models/snapshot.%09d.model"%(ci)
    model.save(MODEL_NAME)

def pred():
  model  = load_model(sorted(glob.glob('./*.model'))[-1] )
  for name in glob.glob('./samples/*'):
    text = open(name).read() * 10
    #print(text)
    tag_index = pickle.loads(open('tag_index.pkl', 'rb').read())
    term_vec = pickle.loads(open('term_vec.pkl', 'rb').read())
    m = MeCab.Tagger('-Owakati')
    terms = m.parse(text).split()
    contexts = []
    for term in terms[:200]:
      try:
        contexts.append(term_vec[term]) 
      except KeyError as e:
        contexts.append(term_vec["ダミー"])
    result = model.predict(np.array([contexts]))
    result = {i:w for i,w in enumerate(result.tolist()[0])}
    for tag, index in sorted(tag_index.items(), key=lambda x:result[x[1]]):
      print(name, tag, result[index], index)

def main():
  if '--preexe' in sys.argv:
     preexe()
  if '--train' in sys.argv:
     train()
  if '--pred' in sys.argv:
     pred()
if __name__ == '__main__':
  main()
