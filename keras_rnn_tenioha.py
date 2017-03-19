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
  model.add(GRU(128*20, return_sequences=False, input_shape=(maxlen, in_dim)))
  model.add(BN())
  model.add(Dense(out_dim))
  model.add(Activation('linear'))
  model.add(Activation('sigmoid'))
  #model.add(Activation('softmax'))
  optimizer = Adam()
  model.compile(loss='binary_crossentropy', optimizer=optimizer) 
  return model


TENIWOHA = set( list( filter(lambda x:x!="", """が,の,を,に,へ,と,で,や,の,に,と,や,か,は,も,ば,と,が,し,て,か,な,ぞ,わ,よ""".split(',') ) ) )
TENIWOHA_IDX = { }

for i, x in enumerate(list("がのをにへとでやかはもばしてなぞわよ")):
  TENIWOHA_IDX[x] = i
TENIWOHA_INV = { i:x for x,i in TENIWOHA_IDX.items() }

def preexe():
  print(TENIWOHA)
  dataset = []
  term_vec = pickle.loads(open('term_vec.pkl', 'rb').read())
  with open('dump.news.wakati', 'r') as f:
    for fi, line in enumerate(f):
      if fi > 10000: break
      terms = line.split()
      for cur in range(10, len(terms) - 10, 1):
        if terms[cur] in TENIWOHA:
           try:
             head = list(map(lambda x:term_vec[x], terms[cur-10:cur]))
             ans  = terms[cur]
             tail = list(map(lambda x:term_vec[x], terms[cur+1:cur+10]))
             pure_text = terms[cur-10:cur+10]
             dataset.append( (head, TENIWOHA_IDX[ans], tail, pure_text) )
           except KeyError as e:
             pass
  print("all data set is %d"%len(dataset))
  open('dataset.pkl', 'wb').write(pickle.dumps(dataset))

def train():
  print("importing data from algebra...")
  datasets = pickle.loads(open('dataset.pkl', 'rb').read())
  sentences = []
  answers   = []
  to_use = datasets[:50000]
  random.shuffle(to_use)
  for dbi, series in enumerate(to_use):
    # 64GByteで最大80万データ・セットくらいまで行ける
    head, ans, tail, pure_text = series 
    head.extend(tail)
    sentences.append(np.array(head))
    answers.append(ans)
  print('nb sequences:', len(sentences))

  print('Vectorization...')
  X = np.zeros((len(sentences), len(sentences[0]), 256), dtype=np.float64)
  y = np.zeros((len(sentences), len(TENIWOHA_IDX)), dtype=np.int)
  for i, sentence in enumerate(sentences):
    if i%10000 == 0:
      print("building training vector... iter %d"%i)
    for t, vec in enumerate(sentence):
      X[i, t, :] = vec
    y[i, answers[i]] = 1
  model    = build_model(maxlen=len(sentences[0]), in_dim=256, out_dim=len(TENIWOHA))
  for iteration in range(1, 41):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)
    MODEL_NAME = "./models/snapshot.%09d.model"%(iteration)
    model.save(MODEL_NAME)

def pred():
  model_type = sorted(glob.glob('./models/snapshot.*.model'))[-1]
  print("start to loading dataset")
  datasets = pickle.loads(open('dataset.pkl', 'rb').read())
  print("model type is %s"%model_type)
  model  = load_model(model_type)
  sentences = []
  answers   = []
  texts     = []
  for dbi, series in enumerate(datasets[0:10]):
    head, ans, tail, pure_text = series 
    head.extend(tail)
    sentences.append(np.array(head))
    answers.append(ans)
    texts.append(pure_text)
  X = np.zeros((len(sentences), len(sentences[0]), 256), dtype=np.float64)
  y = np.zeros((len(sentences), len(TENIWOHA_IDX)), dtype=np.int)
  for i, sentence in enumerate(sentences):
    if i%10000 == 0:
      print("building training vector... iter %d"%i)
    for t, vec in enumerate(sentence):
      X[i, t, :] = vec
  results = model.predict(X)
  for sent, text, result in zip(sentences, texts, results):
    print([(i,t) for i,t in enumerate(text)])
    for i,f in sorted([(i,f) for i,f in enumerate(result.tolist())], key=lambda x:x[1]*-1):
      print(TENIWOHA_INV[i], f)

def main():
  if '--preexe' in sys.argv:
     preexe()
  if '--train' in sys.argv:
     train()
  if '--pred' in sys.argv:
     pred()
if __name__ == '__main__':
  main()
