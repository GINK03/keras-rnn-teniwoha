import os
import sys
import glob
import math
import json
import MeCab
from pathlib import Path 
import pickle

def dump():
  os.system("./fasttext skipgram -dim 256 -minCount 1 -input ./dump.news.wakati  -output model")
  term_vec = {}
  with open('model.vec', 'r') as f:
    next(f)
    for line in f:
      ents = line.split()
      term = ' '.join(ents[:-256])
      vec  = list(map(float, ents[-256:]))
      term_vec[term] = vec
    open('term_vec.pkl', 'wb').write(pickle.dumps(term_vec))
if __name__ == '__main__':
  if '--dump' in sys.argv:
    dump()
