
# following sentdex's pythonprogramming tutorial (8-10):

import nltk

# Note, what is IDLE?

# print(nltk.__file__)
# go to this location, and find data.py.

from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

# Whoa wordnet is awesome!
from nltk.corpus import wordnet

# sample text
# sample = gutenberg.raw("bible-kjv.txt")
#
# tok = sent_tokenize(sample)
#
# for x in range(5):
#     print(tok[x])

syns = wordnet.synsets('program')

print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))
