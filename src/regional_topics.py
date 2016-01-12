import argparse
import sys
from collections import Counter, defaultdict
from gensim import corpora, models
from gensim.models import ldamodel
from nltk.corpus import stopwords
import re

topic_sep = re.compile(r"0\.[0-9]{3}\*")

parser = argparse.ArgumentParser(description="""
takes as input a TSV file with a variable and texts, computes topics, and shows the distribution over topics for each variable
each line is one document
documents are assumed to be tokenized
""")

parser.add_argument('input', help='input file')
parser.add_argument('topics', help='number of topics', type=int)
parser.add_argument('--min', help='minimum occurrence', type=int, default=1)
parser.add_argument('--stopwords', help='stopwords', type=str)
parser.add_argument('--tfidf', help='apply TF-IDF', action='store_true')

args = parser.parse_args()

variables = []
texts = []
word_counts = Counter()
statsA = defaultdict(lambda: Counter())
statsB = defaultdict(lambda: Counter())

if args.stopwords:
    stops = set(stopwords.words(args.stopwords))
else:
    stops = set()

# read in files
print("reading input", file=sys.stderr)
for i, line in enumerate(open(args.input)):
    if i > 0:
        if i % 1000 == 0:
            print('%s' % (i), file=sys.stderr)
        elif i % 100 == 0:
            print('.', file=sys.stderr, end='')

    try:
        variable, text = line.strip().rsplit('\t', 2)
        variables.append(variable)
        text = [word for word in text.split(' ') if word not in stops and len(word) > 1]
        texts.append(text)
        word_counts.update(text)
    except ValueError:
        print("\n\tError in line %s:'%s'" % (i+1, line), file=sys.stderr)
        continue
        
# reduce by min threshold
print("thresholding (min > %s)" %args.min, file=sys.stderr)
texts = [[token for token in text if word_counts[token] > args.min] for text in texts]
print("corpus: %s texts" % (len(texts)), file=sys.stderr)


# get dictionary
print("creating dictionary", file=sys.stderr)
dictionary = corpora.Dictionary(texts)
print(dictionary)
print("translating corpus to IDs", file=sys.stderr)
corpus = [dictionary.doc2bow(text) for text in texts]

if args.tfidf:
    print("translating corpus by TFIDF", file=sys.stderr)
    tfidf = models.TfidfModel(corpus)
    model_corpus = tfidf[corpus]
else:
    model_corpus = corpus

# run topic models
print("fitting model", file=sys.stderr)
<<<<<<< HEAD
model = ldamodel.LdaModel(model_corpus, id2word=dictionary, num_topics=args.topics)
=======
model = ldamodel.LdaModel(model_corpus, id2word=dictionary, num_topics=args.topics, chunksize=int(len(texts)/1000), passes=1)
>>>>>>> 021225024c6411a0a51b2e8ceb31ae279ed9d61c
# transform corpus
print("transforming input", file=sys.stderr)
topic_corpus = model[model_corpus]

for i, topics in enumerate(model.print_topics(num_topics=args.topics, num_words=10)):
    topics = re.sub(topic_sep, '', topics).split(' + ')
    print("T%s:\t%s" % (i, ', '.join(topics)))

# collect stats
print('\nCollecting stats', file=sys.stderr)
for i, doc in enumerate(topic_corpus):
    d = dict(doc)
    for var in variables[i].split(','):
        statsA[var].update(d)
        for key, value in d.items():
            statsB[key][var] += value

# output stats per variable
for variable, counts in sorted(statsA.items()):
    total = sum(counts.values())
    print(variable, '\t'.join(["%s:%.4f" % (key, value/total) for key, value in counts.items()]))

print()

for topic, counts in statsB.items():
    total = sum(counts.values())
    print(topic, '\t'.join(["%s:%.4f" % (key, value/total) for key, value in counts.items()]))

