import argparse
from collections import Counter
from gensim import corpora
from gensim.models import ldamodel


parser = argparse.ArgumentParser(description="""
takes as input a TSV file with a variable and texts, computes topics, and shows the distribution over topics for each variable
each line is one document
documents are assumed to be tokenized
""")

parser.add_argument('input', help='input file')
parser.add_argument('topics', help='number of topics', type=int)
parser.add_argument('--min', help='minimum occurrence', type=int, default=1)

args = parser.parse_args()

# read in files
variables = []
texts = []
word_counts = Counter()
for line in open(args.input):
    variable, text = line.strip().split('\t')
    variables.append(variable)
    text = text.split(' ')
    texts.append(text)
    word_counts.update(text)

# reduce by min threshold
texts = [[token for token in text if word_counts[token] > args.min] for text in texts]


# get dictionary
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# run topic models
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=args.topics)
# transform corpus
topic_corpus = model[corpus]


# collect stats
print(topic_corpus)

# output stats per variable