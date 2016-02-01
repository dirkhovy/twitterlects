import sys
from collections import Counter, defaultdict
from itertools import takewhile
import numpy as np
import json
import argparse
import pandas as pd
from numba import jit
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering


def laskers(x, y):
    return -np.log2(x.dot(y))


def kl(x, y):
    return entropy(x, y)


@jit
def js(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

parser = argparse.ArgumentParser(description="collect data to compare regions")
parser.add_argument('input', help='input files', type=str, default=None, nargs='+')
parser.add_argument('--clusters', help='number of clusters, can be CSV', type=str, default=None)
parser.add_argument('--distance', choices=['kl', 'laskers', 'js'], help='similarity function on vocab', default='js')
parser.add_argument('--geo', help='use geographic distance', action='store_true', default=False)
parser.add_argument('--idf', help='weigh vocabulary terms by IDF', choices=['docs', 'regions'], default=None)
parser.add_argument('--linkage', help='linkage for clustering', choices=['complete', 'ward', 'average'],
                    default='ward')
parser.add_argument('--min_support', help='minimum documents for a region to be counted', type=int, default=10)
parser.add_argument('--nounfilter',
                    help='filter out words that are uppercase at least N% of cases in non-initial contexts (1.0=include all, 0.0=allow no uppercase whatsoever)',
                    default=1.0, type=float)
parser.add_argument('--num_neighbors', help='number of neighbors in coordinate grid', default=5, type=int)
parser.add_argument('--N', help='minimum occurrence of words', type=int, default=10)
parser.add_argument('--prefix', help='output prefix', type=str, default='output')
parser.add_argument('--stem', help='stem words', action='store_true', default=False)

args = parser.parse_args()

review_frequency = Counter()
counts = defaultdict(lambda: defaultdict(lambda: 1))
support = defaultdict(int)
inverted_stems = defaultdict(set)
noun_propensity = defaultdict(int)

regions = None
adjacency = None
region_name2int = None
D = None  # probably obsolete, now that we have neighborhood adjacency
info = []

for input_file in args.input:
    with open(input_file) as model_input:
        model = json.load(model_input)

        review_frequency.update(model['review_frequency'])
        counts.update(model['counts'])
        support.update(model['support'])
        noun_propensity.update(model['noun_propensity'])

        if model['inverted_stems'] is not None:
            with open(model['inverted_stems']) as inverted_stems_file:
                for line in inverted_stems_file:
                    stem, words = line.strip().split('\t')
                    inverted_stems[stem] = inverted_stems[stem].union(set(words.split(', ')))

        # these things should be the same for all, irrespective of what has been observed in the data
        regions = model['regions']
        adjacency_matrix = model['adjacency']
        region_name2int = model['region_name2int']
        D = model['D']
        adjacency = pd.DataFrame(data=adjacency_matrix, index=regions, columns=regions)

        info_bits = model['info']
        for i, info_bit in enumerate(info_bits):
            if info_bit not in set(info):
                info.insert(i, info_bit)


info.extend(['min-freq%s' % args.N, args.distance])
if args.idf:
    info.append('IDF-%s' % args.idf)
if args.geo:
    info.append('geo-distance')
if args.nounfilter:
    info.append('nouns-filtered')

distances = {'kl': kl, 'laskers': laskers, 'js': js}

print(counts)
# filter out nouns
if args.nounfilter:
    non_nouns = set([w for w, c in counts.items() if noun_propensity[w] / sum(c.values()) < args.nounfilter])
    counts = {word: counts[word] for word in non_nouns}
print('Total vocab size: %s' % len(counts), file=sys.stderr)

# throw out words seen less than N times
totals = Counter(dict(((k, sum(v.values())) for k, v in counts.items())))
# get top N words
top_N = [i[0] for i in takewhile(lambda f: f[1] > args.N, totals.most_common())]

# reset topN word count for all regions that have not enough support to 1
for target in regions:
    if support[target] > args.min_support:
        for word in top_N:
            counts[word][target] = 1

# idf transformation before normalization
if args.idf:
    total_D = sum(review_frequency.values())
    for word, region_names in counts.items():
        # normalize by the number of documents it occurred in
        if args.idf == 'docs':
            idf = np.log(total_D / (1 + review_frequency[word]))
        # normalize by the number of regions it occurred in
        else:
            idf = np.log(total_D / (1 + len(counts[word].keys())))

        for target in region_names:
            counts[word][target] *= idf

print('Vocab size with at least %s occurrences: %s' % (args.N, len(top_N)), file=sys.stderr)
print(top_N[:50])


# compute vocab distro per region
print('Computing distribution...', file=sys.stderr)
distros = []
for target in regions:
    frequencies = np.array([counts[word].get(target, 0) for word in top_N])
    distro = frequencies / frequencies.sum()
    distros.append(distro)

# 2 stats
all_distros = np.array(distros)
row_indices = list(range(len(distros)))

print('Computing region differences...', file=sys.stderr)
g2_file = open('%s%s.G2-regions.tsv' % (args.prefix, '.'.join(info)), 'w')
for i, row in enumerate(all_distros):
    rest_indices = row_indices.copy()
    rest_indices.remove(i)
    m = all_distros[rest_indices].mean(axis=0)
    g2 = np.log(row / m)
    top_vocab = g2.argsort()[-50:]

    if args.stem:
        g2_file.write("%s\n\n" % '\n'.join('%s\t%s\t%s' % (regions[i], w, g) for (w, g) in
                                           reversed(list(zip([min(inverted_stems[top_N[x]]) for x in top_vocab],
                                                             [g2[x] for x in top_vocab])))))

    else:
        g2_file.write("%s\n\n" % '\n'.join('%s\t%s\t%s' % (regions[i], w, g) for (w, g) in
                                           reversed(
                                               list(zip([top_N[x] for x in top_vocab], [g2[x] for x in top_vocab])))))

g2_file.close()

# compute matrix L, distance between regions based on vocab distros
print('Computing linguistic distances:', file=sys.stderr)
distance_function = distances[args.distance]
L = pd.DataFrame(0.0, index=regions, columns=regions)
k = 0
# max_computations = int(len(regions) * ((args.num_neighbors*2+1)**2)
for i, x in enumerate(distros):
    r1 = regions[i]

    r1_neighbors = set(adjacency[r1][adjacency[r1] == True].index.tolist())

    for j, y in enumerate(distros[i:]):

        r2 = regions[i + j]

        if r2 not in r1_neighbors:
            continue

        k += 1
        if k > 0:
            if k % 5000 == 0:
                print('%s' % (k), file=sys.stderr)
            elif k % 100 == 0:
                print('.', file=sys.stderr, end='')

        if args.distance == 'js':
            distance = js(x, y)
            L.ix[r1, r2] = distance
            L.ix[r2, r1] = distance
        else:
            L.ix[r1, r2] = distance_function(x, y)
            L.ix[r2, r1] = distance_function(y, x)

print('%s' % (k), file=sys.stderr)

if args.geo:
    print('\nDistances in km:\n', D.to_latex(float_format=lambda x: '%.2f' % x), file=sys.stderr)

Y = L
if args.geo:
    Y = D
    print('\nmulitplied:\n', Y)

# compute clusters over K
print('Computing clusters...', file=sys.stderr)
if args.clusters:
    for num_c in map(int, args.clusters.split(',')):

        clustering = AgglomerativeClustering(linkage=args.linkage, n_clusters=num_c, connectivity=adjacency)
        cluster_names = clustering.fit_predict(Y, adjacency)
        # region2cluster = list(zip(regions, hierarchy.fcluster(row_linkage, num_c, criterion='maxclust')))
        region2cluster = list(zip(regions, cluster_names))

        cluster_file = open('%s%s.%sclusters.tsv' % (args.prefix, '.'.join(info), num_c), 'w')
        cluster_file.write('%s\n' % '\n'.join(('%s\t%s' % (r, clusters) for (r, clusters) in region2cluster)))
        cluster_file.close()

        g2_file = open('%s%s.G2-clusters.%sclusters.tsv' % (args.prefix, '.'.join(info), num_c), 'w')

        for i in range(0, num_c):
            in_cluster = [region_name2int[r] for r, c in region2cluster if c == i]
            rest_indices = row_indices.copy()
            for j in in_cluster:
                rest_indices.remove(j)

            mean_rest = all_distros[rest_indices].mean(axis=0)
            mean_in = all_distros[in_cluster].mean(axis=0)

            g2 = np.log(mean_in / mean_rest)
            top_vocab = g2.argsort()[-50:]
            g2_file.write('Cluster %s: %s\n' % (i, ', '.join([str(regions[x]) for x in in_cluster])))
            if args.stem:
                g2_file.write("%s\n\n" % '\n'.join('\t%s\t%s' % (w, g) for (w, g) in reversed(
                    list(zip([min(inverted_stems[top_N[x]]) for x in top_vocab], [g2[x] for x in top_vocab])))))
            else:
                g2_file.write("%s\n\n" % '\n'.join('\t%s\t%s' % (w, g) for (w, g) in reversed(
                    list(zip([top_N[x] for x in top_vocab], [g2[x] for x in top_vocab])))))

        g2_file.close()
