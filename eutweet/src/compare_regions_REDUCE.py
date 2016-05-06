import sys
from collections import defaultdict
from itertools import count
import numpy as np
import json
import argparse
import pandas as pd
import time
from numba import jit
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from scipy import sparse, io
from sklearn.preprocessing import normalize


def laskers(x, y):
    return -np.log2(x.dot(y))


def kl(x, y):
    return entropy(x, y)

@jit
def js(_P, _Q):
    # _P = P / norm(P, ord=1)
    # _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

distances = {'kl': kl, 'laskers': laskers, 'js': js}


parser = argparse.ArgumentParser(description="collect data to compare regions")
parser.add_argument('input', help='input files', type=str, default=None, nargs='+')
parser.add_argument('--coord_size', help='size of coordinate grid in degrees', default=0.1, type=float)
parser.add_argument('--country', choices=['denmark', 'germany', 'france', 'uk', 'eu'], help='which country to use',
                    default='eu')
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

info = []
rows = []
cols = []
values = []
support = defaultdict(int)
word2int_count = count()
word2int = defaultdict(word2int_count.__next__)

regions_file = '%s-%s.regions.json' % (args.country, args.coord_size)
try:
    with open(regions_file) as region_input:
        regions = list(map(tuple, json.load(region_input)))
except OSError:
    print("ERROR: regions file %s not found! Please pre-compute" % ('%s-%s.regions.csv' % (args.country, args.coord_size)), file=sys.stderr, flush=True)

print("%s regions" % len(regions), file=sys.stderr, flush=True)
region_name2int = dict([(name, i) for (i, name) in enumerate(regions)])

adjacency = pd.read_csv('%s-%s.adjacency.csv' % (args.country, args.coord_size))

# read counts
for input_file in args.input:
    with open(input_file) as model_input:
        model = json.load(model_input)
        support.update(model['support'])
        model_words = model['words']

        counts_model = io.mmread()
        model_rows = counts_model.indices
        model_cold = counts_model.indptr
        model_values = [word2int[model_words[i]] for i in counts_model.data]

        info_bits = model['info']
        for i, info_bit in enumerate(info_bits):
            if info_bit not in set(info):
                info.insert(i, info_bit)

print("\ncreating sparse count matrix", file=sys.stderr, flush=True)
counts = sparse.coo_matrix((values, (rows, cols)), dtype=np.float, shape=(len(regions), len(word2int))).tocsr()
num_inst, num_feats = counts.shape
print("\n%s regions, %s features" % (num_inst, num_feats), file=sys.stderr, flush=True)


info.extend(['min-freq%s' % args.N, args.distance])
if args.idf:
    info.append('IDF-%s' % args.idf)
if args.geo:
    info.append('geo-distance')
if args.nounfilter:
    info.append('nouns-filtered')


int2word = {i: word for word, i in word2int.items()}

# throw out words seen less than N times
totals = counts.sum(axis=0)
# get top N words
# top_N = [i[0] for i in takewhile(lambda f: f[1] > args.N, totals.most_common())]
top_N = np.argwhere(totals > args.N)[:, 1]
print('Vocab size with at least %s occurrences: %s' % (args.N, len(top_N)), file=sys.stderr, flush=True)
counts = counts[:, top_N]
# keep track of what the original word ID for each position in the reduced matrix was
reduced2orgID = {i: int2word[value] for i, value in enumerate(top_N)}
num_inst, num_feats = counts.shape
print("\nreduced to %s words" % (num_feats), file=sys.stderr, flush=True)


# reset topN word count for all regions that have not enough support to 1
ignore_regions = {target for target in regions if support[region_name2int[target]] < args.min_support}
print('found %s unsupported regions (fewer than %s entries), leaving %s supported land regions...' % (len(ignore_regions), args.min_support, len(regions) - len(ignore_regions)),
      file=sys.stderr, flush=True)
with open('%s%s.support.tsv' % (args.prefix, '.'.join(info)), 'w') as support_file:
    support_file.write('\n'.join(["%s\t%s" % (target, support[region_name2int[target]]) for target in regions]))



# idf transformation before normalization
if args.idf:
    start = time.time()
    print('Computing IDF values...', file=sys.stderr, flush=True)
    if args.idf == 'docs':
        idf = np.log(counts.sum() / (counts.sum(axis=0)))
    # normalize by the number of regions it occurred in
    else:
        idf = np.log(counts.sum() / (np.bincount(counts.nonzero()[1])))

    counts = counts.multiply(idf)
    print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)

# permute count matrix to check whether outcome is still sensible
if args.random:
    counts = np.random.random((counts.shape))
    print(
        '\n\n****************** WARNING ******************\n*         USING RANDOM PERMUTATION          *\n*********************************************\n\n',
        file=sys.stderr, flush=True)
    info.append('RANDOM')

# compute vocab distro per region
start = time.time()
print('Computing distribution...', file=sys.stderr, flush=True)
distros = normalize(counts, norm='l1', axis=1)
print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)

# compute matrix L, distance between regions based on vocab distros
start = time.time()
print('Computing linguistic distances:', file=sys.stderr, flush=True)
distance_function = distances[args.distance]
L = pd.DataFrame(0.0, index=regions, columns=regions)
k = 0
# max_computations = int(len(regions) * ((args.num_neighbors*2+1)**2)
for i, x in enumerate(distros):

    r1 = regions[i]
    if r1 in ignore_regions or r1 not in land_regions:
        continue

    r1_neighbors = set(adjacency[r1][adjacency[r1] == True].index.tolist())

    for j, y in enumerate(distros[i:]):

        r2 = regions[i + j]
        if r2 not in r1_neighbors or r2 in ignore_regions:
            continue

        k += 1
        if k > 0:
            if k % 5000 == 0:
                print('%s' % (k), file=sys.stderr, flush=True)
            elif k % 100 == 0:
                print('.', file=sys.stderr, flush=True, end='')

        if args.distance == 'js':
            distance = js(x, y)
            L.ix[r1, r2] = distance
            L.ix[r2, r1] = distance
        else:
            L.ix[r1, r2] = distance_function(x, y)
            L.ix[r2, r1] = distance_function(y, x)
print('%s' % (k), file=sys.stderr, flush=True)
print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)

Y = L

# compute clusters over K
all_distros = np.copy(distros)
row_indices = list(range(distros.shape[0]))
start = time.time()
print('Computing clusters...', file=sys.stderr, flush=True)
if args.clusters:
    for num_c in map(int, args.clusters.split(',')):

        clustering = AgglomerativeClustering(linkage=args.linkage, n_clusters=num_c, connectivity=adjacency)
        cluster_names = clustering.fit_predict(Y, adjacency)
        region2cluster = list(zip(regions, cluster_names))

        cluster_file = open('%s%s.%sclusters.tsv' % (args.prefix, '.'.join(info), num_c), 'w')
        cluster_file.write('%s\n' % '\n'.join(('%s\t%s' % (r, clusters) for (r, clusters) in region2cluster)))
        cluster_file.close()

        g2_file = open('%s%s.G2.%sclusters.tsv' % (args.prefix, '.'.join(info), num_c), 'w')

        for i in range(0, num_c):
            in_cluster = []
            rest_indices = []
            for r, c in region2cluster:
                if r in land_regions and r not in ignore_regions:
                    if c == i:
                        in_cluster.append(region_name2int[r])
                    else:
                        rest_indices.append(region_name2int[r])

            mean_rest = all_distros[rest_indices].mean(axis=0)
            mean_in = all_distros[in_cluster].mean(axis=0)

            g2 = np.log(mean_in / mean_rest)
            # remove NaN, inf, and -inf
            g2 = np.where(np.isnan(g2), 0, g2)
            g2 = np.where(g2 == float('inf'), 0, g2)
            g2 = np.where(g2 == float('-inf'), 0, g2)

            # select the top 50, or all non-zero values
            top_50 = g2.argsort()[-min(50, len(g2.nonzero()[0])):]
            top_vocab = [reduced2orgID[x] for x in top_50]

            g2_file.write('Cluster %s: %s\n' % (i, ', '.join([str(regions[x]) for x in in_cluster])))
            g2_file.write("%s\n\n" % '\n'.join('\t%s\t%s' % (w, g) for (w, g) in reversed(
                list(zip([x for x in top_vocab], [g2[x] for x in top_50])))))

        g2_file.close()
print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)
