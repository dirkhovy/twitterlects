import argparse
import json
import re
import sys
from collections import defaultdict, Counter
from itertools import islice, takewhile
from math import radians, cos
import fiona
import matplotlib.pyplot as plt
import nltk.data
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from numpy.linalg import norm
from scipy.stats import entropy
from shapely.geometry import shape
from sklearn.cluster import AgglomerativeClustering

sns.set(font="monospace")
sns.set_context('poster')

EARTH_RADIUS = 6371
numbers = re.compile(r"[0123456789]")
names = re.compile(r"@[^ ]*")
urls = re.compile(r"http[^ ]*")

# latitudes are the bigger numbers, going N-S, longitude are the smaller numbers, going W-E
country_boxes = {
    'denmark': [54.5, 58., 7.5, 15.5],
    'germany': [47., 55.5, 5.5, 15.5],
    'france': [41., 51.5, -5.5, 10.],
    'usa_all': [18., 72., -180., -67.5],
    'usa': [24., 50., -125., -65.5],
    'world': [-60., 75., -179., 179.],
    'europe': [36., 70., -11., 40.],
    'uk': [49.5, 61., -11., 2.]
}


def laskers(x, y):
    return -np.log2(x.dot(y))


def kl(x, y):
    return entropy(x, y)


def js(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def get_shortest_in(needle, haystack):
    '''

    :param needle: single (lat,long) tuple.
    :param haystack: numpy array to find the point in that has the shortest distance to needle
    :return:
    '''
    dlat = radians(haystack[0]) - radians(needle[0])
    dlon = radians(haystack[1]) - radians(needle[1])
    a = np.square(np.sin(dlat / 2.0)) + cos(radians(needle[0])) * np.cos(radians(haystack[0])) * np.square(
        np.sin(dlon / 2.0))
    great_circle_distance = 2 * np.arcsin(np.minimum(np.sqrt(a), np.repeat(1, 1)))
    d = EARTH_RADIUS * great_circle_distance
    return d.tolist()


parser = argparse.ArgumentParser(description="compare regions")
parser.add_argument('--trustpilot', help='input file')
parser.add_argument('--twitter', help='input file')
parser.add_argument('--country', choices=['denmark', 'germany', 'france'], help='which country to use',
                    default='denmark')
parser.add_argument('--nuts', help='NUTS regions shape file',
                    default="/Users/dirkhovy/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp")
parser.add_argument('--nuts_level', help='NUTS level', type=int, default=2)
parser.add_argument('--N', help='minimum occurrence of words', type=int, default=10)
parser.add_argument('--limit', help='max instances', type=int, default=None)
parser.add_argument('--clusters', help='number of clusters, can be CSV', type=str, default=None)
parser.add_argument('--prefix', help='output prefix', type=str, default='output')
parser.add_argument('--idf', help='weigh vocabulary terms by IDF', action='store_true', default=False)
parser.add_argument('--show', help='show dendrogram', action='store_true', default=False)
parser.add_argument('--stem', help='stem words', action='store_true', default=False)
parser.add_argument('--geo', help='use geographic distance', action='store_true', default=False)
parser.add_argument('--distance', choices=['kl', 'laskers', 'js'], help='similarity function on vocab', default='js')
parser.add_argument('--target', choices=['region', 'gender', 'coords'], help='target variable', default='region')
parser.add_argument('--nounfilter', help='filter out words that are uppercase 90% of cases', action='store_true')

args = parser.parse_args()

distances = {'kl': kl, 'laskers': laskers, 'js': js}
country2lang = {'denmark': 'danish', 'germany': 'german', 'france': 'french'}
country2nuts = {'denmark': 'DK', 'germany': 'DE', 'france': 'FR'}

info = [args.country, 'min%s' % args.N, 'NUTS-%s' % args.nuts_level, args.distance]
if args.idf:
    info.append('IDF')
if args.stem:
    info.append('stemmed')
if args.geo:
    info.append('geo-distance')

regions = []
region_centers = {}

word_tokenizer = WordPunctTokenizer()
sentence_tokenizer = nltk.data.load('tokenizers/punkt/%s.pickle' % (country2lang[args.country]))
stemmer = SnowballStemmer(country2lang[args.country])

inverted_stems = defaultdict(set)
noun_propensity = defaultdict(int)


# determine target of the aggregation: gender, NUTS region, or geo-coordinate
if args.target == 'region':
    print("reading NUTS...", file=sys.stderr)
    fiona_shapes = fiona.open(args.nuts)
    shapes = []
    for item in islice(fiona_shapes, None):
        if item['properties']['STAT_LEVL_'] == args.nuts_level:
            nuts_id = item['properties']['NUTS_ID']
            if nuts_id.startswith(country2nuts[args.country]):
                # if nuts_id == 'DK014':
                #     continue

                regions.append(nuts_id)
                nuts_shape = shape(item['geometry'])
                shapes.append(nuts_shape)
                region_centers[nuts_id] = list(nuts_shape.centroid.coords)[0]

    regions = sorted(regions)

    print("computing adjacency...", file=sys.stderr)
    adjacency = pd.DataFrame(False, index=regions, columns=regions)
    for i in range(len(shapes)):
        nuts_shape = shapes[i].buffer(0.1)
        for j in range(len(shapes)):
            if i == j:
                continue
            adjacency.iloc[i, j] = nuts_shape.intersects(shapes[j])
            # TODO: what about the inverse?

    print(adjacency, file=sys.stderr)

# geo-coordinates are 0.1 increments of lat-lng pairs, based on the country's bounding box
elif args.target == 'coords':
    country_lats = np.arange(country_boxes[args.country][0], country_boxes[args.country][1], 0.2)
    country_lngs = np.arange(country_boxes[args.country][2], country_boxes[args.country][3], 0.2)

    for lat in country_lats:
        lat = float('%.1f' % lat)
        for lng in country_lngs:
            lng = float('%.1f' % lng)
            regions.append((lat, lng))
    print("%s regions" % len(regions))

    print("computing adjacency...", file=sys.stderr)
    adjacency = pd.DataFrame(False, index=regions, columns=regions)
    regions_set = set(regions)
    covered = set()
    for i, (lat1, lng1) in enumerate(regions):
        if i > 0 and i%100 == 0:
            print(i, file=sys.stderr)
        elif i%10 == 0:
            print('.', file=sys.stderr, end='')

        neighbors = [(lat1+lat_dist, lng1+lng_dist) for lat_dist in [-0.2, 0.0, 0.2] for lng_dist in [-0.2, 0.0, 0.2] if (lat_dist != 0.0 or lng_dist != 0.0)]
        neighbors = set(neighbors).difference(covered)
        neighbors = neighbors.intersection(regions_set)
        covered.update(neighbors)

        for neighbor in neighbors:
            adjacency.ix[(lat1, lng1), neighbor] = True
            adjacency.ix[neighbor, (lat1, lng1)] = True

    print(adjacency, file=sys.stderr)

else:
    regions = ['F', 'M']

region_name2int = dict([(name, i) for (i, name) in enumerate(regions)])
review_frequency = Counter()


# compute matrix D, distances between regions
if args.geo:
    D = pd.DataFrame(0.0, index=regions, columns=regions)  # np.zeros((len(regions), len(regions)), dtype=float)

    if args.target == 'region':
        for i, r1 in enumerate(regions):
            for j, r2 in enumerate(regions[i + 1:]):
                x = get_shortest_in(region_centers[r1], np.array(region_centers[r2]))[0]
                # x = np.log(get_shortest_in(region_centers[r1], np.array(region_centers[r2]))[0])
                D.ix[r1, r2] = x
                D.ix[r2, r1] = x
    elif args.target == 'coords':
        for i, lat_lng1 in enumerate(regions):
            for j, lat_lng2 in enumerate(regions[i + 1:]):
                x = get_shortest_in(lat_lng1, lat_lng2)[0]
                D.ix[lat_lng1, lat_lng2] = x
                D.ix[lat_lng2, lat_lng1] = x


# collect total vocab
counts = defaultdict(lambda: defaultdict(lambda: 1))

if args.trustpilot:
    for line_no, line in enumerate(islice(open(args.trustpilot), None)):
        if line_no > 0:
            if line_no % 1000 == 0:
                print("%s" % (line_no), file=sys.stderr)
            elif line_no % 100 == 0:
                print('.', file=sys.stderr, end=' ')

        if args.limit and line_no == args.limit:
            break

        try:
            user = json.loads(line)
            reviews = user.get('reviews', None)
            if args.target == 'region':
                target = user['NUTS-%s' % args.nuts_level]
            elif args.target == 'coords':
                user_lat = '%.1f' % float(user['geocodes'][0]['lat'])
                user_lng = '%.1f' % float(user['geocodes'][0]['lng'])
                target = (float(user_lat), float(user_lng))
            else:
                target = user.get('gender', None)

            # if target == 'DK014':
            #     continue

            for review in reviews:
                body = review.get('text', None)
                # exclude empty reviews
                if body:
                    for text in body:
                        text = re.sub(numbers, '0', text)
                        text = re.sub(urls, '', text)
                        text = re.sub(r'\n', ' ', text)
                        # TODO: better stopword filter
                        org_words = (' '.join([' '.join(word_tokenizer.tokenize(x)) for x in
                                           sentence_tokenizer.tokenize(text)])).split()
                        words_lower = list(map(str.lower, org_words))

                        if args.stem:
                            stemmed_words = list(map(stemmer.stem, org_words))
                            # update inverted index
                            for stem, word in zip(stemmed_words, words_lower):
                                inverted_stems[stem].add(word)

                            words = list(filter(lambda word: word != '', stemmed_words))
                        else:
                            words = org_words


                        for w, word in enumerate(words):
                            if org_words[w][0].isupper():
                                noun_propensity[word] += 1

                            counts[word][target] += 1
                        review_frequency.update(set(words))

        except ValueError:
            continue
        except KeyError:
            continue

if args.twitter:
    for line_no, line in enumerate(islice(open(args.twitter), None)):
        if line_no > 0:
            if line_no % 1000 == 0:
                print("%s" % (line_no), file=sys.stderr)
            elif line_no % 100 == 0:
                print('.', file=sys.stderr, end=' ')

        if args.limit and line_no == args.limit:
            break

        try:
            user = json.loads(line)
            body = user.get('body', None)

            # exclude empty tweets
            if body is None:
                continue

            ############################
            # retrieve relevant fields #
            ############################
            languages = []
            try:
                gnip_lang = user['gnip']['language']['value']
                languages.append(gnip_lang)
            except KeyError:
                pass
            try:
                twitter_lang = user['twitter_lang']
                languages.append(twitter_lang)
            except KeyError:
                pass
            # try:
            #     user_lang = user['actor']['languages']
            #     languages.extend(user_lang)
            # except KeyError:
            #     pass
            try:
                lang_id = user['lang_id'][0]
                languages.append(lang_id)
            except KeyError:
                pass

            if 'da' not in languages:
                continue

            if args.target == 'region':
                user_regions = user['actor']['NUTS%s' % args.nuts_level]['region']
            elif args.target == 'coords':
                #TODO: get user coords, not tweet coords
                user_regions = []
                pass

            if body:
                text = re.sub(numbers, '0', body)
                text = re.sub(r'\n', ' ', text)
                text = re.sub(names, '', text)
                text = re.sub(urls, '', text)

                # TODO: better stopword filter
                org_words = (' '.join([' '.join(word_tokenizer.tokenize(x)) for x in
                                   sentence_tokenizer.tokenize(text)])).split()
                words_lower = list(map(str.lower, org_words))

                if args.stem:
                    stemmed_words = list(map(stemmer.stem, org_words))
                    # update inverted index
                    for stem, word in zip(stemmed_words, words_lower):
                        inverted_stems[stem].add(word)

                    words = list(filter(lambda word: word != '', stemmed_words))
                else:
                    words = org_words


                for w, word in enumerate(words):
                    if org_words[w][0].isupper():
                        noun_propensity[word] += 1

                    counts[word][target] += 1
                review_frequency.update(set(words))

        except ValueError:
            continue
        except KeyError:
            continue

if args.nounfilter:
    non_nouns = set([w for w, c in counts.items() if noun_propensity[w]/sum(c.values()) < 0.9])
    counts = {word: counts[word] for word in non_nouns}
print('Total vocab size: %s' % len(counts), file=sys.stderr)

# throw out words seen less than N times
totals = Counter(dict(((k, sum(v.values())) for k, v in counts.items())))
# get top N words
top_N = [i[0] for i in takewhile(lambda f: f[1] > args.N, totals.most_common())]

# idf transformation before normalization
if args.idf:
    total_D = sum(review_frequency.values())
    for word, region_names in counts.items():
        idf = np.log(total_D / (1 + review_frequency[word]))
        for target in region_names:
            counts[word][target] *= idf

print('Vocab size with at least %s occurrences: %s' % (args.N, len(top_N)), file=sys.stderr)
print(top_N[:50])

if args.stem:
    print('Writing stem indices...', file=sys.stderr)
    stem_file = open('%s%s.inverted_stems.tsv' % (args.prefix, '.'.join(info)), 'w')
    for stem, words in inverted_stems.items():
        stem_file.write("%s\t%s\n" % (stem, ', '.join(words)))
    stem_file.close()
    print('done', file=sys.stderr)

print('Computing distribution...', file=sys.stderr)

# compute vocab distro per region
distros = []
for target in regions:
    frequencies = np.array([counts[word][target] for word in top_N])
    distro = frequencies / frequencies.sum()
    distros.append(distro)

# 2 stats
all_distros = np.array(distros)
row_indices = list(range(len(distros)))

g2_file = open('%s%s.G2-regions.tsv' % (args.prefix, '.'.join(info)), 'w')

for i, row in enumerate(all_distros):
    rest_indices = row_indices.copy()
    rest_indices.remove(i)
    m = all_distros[rest_indices].mean(axis=0)
    g2 = np.log(row / m)
    top_vocab = g2.argsort()[-50:]

    if args.stem:
        g2_file.write("%s\n\n" % '\n'.join('%s\t%s\t%s' % (regions[i], w, g) for (w, g) in
                                       reversed(list(zip([min(inverted_stems[top_N[x]]) for x in top_vocab], [g2[x] for x in top_vocab])))))

    else:
        g2_file.write("%s\n\n" % '\n'.join('%s\t%s\t%s' % (regions[i], w, g) for (w, g) in
                                       reversed(list(zip([top_N[x] for x in top_vocab], [g2[x] for x in top_vocab])))))

g2_file.close()

# compute matrix L, distance between regions based on vocab distros
distance_function = distances[args.distance]
L = pd.DataFrame(0.0, index=regions, columns=regions)
for i, x in enumerate(distros):
    for j, y in enumerate(distros[i:]):
        r1 = regions[i]
        r2 = regions[i + j]
        L.ix[r1, r2] = distance_function(x, y)
        L.ix[r2, r1] = distance_function(y, x)

if args.geo:
    print('\nDistances in km:\n', D.to_latex(float_format=lambda x: '%.2f' % x), file=sys.stderr)

# print("\nLinguistic distances:\n", L.to_latex(), file=sys.stderr)

# C = (L).corr()#L.corrwith(D, axis=1)
# print('\ncorrelation:\n', C)

# combine L and D in matrix K, either pointwise or concatenated
# LS = L.stack()
# DS = D.stack()
#
# X = pd.DataFrame(data={'LS':LS, 'DS':DS})
# X['region'] = [tup[0] for tup in X.index.get_values()]
# Y = X.groupby('region').mean()
Y = L
if args.geo:
    Y = D
    print('\nmulitplied:\n', Y)

# row_linkage = hierarchy.linkage(
#     distance.pdist(Y), method='average')
#
# col_linkage = hierarchy.linkage(
#     distance.pdist(Y.T), method='average')
#
# seaborn.clustermap(Y, col_cluster=False, standard_scale=1,row_linkage=row_linkage, col_linkage=col_linkage)

# compute clusters over K
if args.clusters:
    for num_c in map(int, args.clusters.split(',')):

        clustering = AgglomerativeClustering(linkage='complete', n_clusters=num_c, connectivity=adjacency)
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

# plot clusters on map
if args.show:
    plt.show()
