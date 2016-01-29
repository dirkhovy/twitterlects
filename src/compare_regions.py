import argparse
import bisect
import json
# import ujson as json
import re
import sys
import fiona
import matplotlib.pyplot as plt
import nltk.data
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from numba import jit
from numpy.linalg import norm
from scipy.stats import entropy
from shapely.geometry import shape, Point, Polygon
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict, Counter
from itertools import islice, takewhile
from math import radians, cos
from nltk.corpus import stopwords

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


@jit
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
parser.add_argument('--bigrams', help='use bigrams', action="store_true")
parser.add_argument('--clusters', help='number of clusters, can be CSV', type=str, default=None)
parser.add_argument('--coord_size', help='size of coordinate grid in degrees', default=0.1, type=float)
parser.add_argument('--country', choices=['denmark', 'germany', 'france', 'uk'], help='which country to use',
                    default='denmark')
parser.add_argument('--distance', choices=['kl', 'laskers', 'js'], help='similarity function on vocab', default='js')
parser.add_argument('--geo', help='use geographic distance', action='store_true', default=False)
parser.add_argument('--idf', help='weigh vocabulary terms by IDF', choices=['docs', 'regions'], default=None)
parser.add_argument('--limit', help='max instances', type=int, default=None)
parser.add_argument('--linkage', help='linkage for clustering', choices=['complete', 'ward', 'average'],
                    default='complete')
parser.add_argument('--min_support', help='minimum documents for a region to be counted', type=int, default=100)
parser.add_argument('--nounfilter',
                    help='filter out words that are uppercase at least N% of cases in non-initial contexts (1.0=include all, 0.0=allow no uppercase whatsoever)',
                    default=1.0, type=float)
parser.add_argument('--num_neighbors', help='number of neighbors in coordinate grid', default=5, type=int)
parser.add_argument('--nuts', help='NUTS regions shape file',
                    default="/Users/dirkhovy/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp")
parser.add_argument('--nuts_level', help='NUTS level', type=int, default=2)
parser.add_argument('--N', help='minimum occurrence of words', type=int, default=10)
parser.add_argument('--prefix', help='output prefix', type=str, default='output')
parser.add_argument('--show', help='show dendrogram', action='store_true', default=False)
parser.add_argument('--stem', help='stem words', action='store_true', default=False)
parser.add_argument('--stopwords', help='stopwords', type=str)
parser.add_argument('--target', choices=['region', 'gender', 'coords'], help='target variable', default='region')

args = parser.parse_args()

distances = {'kl': kl, 'laskers': laskers, 'js': js}
country2lang = {'denmark': 'danish', 'germany': 'german', 'france': 'french', 'uk': 'english'}
country2nuts = {'denmark': 'DK', 'germany': 'DE', 'france': 'FR', 'uk': 'UK'}
country_lats = np.arange(country_boxes[args.country][0], country_boxes[args.country][1], args.coord_size)
country_lngs = np.arange(country_boxes[args.country][2], country_boxes[args.country][3], args.coord_size)

if args.stopwords:
    stops = set(stopwords.words(args.stopwords))
else:
    stops = set()

info = [args.country, 'min%s' % args.N, 'NUTS-%s' % args.nuts_level, args.distance, args.target]
if args.trustpilot:
    info.append('Trustpilot')
if args.twitter:
    info.append('Twitter')
if args.bigrams:
    info.append('bigrams')
if args.idf:
    info.append('IDF-%s' % args.idf)
if args.stem:
    info.append('stemmed')
if args.geo:
    info.append('geo-distance')
if args.nounfilter:
    info.append('nouns-filtered')
if args.stopwords:
    info.append('stopword-filtered')
if args.target == 'coords':
    info.append('%s-neighbors.size-%s' % (args.num_neighbors, args.coord_size))

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
            adjacency.iloc[j, i] = shapes[j].intersects(nuts_shape)
            # TODO: what about the inverse?

    print(adjacency, file=sys.stderr)

# geo-coordinates are 0.1 increments of lat-lng pairs, based on the country's bounding box
elif args.target == 'coords':
    num_lats = country_lats.shape[0]
    num_lngs = country_lngs.shape[0]

    for lat in range(num_lats):
        for lng in range(num_lngs):
            regions.append((lat, lng))
    print("%s regions" % len(regions))

    print("computing adjacency...", file=sys.stderr)
    adjacency = pd.DataFrame(False, index=regions, columns=regions)
    regions_set = set(regions)
    covered = set()
    num_neighbors = args.num_neighbors
    for i in range(num_lats):
        for j in range(num_lngs):
            neighbors = [(i + lat_iterator, j + lng_iterator) for lat_iterator in
                         list(range(-num_neighbors, num_neighbors + 1)) for lng_iterator in
                         list(range(-num_neighbors, num_neighbors + 1))]
            for neighbor in neighbors:
                if neighbor in regions_set and neighbor != (i, j):
                    adjacency.ix[(i, j), neighbor] = True
                    adjacency.ix[neighbor, (i, j)] = True
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
        for i in enumerate(regions):
            lat_lng1 = (country_lats[regions[i][0]], country_lngs[regions[i][1]])
            for j in enumerate(regions[i + 1:]):
                lat_lng2 = (country_lats[regions[j][0]], country_lngs[regions[j][1]])
                x = get_shortest_in(lat_lng1, lat_lng2)[0]
                D.ix[lat_lng1, lat_lng2] = x
                D.ix[lat_lng2, lat_lng1] = x

# collect total vocab
counts = defaultdict(lambda: defaultdict(lambda: 1))
support = defaultdict(int)

if args.trustpilot:
    print("\nProcessing Trustpilot", file=sys.stderr)
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
                user_lat = '%.3f' % float(user['geocodes'][0]['lat'])
                user_lng = '%.3f' % float(user['geocodes'][0]['lng'])
                target = (bisect.bisect(country_lats, float(user_lat)), bisect.bisect(country_lngs, float(user_lng)))
            else:
                target = user.get('gender', None)

            for review in reviews:
                body = review.get('text', None)
                # exclude empty reviews
                if body:
                    for text in body:
                        # increase support for this region
                        support[target] += 1

                        text = re.sub(numbers, '0', text)
                        text = re.sub(urls, '', text)
                        text = re.sub(r'\n', ' ', text)

                        org_words = (' '.join([' '.join(word_tokenizer.tokenize(x)) for x in
                                               sentence_tokenizer.tokenize(text)])).split()
                        org_words = [word for word in org_words if word.lower() not in stops and len(word) > 1]
                        words_lower = [word.lower() for word in org_words]

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

                        if args.bigrams:
                            bigrams = [' '.join(bigram) for bigram in nltk.bigrams(words) if
                                       ' '.join(bigram).strip() is not '']
                            for bigram in bigrams:
                                counts[bigram][target] += 1
                                if args.stem:
                                    inverted_stems[bigram].add(bigram)
                            review_frequency.update(set(bigrams))


        except ValueError:
            continue
        except KeyError:
            continue

if args.twitter:
    print("\nProcessing Twitter", file=sys.stderr)
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

            if 'da' not in languages and 'no' not in languages:
                continue

            if args.target == 'region':
                user_regions = user['actor']['NUTS%s' % args.nuts_level]['region']

            elif args.target == 'coords':
                user_regions = []
                try:
                    # lat, lng
                    user_regions = [tuple(user['geo']['coordinates'])]
                except KeyError:
                    try:
                        coords = list(Point(Polygon(user['location']['geo']['coordinates'][0]).centroid).coords)[0]
                        user_regions = [(coords[1], coords[0])]
                    except KeyError:
                        pass
                    except NotImplementedError:
                        pass

                user_regions = [(bisect.bisect(country_lats, float('%.3f' % user_lat)),
                                 bisect.bisect(country_lngs, float('%.3f' % user_lng))) for (user_lat, user_lng) in
                                user_regions]

            if body:
                # increase support for this region
                support[target] += 1

                text = re.sub(numbers, '0', body)
                text = re.sub(r'\n', ' ', text)
                text = re.sub(names, '', text)
                text = re.sub(urls, '', text)

                org_words = (' '.join([' '.join(word_tokenizer.tokenize(x)) for x in
                                       sentence_tokenizer.tokenize(text)])).split()
                org_words = [word for word in org_words if word.lower() not in stops and len(word) > 1]

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
                    if w > 0 and org_words[w][0].isupper() and not org_words[w].isupper():
                        noun_propensity[word] += 1
                    for target in user_regions:
                        counts[word][target] += 1
                review_frequency.update(set(words))

                if args.bigrams:
                    bigrams = [' '.join(bigram) for bigram in nltk.bigrams(words) if ' '.join(bigram).strip() is not '']
                    for bigram in bigrams:
                        counts[bigram][target] += 1
                        if args.stem:
                            inverted_stems[bigram].add(bigram)
                    review_frequency.update(set(bigrams))


        except ValueError as ve:
            # print(ve, file=sys.stderr)
            continue
        except KeyError as ke:
            # print(ke, file=sys.stderr)
            continue


#==========================================================
# split here:
# save counts and support

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

if args.stem:
    print('Writing stem indices...', file=sys.stderr)
    stem_file = open('%s%s.inverted_stems.tsv' % (args.prefix, '.'.join(info)), 'w')
    for stem, words in inverted_stems.items():
        stem_file.write("%s\t%s\n" % (stem, ', '.join(words)))
    stem_file.close()
    print('done', file=sys.stderr)

# compute vocab distro per region
print('Computing distribution...', file=sys.stderr)
distros = []
for target in regions:
    frequencies = np.array([counts[word][target] for word in top_N])
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

# plot clusters on map
if args.show:
    plt.show()
