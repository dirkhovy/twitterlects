import argparse
import bisect
import json
import re
import sys
import fiona
import matplotlib.pyplot as plt
import nltk.data
import numpy as np
import pandas as pd
import seaborn as sns
import time
from mpl_toolkits.basemap import Basemap
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from numba import jit
from scipy.stats import entropy
from shapely.geometry import shape, Point, Polygon
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict, Counter
from itertools import islice
from math import radians, cos
from nltk.corpus import stopwords
from itertools import count
from scipy import sparse
from sklearn.preprocessing import normalize

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
    _M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, _M) + entropy(Q, _M))


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


def neighbors(x, y, M, n):
    X, Y = M.shape
    return [(x2, y2) for x2 in range(x-n, x+n+1)
                               for y2 in range(y-n, y+n+1)
                               if ((0 <= x < X) and
                                   (0 <= y < Y) and
                                   (x != x2 or y != y2) and
                                   (0 <= x2 < X) and
                                   (0 <= y2 < Y) and
                                   M[x,y] != 0 and
                                   M[x2,y2] != 0
                                  )]


parser = argparse.ArgumentParser(description="compare regions")
parser.add_argument('--trustpilot', help='input file')
parser.add_argument('--twitter', help='input file')
parser.add_argument('--bigrams', help='use bigrams', action="store_true")
parser.add_argument('--chars', help='use character n-grams', action="store_true")
parser.add_argument('--clusters', help='number of clusters, can be CSV', type=str, default=None)
parser.add_argument('--coord_size', help='size of coordinate grid in degrees', default=0.1, type=float)
parser.add_argument('--country', choices=['denmark', 'germany', 'france', 'uk'], help='which country to use',
                    default='denmark')
parser.add_argument('--distance', choices=['kl', 'laskers', 'js'], help='similarity function on vocab', default='js')
parser.add_argument('--geo', help='use geographic distance', action='store_true', default=False)
parser.add_argument('--idf', help='weigh vocabulary terms by IDF', choices=['docs', 'regions'], default=None)
parser.add_argument('--limit', help='max instances', type=int, default=None)
parser.add_argument('--linkage', help='linkage for clustering', choices=['complete', 'ward', 'average'],
                    default='ward')
parser.add_argument('--min_support', help='minimum documents for a region to be counted', type=int, default=10)
parser.add_argument('--nounfilter',
                    help='filter out words that are uppercase at least N% of cases in non-initial contexts (1.0=include all, 0.0=allow no uppercase whatsoever)',
                    default=1.0, type=float)
parser.add_argument('--num_neighbors', help='number of neighbors in coordinate grid', default=5, type=int)
parser.add_argument('--nuts', help='NUTS regions shape file',
                    default="/Users/dirkhovy/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp")
parser.add_argument('--nuts_level', help='NUTS level', type=int, default=2)
parser.add_argument('--N', help='minimum occurrence of words', type=int, default=10)
parser.add_argument('--prefix', help='output prefix', type=str, default='output')
parser.add_argument('--random', help='permute count matrix at random (for comparison reasons only)',
                    action='store_true', default=False)
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

rows = []
cols = []
values = []

word2int_count = count()
word2int = defaultdict(word2int_count.__next__)

if args.stopwords:
    stops = set(stopwords.words(args.stopwords))
else:
    stops = set()

info = [args.country, 'min-freq%s' % args.N, args.distance, args.target, 'min-support%s' % args.min_support]
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
else:
    info.append('NUTS-%s' % args.nuts_level)

regions = []
region_centers = {}

word_tokenizer = WordPunctTokenizer()
sentence_tokenizer = nltk.data.load('tokenizers/punkt/%s.pickle' % (country2lang[args.country]))
stemmer = SnowballStemmer(country2lang[args.country])

inverted_stems = defaultdict(set)
noun_propensity = defaultdict(int)

# determine target of the aggregation: gender, NUTS region, or geo-coordinate
if args.target == 'region':
    print("reading NUTS...", file=sys.stderr, flush=True)
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
    print("%s regions" % len(regions), file=sys.stderr, flush=True)

    print("computing adjacency...", file=sys.stderr, flush=True)
    adjacency = pd.DataFrame(False, index=regions, columns=regions)
    for i in range(len(shapes)):
        nuts_shape = shapes[i].buffer(0.1)
        for j in range(len(shapes)):
            if i == j:
                continue
            adjacency.iloc[i, j] = nuts_shape.intersects(shapes[j])
            adjacency.iloc[j, i] = shapes[j].intersects(nuts_shape)
            # TODO: what about the inverse?

    land_regions = list(regions)

# geo-coordinates are 0.1 increments of lat-lng pairs, based on the country's bounding box
elif args.target == 'coords':
    num_lats = country_lats.shape[0]
    num_lngs = country_lngs.shape[0]

    for lat in range(num_lats):
        for lng in range(num_lngs):
            regions.append((lat, lng))

    num_regions = len(regions)
    print("%s regions" % num_regions, file=sys.stderr, flush=True)

    m = Basemap(llcrnrlat=country_boxes[args.country][0],
                urcrnrlat=country_boxes[args.country][1],
                llcrnrlon=country_boxes[args.country][2],
                urcrnrlon=country_boxes[args.country][3],
                resolution='l', projection='merc')

    lon_bins_2d, lat_bins_2d = np.meshgrid(country_lngs, country_lats)

    country_lngs_m, country_lats_m = m(lon_bins_2d, lat_bins_2d)

    print("computing land mask...", file=sys.stderr, flush=True)
    land = np.reshape(np.array([m.is_land(country_lngs_m[n1, n2], country_lats_m[n1, n2]) for (n1, n2) in regions]), (num_lats, num_lngs))
    land_ravel = land.ravel()

    # get maps between the two views
    coord2id = dict(zip([(i,j) for i in range(num_lats) for j in range(num_lngs)], range(num_regions)))
    id2coord = {v:k for k,v in coord2id.items()}

    print("computing adjacency...", file=sys.stderr, flush=True)
    start = time.time()
    # fully connected
    adjacency = np.zeros((num_regions, num_regions))
    for i in range(num_lats):
        for j in range(num_lngs):
            coord = coord2id[(i,j)]
            for knn in [coord2id[kn] for kn in neighbors(i,j,land, 10)]:
                adjacency[coord, knn] = 1

    land_regions = land_ravel.nonzero()[0].tolist()
    print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)
    # adjacency.to_csv('%s-%s.adjacency.csv' % (args.country, args.coord_size))
    print('%s land regions (out of %s)' % (len(land_regions), len(regions)), file=sys.stderr, flush=True)

else:
    regions = ['F', 'M']

# land_region_name2int = dict([(name, i) for (i, name) in enumerate(land_regions)])
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
support = defaultdict(int)
start = time.time()

if args.trustpilot:
    print("\nProcessing Trustpilot", file=sys.stderr, flush=True)
    for line_no, line in enumerate(islice(open(args.trustpilot), None)):
        if line_no > 0:
            if line_no % 1000 == 0:
                print("%s" % (line_no), file=sys.stderr, flush=True)
            elif line_no % 100 == 0:
                print('.', file=sys.stderr, flush=True, end=' ')

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

            target = coord2id[target]

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

                        # use characters only
                        if args.chars:
                            text = "S" + text.lower() + "E"

                            for n in range(2, 7):
                                for chargram in [word2int[text[i:i+n]] for i in range(len(text)-n+1)]:
                                    rows.append(target)
                                    cols.append(chargram)
                                    values.append(1)

                        # use your words...
                        else:
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

                            wids = [word2int[word] for word in words]
                            for w, wid in enumerate(wids):
                                if org_words[w][0].isupper():
                                    noun_propensity[wid] += 1

                                rows.append(target)
                                cols.append(wid)
                                values.append(1)
                                # counts[word][target] += 1
                            review_frequency.update(set(wids))

                            if args.bigrams:
                                bigrams = [' '.join(bigram) for bigram in nltk.bigrams(words) if
                                           ' '.join(bigram).strip() is not '']
                                if args.stem:
                                    for bigram in bigrams:
                                        inverted_stems[bigram].add(bigram)

                                wids = [word2int[word] for word in bigrams]
                                for bigram in wids:
                                    rows.append(target)
                                    cols.append(wid)
                                    values.append(1)
                                    # counts[bigram][target] += 1
                                review_frequency.update(set(wids))


        except ValueError:
            continue
        except KeyError:
            continue

if args.twitter:
    print("\nProcessing Twitter", file=sys.stderr, flush=True)
    for line_no, line in enumerate(islice(open(args.twitter), None)):
        if line_no > 0:
            if line_no % 1000 == 0:
                print("%s" % (line_no), file=sys.stderr, flush=True)
            elif line_no % 100 == 0:
                print('.', file=sys.stderr, flush=True, end=' ')

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

            # if 'da' not in languages and 'no' not in languages:
            #     continue

            if args.target == 'region':
                user_regions = user['actor']['NUTS%s' % args.nuts_level]['region']

            elif args.target == 'coords':
                user_regions = []
                try:
                    # lat, lng
                    user_regions = [coord2id[tuple(user['geo']['coordinates'])]]
                except KeyError:
                    try:
                        coords = list(Point(Polygon(user['location']['geo']['coordinates'][0]).centroid).coords)[0]
                        user_regions = [(coords[1], coords[0])]
                    except KeyError:
                        pass
                    except NotImplementedError:
                        pass

                # TODO: fix problem here:
                # TypeError: 'int' object is not iterable in <listcomp>
                try:
                    user_regions = [coord2id[(bisect.bisect(country_lats, float('%.3f' % user_lat)),
                                                     bisect.bisect(country_lngs, float('%.3f' % user_lng)))] for
                                    (user_lat, user_lng) in
                                    user_regions]
                except TypeError:
                    continue
                    print("Can't translate regions:", user_regions, file=sys.stderr, flush=True)

            if body:
                # increase support for this region
                for target in user_regions:
                    support[target] += 1

                text = re.sub(numbers, '0', body)
                text = re.sub(r'\n', ' ', text)
                text = re.sub(names, '', text)
                text = re.sub(urls, '', text)

                if args.chars:
                    text = "S" + text.lower() + "E"
                    for n in range(2, 7):
                        for chargram in [word2int[text[i:i+n]] for i in range(len(text)-n+1)]:
                            rows.append(target)
                            cols.append(chargram)
                            values.append(1)

                # use your words
                else:
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

                    wids = [word2int[word] for word in words]
                    for w, wid in enumerate(wids):
                        if w > 0 and org_words[w][0].isupper() and not org_words[w].isupper():
                            noun_propensity[wid] += 1
                        for target in user_regions:
                            rows.append(target)
                            cols.append(wid)
                            values.append(1)
                    review_frequency.update(set(wids))

                    if args.bigrams:
                        bigrams = [' '.join(bigram) for bigram in nltk.bigrams(words) if ' '.join(bigram).strip() is not '']
                        if args.stem:
                            for bigram in bigrams:
                                inverted_stems[bigram].add(bigram)
                        bigrams = [word2int[word] for word in bigrams]
                        for bigram in bigrams:
                            for target in user_regions:
                                rows.append(target)
                                cols.append(wid)
                                values.append(1)
                        review_frequency.update(set(bigrams))


        except ValueError as ve:
            # print(ve, file=sys.stderr, flush=True)
            continue
        except KeyError as ke:
            # print(ke, file=sys.stderr, flush=True)
            continue

print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)

print("\ncreating sparse count matrix", file=sys.stderr, flush=True)
counts = sparse.coo_matrix((values, (rows, cols)), dtype=np.float, shape=(len(regions), len(word2int))).tocsr()
num_inst, num_feats = counts.shape
print("\n%s regions, %s features" % (num_inst, num_feats), file=sys.stderr, flush=True)

# ==========================================================
# split here:
# save counts and support
print("smallest observed support for regions: %s" % (min(support.values())), file=sys.stderr, flush=True)

# filter out nouns
# TODO: solve this for sparse matrix!!!
# if args.nounfilter:
#     non_nouns = set([w for w, c in counts.items() if noun_propensity[w] / sum(c.values()) < args.nounfilter])
#     counts = {word: counts[word] for word in non_nouns}

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
ignore_regions = {target for target in regions if support[coord2id[target]] < args.min_support}
print('found %s unsupported regions (fewer than %s entries), leaving %s supported land regions...' % (len(ignore_regions), args.min_support, len(regions) - len(ignore_regions)),
      file=sys.stderr, flush=True)
with open('%s%s.support.tsv' % (args.prefix, '.'.join(info)), 'w') as support_file:
    support_file.write('\n'.join(["%s\t%s" % (target, support[coord2id[target]]) for target in regions]))


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

Y = distros

if args.kernel:
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


# if args.stem:
#     print('Writing stem indices...', file=sys.stderr, flush=True)
#     stem_file = open('%s%s.inverted_stems.tsv' % (args.prefix, '.'.join(info)), 'w')
#     for stem, words in inverted_stems.items():
#         stem_file.write("%s\t%s\n" % (stem, ', '.join(words)))
#     stem_file.close()
#     print('done', file=sys.stderr, flush=True)

# 2 stats

# start = time.time()
# print('Computing region differences...', file=sys.stderr, flush=True)
# g2_file = open('%s%s.G2-regions.tsv' % (args.prefix, '.'.join(info)), 'w')
# for i, row in enumerate(all_distros):
#     rest_indices = row_indices.copy()
#     rest_indices.remove(i)
#     m = all_distros[rest_indices].mean(axis=0)
#     g2 = np.log(row / m)
#     top_vocab = g2.argsort()[-50:]
#
#     if args.stem:
#         g2_file.write("%s\n\n" % '\n'.join('%s\t%s\t%s' % (land_regions[i], w, g) for (w, g) in
#                                            reversed(list(zip([min(inverted_stems[top_N[x]]) for x in top_vocab],
#                                                              [g2[x] for x in top_vocab])))))
#
#     else:
#         g2_file.write("%s\n\n" % '\n'.join('%s\t%s\t%s' % (land_regions[i], w, g) for (w, g) in
#                                            reversed(
#                                                list(zip([top_N[x] for x in top_vocab], [g2[x] for x in top_vocab])))))
#
# g2_file.close()
# print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)


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
                        in_cluster.append(coord2id[r])
                    else:
                        rest_indices.append(coord2id[r])

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
            if args.stem:
                g2_file.write("%s\n\n" % '\n'.join('\t%s\t%s' % (w, g) for (w, g) in reversed(
                    list([(g2[i], min(inverted_stems[x])) for i, x in zip(top_50, top_vocab)]))))
            else:
                g2_file.write("%s\n\n" % '\n'.join('\t%s\t%s' % (w, g) for (w, g) in reversed(
                    list(zip([x for x in top_vocab], [g2[x] for x in top_50])))))

        g2_file.close()
print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)

# plot clusters on map
if args.show:
    plt.show()
