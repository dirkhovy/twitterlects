import argparse
import bisect
import json
# import ujson as json
import re
import sys
import fiona
import nltk.data
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from shapely.geometry import shape, Point, Polygon
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


parser = argparse.ArgumentParser(description="collect data to compare regions")
parser.add_argument('--trustpilot', help='input file')
parser.add_argument('--twitter', help='input file')
parser.add_argument('--bigrams', help='use bigrams', action="store_true")
parser.add_argument('--coord_size', help='size of coordinate grid in degrees', default=0.1, type=float)
parser.add_argument('--country', choices=['denmark', 'germany', 'france', 'uk', 'eu'], help='which country to use',
                    default='eu')
parser.add_argument('--geo', help='use geographic distance', action='store_true', default=False)
parser.add_argument('--limit', help='max instances', type=int, default=None)
parser.add_argument('--nuts', help='NUTS regions shape file',
                    default="/Users/dirkhovy/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp")
parser.add_argument('--nuts_level', help='NUTS level', type=int, default=2)
parser.add_argument('--prefix', help='output prefix', type=str, default='output')
parser.add_argument('--stem', help='stem words', action='store_true', default=False)
parser.add_argument('--stopwords', help='stopwords', type=str)
parser.add_argument('--target', choices=['region', 'gender', 'coords'], help='target variable', default='region')

args = parser.parse_args()

country2lang = {'denmark': 'danish', 'germany': 'german', 'france': 'french', 'uk': 'english', 'eu':None}
country2nuts = {'denmark': 'DK', 'germany': 'DE', 'france': 'FR', 'uk': 'UK', 'eu':None}
country_lats = np.arange(country_boxes[args.country][0], country_boxes[args.country][1], args.coord_size)
country_lngs = np.arange(country_boxes[args.country][2], country_boxes[args.country][3], args.coord_size)

if args.stopwords:
    stops = set(stopwords.words(args.stopwords))
else:
    stops = set()

info = [args.country, args.target]
if args.trustpilot:
    info.append('Trustpilot')
if args.twitter:
    info.append('Twitter')
if args.bigrams:
    info.append('bigrams')
if args.stem:
    info.append('stemmed')
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

review_frequency = Counter()
counts = defaultdict(lambda: defaultdict(lambda: 1))
support = defaultdict(int)
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
    print("%s regions" % len(regions))
    adjacency = pd.DataFrame(False, index=regions, columns=regions)

    print("computing adjacency...", file=sys.stderr)
    for i in range(len(shapes)):
        nuts_shape = shapes[i].buffer(0.1)
        for j in range(len(shapes)):
            if i == j:
                continue
            adjacency.iloc[i, j] = nuts_shape.intersects(shapes[j])
            adjacency.iloc[j, i] = shapes[j].intersects(nuts_shape)
            # TODO: what about the inverse?


# geo-coordinates are 0.1 increments of lat-lng pairs, based on the country's bounding box
elif args.target == 'coords':
    num_lats = country_lats.shape[0]
    num_lngs = country_lngs.shape[0]

    for lat in range(num_lats):
        for lng in range(num_lngs):
            regions.append((lat, lng))
    print("%s regions" % len(regions))
    adjacency = pd.DataFrame(False, index=regions, columns=regions)

    print("computing adjacency...", file=sys.stderr)
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

else:
    regions = ['F', 'M']

region_name2int = dict([(name, i) for (i, name) in enumerate(regions)])

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
else:
    D = None

# collect total vocab
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
                for target in user_regions:
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
# save counts

if args.stem:
    stem_file_name = '%s.%s.inverted_stems.tsv' % (args.prefix, '.'.join(info))
    print('Writing stem indices...', file=sys.stderr)
    stem_file = open(stem_file_name, 'w')
    for stem, words in inverted_stems.items():
        stem_file.write("%s\t%s\n" % (stem, ', '.join(words)))
    stem_file.close()
    print('done', file=sys.stderr)
else:
    stem_file_name = None

model = {"counts": counts,
    "noun_propensity": noun_propensity,
    "review_frequency": review_frequency,
    "regions": regions,
    "support": support,
    "adjacency": adjacency.values.tolist(),
    "inverted_stems": stem_file_name,
    "info": info,
    "D": D,# probably obsolete, now that we have neighborhood adjacency
    "region_name2int": region_name2int
     }

print('Saving model to "%s.%s.json"' % (args.prefix, '.'.join(info)), file=sys.stderr)
with open("%s.%s.json" % (args.prefix, '.'.join(info)), "w") as jmodel:
    json.dump(model, jmodel)
print('done', file=sys.stderr)
