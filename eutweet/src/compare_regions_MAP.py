import argparse
import bisect
import gzip
import json
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import time
from shapely.geometry import Point, Polygon
from collections import defaultdict, Counter
from itertools import islice, count
from geo_info import *
from scipy import sparse, io

sns.set(font="monospace")
sns.set_context('poster')

numbers = re.compile(r"[0123456789]")
names = re.compile(r"@[^ ]*")
urls = re.compile(r"http[^ ]*")


parser = argparse.ArgumentParser(description="collect data to compare regions")
parser.add_argument('--twitter', help='input file')
parser.add_argument('--coord_size', help='size of coordinate grid in degrees', default=0.1, type=float)
parser.add_argument('--country', choices=['denmark', 'germany', 'france', 'uk', 'eu'], help='which country to use',
                    default='eu')
parser.add_argument('--geo', help='use geographic distance', action='store_true', default=False)
parser.add_argument('--limit', help='max instances', type=int, default=None)
parser.add_argument('--num_neighbors', help='number of neighbors in coordinate grid', default=5, type=int)
parser.add_argument('--nuts', help='NUTS regions shape file',
                    default="/Users/dirkhovy/working/lowlands/GeoStats/data/nuts/NUTS_RG_03M_2010.shp")
parser.add_argument('--prefix', help='output prefix', type=str, default='output')
parser.add_argument('--target', choices=['region', 'gender', 'coords'], help='target variable', default='coords')

args = parser.parse_args()


country_lats = np.arange(country_boxes[args.country][0], country_boxes[args.country][1], args.coord_size)
country_lngs = np.arange(country_boxes[args.country][2], country_boxes[args.country][3], args.coord_size)

rows = []
cols = []
values = []
support = defaultdict(int)

word2int_count = count()
word2int = defaultdict(word2int_count.__next__)


info = [args.country, args.target]
if args.geo:
    info.append('geo-distance')
if args.target == 'coords':
    info.append('%s-neighbors.size-%s' % (args.num_neighbors, args.coord_size))

regions_file = '%s-%s.regions.json' % (args.country, args.coord_size)
try:
    with open(regions_file) as region_input:
        regions = list(map(tuple, json.load(region_input)))
except OSError:
    print("ERROR: regions file %s not found! Please pre-compute" % ('%s-%s.regions.csv' % (args.country, args.coord_size)), file=sys.stderr, flush=True)
    sys.exit()

print("%s regions" % len(regions), file=sys.stderr, flush=True)

region_name2int = dict([(name, i) for (i, name) in enumerate(regions)])
review_frequency = Counter()

# compute matrix D, distances between regions
if args.geo:
    D = pd.DataFrame(0.0, index=regions, columns=regions)  # np.zeros((len(regions), len(regions)), dtype=float)

    for i in enumerate(regions):
        lat_lng1 = (country_lats[regions[i][0]], country_lngs[regions[i][1]])
        for j in enumerate(regions[i + 1:]):
            lat_lng2 = (country_lats[regions[j][0]], country_lngs[regions[j][1]])
            x = get_shortest_in(lat_lng1, lat_lng2)[0]
            D.ix[lat_lng1, lat_lng2] = x
            D.ix[lat_lng2, lat_lng1] = x

# collect total vocab
start = time.time()

# collect total vocab
print("\nProcessing Twitter", file=sys.stderr)
with gzip.open(args.twitter, "rb") as f:
    # for line_no, line in enumerate(islice(open(args.twitter), None)):
    for line_no, line in enumerate(islice(f, None)):
        if line_no > 0:
            if line_no % 100 == 0:
                print("%s" % (line_no), file=sys.stderr, flush=True)
            elif line_no % 10 == 0:
                print('.', file=sys.stderr, end=' ', flush=True)

        if args.limit and line_no == args.limit:
            break

        try:
            user = json.loads(line.decode("utf-8"))
            # user = json.loads(line)
            body = user.get('text', None)

            # exclude empty tweets
            if body is None:
                continue

            ############################
            # retrieve relevant fields #
            ############################
            languages = []
            try:
                twitter_lang = user['lang']
                languages.append(twitter_lang)
            except KeyError:
                pass
            try:
                lang_id = user['langid'][0]
                languages.append(lang_id)
            except KeyError:
                pass


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

            # exclude English Tweets from outside the UK
            try:
                for (ulat, ulng) in user_regions:
                    if 'en' in languages and not in_gb(ulat, ulng):
                        continue
            except TypeError:
                print("Problem with location %s" % user_regions, file=sys.stderr, flush=True)

            try:
                user_regions = [region_name2int[(bisect.bisect(country_lats, float('%.3f' % user_lat)),
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

                text = "S" + text.lower() + "E"
                for n in range(2, 7):
                    for chargram in [word2int[text[i:i+n]] for i in range(len(text)-n+1)]:
                        rows.append(target)
                        cols.append(chargram)
                        values.append(1)


        except ValueError as ve:
            # print(ve, file=sys.stderr)
            continue
        except KeyError as ke:
            # print(ke, file=sys.stderr)
            continue

print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)

#==========================================================
# save counts
print("\ncreating sparse count matrix", file=sys.stderr, flush=True)
counts = sparse.coo_matrix((values, (rows, cols)), dtype=np.float, shape=(len(regions), len(word2int))).tocsr()
num_inst, num_feats = counts.shape
print("\n%s regions, %s features" % (num_inst, num_feats), file=sys.stderr, flush=True)

io.mmwrite('%s.%s.counts' % (args.prefix, '.'.join(info)), counts)

int2word = {i:w for w, i in word2int.items()}
words = [int2word[i] for i in range(len(int2word))]

model = {
    "support": support,
    "info": info,
    "words": words
     }

print('Saving model to "%s.%s.json"' % (args.prefix, '.'.join(info)), file=sys.stderr)
with open("%s.%s.json" % (args.prefix, '.'.join(info)), "w") as jmodel:
    json.dump(model, jmodel)
print('done', file=sys.stderr)
