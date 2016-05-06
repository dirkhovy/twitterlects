import json
import pandas as pd
import sys
import numpy as np
import argparse
import time
from geo_info import *
from mpl_toolkits.basemap import Basemap

parser = argparse.ArgumentParser(description="collect data to compare regions")
parser.add_argument('--coord_size', help='size of coordinate grid in degrees', default=0.1, type=float)
parser.add_argument('--country', choices=['denmark', 'germany', 'france', 'uk', 'eu'], help='which country to use',
                    default='eu')
parser.add_argument('--num_neighbors', help='number of neighbors in coordinate grid', default=5, type=int)
parser.add_argument('--prefix', help='output prefix', type=str, default='output')

args = parser.parse_args()
country_lats = np.arange(country_boxes[args.country][0], country_boxes[args.country][1], args.coord_size)
country_lngs = np.arange(country_boxes[args.country][2], country_boxes[args.country][3], args.coord_size)

num_lats = country_lats.shape[0]
num_lngs = country_lngs.shape[0]
regions = []

for lat in range(num_lats):
    for lng in range(num_lngs):
        regions.append((lat, lng))

regions_file = '%s%s-%s.regions.json' % (args.prefix, args.country, args.coord_size)
print('Saving regions to "%s"' % (regions_file), file=sys.stderr, flush=True)
with open(regions_file, "w") as regions_json:
    json.dump(regions, regions_json)
print('done. Found %s regions' % len(regions), file=sys.stderr, flush=True)

m = Basemap(llcrnrlat=country_boxes[args.country][0],
            urcrnrlat=country_boxes[args.country][1],
            llcrnrlon=country_boxes[args.country][2],
            urcrnrlon=country_boxes[args.country][3],
            resolution='l', projection='merc')

lon_bins_2d, lat_bins_2d = np.meshgrid(country_lngs, country_lats)
country_lngs_m, country_lats_m = m(lon_bins_2d, lat_bins_2d)

print("computing adjacency...", file=sys.stderr, flush=True)
adjacency = pd.DataFrame(False, index=regions, columns=regions)
regions_set = set(regions)
land_regions = set()
covered = set()
num_neighbors = args.num_neighbors
start = time.time()
k = 1
for i in range(num_lats):
    for j in range(num_lngs):
        k += 1
        if k % 100 == 0:
            print("%s" % (k), file=sys.stderr, flush=True)
        elif k % 10 == 0:
            print('.', file=sys.stderr, end=' ', flush=True)

        neighbors = [(i + lat_iterator, j + lng_iterator) for lat_iterator in
                     list(range(-num_neighbors, num_neighbors + 1)) for lng_iterator in
                     list(range(-num_neighbors, num_neighbors + 1))]
        for (n1, n2) in neighbors:
            if tuple(sorted(((i, j), (n1, n2)))) in covered:
                continue
            if (n1, n2) in regions_set and \
                            (n1, n2) != (i, j) and \
                            n1 >= 0 and n2 >= 0 and \
                            n1 < num_lats and n2 < num_lngs and \
                    m.is_land(country_lngs_m[n1, n2], country_lats_m[n1, n2]):
                adjacency.ix[(i, j), (n1, n2)] = True
                adjacency.ix[(n1, n2), (i, j)] = True
                land_regions.add((n1, n2))
                covered.add(tuple(sorted(((i, j), (n1, n2)))))

land_regions_file = '%s%s-%s.land_regions.json' % (args.prefix, args.country, args.coord_size)
print('Saving regions to "%s"' % (land_regions_file), file=sys.stderr, flush=True)
with open(land_regions_file, "w") as land_regions_json:
    json.dump(land_regions, land_regions_json)
print('done.', file=sys.stderr, flush=True)

print('done in %.2f sec' % (time.time() - start), file=sys.stderr, flush=True)
adjacency.to_csv('%s%s-%s.adjacency.csv' % (args.prefix, args.country, args.coord_size))
print('%s land regions (out of %s)' % (len(land_regions), len(regions)), file=sys.stderr, flush=True)
