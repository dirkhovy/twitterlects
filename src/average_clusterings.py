"""
produce an average of two clustering methods (say, Ward and Average) with the same number of clusters

"""

import argparse
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


def get_dict(file):
    d = defaultdict(list)
    all_regions = []
    for line in open(file):
        region, cluster = line.strip().split('\t')
        d[cluster].append(region)
        all_regions.append(region)

    all_regions.sort()
    combinations = pd.DataFrame(0, columns=all_regions, index=all_regions)
    for cluster, regions in d.items():
        for i, region1 in enumerate(regions):
            for region2 in regions[i:]:
                combinations[region1][region2] = 1
                combinations[region2][region1] = 1

    return d, combinations
# d1, c1 = get_dict(sys.argv[1])
# d2, c2 = get_dict(sys.argv[2])
#
# combi = c1.values + c2.values
# paired = pd.DataFrame(data=np.where(c1.values.any()>-1 and c2.values.any()>-1, c1.values+c2.values, -1), columns=c1.index, index=c1.index)
# paired += 1
#
# uniq_values = pd.Series(paired.values.ravel()).unique().tolist()
#
# print('\n'.join(["%s\t%s" % (region, cluster) for region, cluster in zip(c1.index, paired[c1.index[0]].tolist())]))

def read(file):
    d = {}
    for line in open(file):
        region, cluster = line.strip().split('\t')
        d[region] = cluster

    return d

d1 = read(sys.argv[1])
d2 = read(sys.argv[2])

regions = list(d1.keys())
regions.sort()

values = ["%s_%s" % (d1[region], d2[region]) for region in regions]
frequency = Counter(values)
print(frequency, file=sys.stderr)
mapping = dict([(value, i+1) for i, (value, _) in enumerate(frequency.most_common(int(sys.argv[3])-1))])
for region, value in zip(regions, values):
    print('%s\t%s' % (region, mapping.get(value, 0)))

