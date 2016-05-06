from math import radians, cos
import numpy as np

EARTH_RADIUS = 6371

# latitudes are the bigger numbers, going N-S, longitude are the smaller numbers, going W-E
country_boxes = {
    'denmark': [54.5, 58., 7.5, 15.5],
    'germany': [47., 55.5, 5.5, 15.5],
    'france': [41., 51.5, -5.5, 10.],
    'usa_all': [18., 72., -180., -67.5],
    'usa': [24., 50., -125., -65.5],
    'world': [-60., 75., -179., 179.],
    'eu': [36., 70., -11., 40.],
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

def in_gb(lat, lng):
    """
    check whether a point is in the UK or not
    NB: does only check for border with France, not North and West (e.g., Iceland)
    :param lat:
    :param lng:
    :return:
    """
    if lng > 2:
        return False
    else:
        if lat > 51:
            return True
        else:
            if lng < 1 and lat >= 50:
                return True
            else:
                return False
