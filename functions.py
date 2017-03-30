import logging
import sys
from enum import Enum

import numpy as np
import pandas as pd
import sqlalchemy as sa

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)


class Order(Enum):
    First = 1
    Second = 2


err = sys.stderr.write


def curve_fitter(data, order=Order.Second):
    from numpy import linalg
    import scipy.stats as stats

    if order not in Order:
        err("Invalid order")
        return None

    mi_x = data.x.min()
    mi_y = data.y.min()
    mi_z = data.deviceTime.min()
    ma_x = data.x.max()
    ma_y = data.y.max()
    ma_z = data.deviceTime.max()

    x = np.array((data.x - mi_x) / (ma_x - mi_x))
    x[np.isnan(x)] = 0.
    y = np.array((data.y - mi_y) / (ma_y - mi_y))
    y[np.isnan(y)] = 0.
    z = np.array((data.deviceTime - mi_z) / (ma_z - mi_z))
    l = np.array([float(x) for x in range(len(z))]).T

    clx = cly = clz = p_val_x = p_val_y = p_val_z = None

    if order == Order.First:
        ay = np.array([y, np.ones(len(y))]).T
        cly = np.dot(linalg.pinv(ay), l)
        ax = np.array([x, np.ones(len(x))]).T
        clx = np.dot(linalg.pinv(ax), l)
        az = np.array([z, np.ones(len(z))]).T
        clz = np.dot(linalg.pinv(az), l)

        df = len(y) - len(cly)
        xhat = np.array([clx[0] * v + clx[1] for v in x])
        yhat = np.array([cly[0] * v + cly[1] for v in y])
        zhat = np.array([clz[0] * v + clz[1] for v in z])

        ssex = np.sum((xhat - l) ** 2)
        ssey = np.sum((yhat - l) ** 2)
        ssez = np.sum((zhat - l) ** 2)

        p_val_y = 1 - stats.chi2.cdf(x=ssey, df=df)
        p_val_x = 1 - stats.chi2.cdf(x=ssex, df=df)
        p_val_z = 1 - stats.chi2.cdf(x=ssez, df=df)

    if order == Order.Second:
        ay = np.array([y ** 2, y, np.ones(len(y))]).T
        cly = np.dot(linalg.pinv(ay), l)
        ax = np.array([x ** 2, x, np.ones(len(x))]).T
        clx = np.dot(linalg.pinv(ax), l)
        az = np.array([z ** 2, z, np.ones(len(z))]).T
        clz = np.dot(linalg.pinv(az), l)

        df = len(y) - len(cly)
        xhat = np.array([clx[0] * v ** 2 + clx[1] * v + clx[2] for v in x])
        yhat = np.array([cly[0] * v ** 2 + cly[1] * v + cly[2] for v in y])
        zhat = np.array([clz[0] * v ** 2 + clz[1] * v + clz[2] for v in z])

        ssex = np.sum((xhat - l) ** 2)
        ssey = np.sum((yhat - l) ** 2)
        ssez = np.sum((zhat - l) ** 2)

        p_val_y = 1 - stats.chi2.cdf(x=ssey, df=df)
        p_val_x = 1 - stats.chi2.cdf(x=ssex, df=df)
        p_val_z = 1 - stats.chi2.cdf(x=ssez, df=df)

    out = {
        'coefficients': [clx, cly, clz],
        'pVals': [p_val_x, p_val_y, p_val_z],
        'boundaries': [mi_x, ma_x, mi_y, ma_y, mi_z, ma_z],
    }
    return out


def fit_first(data, e=100):
    px = np.polyfit(data.deviceTime, data.x, deg=1)
    py = np.polyfit(data.deviceTime, data.y, deg=1)
    xhat = np.array([px[0] * v + px[1] for v in data.deviceTime])
    yhat = np.array([py[0] * v + py[1] for v in data.deviceTime])

    maxEx = np.max(abs(xhat - data.x)) * (np.pi / 180) * 6378000
    maxEy = np.sum(abs(yhat - data.y)) * (np.pi / 180) * 6378000
    pts = {'from': [px[0] * data.deviceTime.min() + px[1],
                    py[0] * data.deviceTime.min() + py[1],
                    data.deviceTime.min()],
           'to': [px[0] * data.deviceTime.max() + px[1],
                  py[0] * data.deviceTime.max() + py[1],
                  data.deviceTime.max()]}
    out = {'pts': pts, 'p': [maxEx, maxEy]}
    return out


def coord_parser(x):
    import re
    co = re.sub(r"[\[]", "", x)
    co = re.sub(r"[\]]", "", co)
    co = re.sub(r" ", "", co)
    co = co.split(',')
    return co


def st_prep(data, i=0, j=4):
    out = []
    i = 0
    j = 4
    while (i + j) <= data.shape[0]:
        p = True
        while p and (i + j < data.shape[0]):
            d0 = data.ix[i:(i + j), ].copy()
            o = curve_fitter(d0, order=Order.First)
            p = np.all([x > 0.8 for x in o['pVals']])
            j += 1
        else:
            if j == 5:
                d0 = data.ix[i:(i + j - 1), ].copy()
            else:
                d0 = data.ix[i:(i + j - 2), ].copy()

            flp = [d0.iloc[0]['x'], d0.iloc[0]['y'], d0.iloc[0]['deviceTime'],
                   d0.iloc[-1]['x'], d0.iloc[-1]['y'], d0.iloc[-1]['deviceTime']]
            o = curve_fitter(d0, order=Order.First)
            p = np.all([x > 0.8 for x in o['pVals']])
            if p:
                if j == 5:
                    oo = {'coef': o['coefficients'], 'pval': o['pVals'],
                          'boundary': o['boundaries'], 'i': i, 'ij': i + j - 1, 'flp': flp}
                    i += j - 1
                else:
                    oo = {'coef': o['coefficients'], 'pval': o['pVals'],
                          'boundary': o['boundaries'], 'i': i, 'ij': i + j - 2, 'flp': flp}
                    i += j - 2
                out.append(oo)

                j = 4
            else:
                i += 1
                j = 4
    helmets = pd.DataFrame(out)
    return helmets


def quadratic(a, b, c):
    d = (b ** 2) - (4 * a * c)
    # find two solutions
    sol1 = (-b - np.sqrt(d)) / (2 * a)
    sol2 = (-b + np.sqrt(d)) / (2 * a)
    out = np.array((sol1, sol2))
    return out


def bezier_solver(t):
    z1 = 1. / 3
    z2 = 2. / 3

    l1 = t['coef'][2][0] * z1 ** 2 + t['coef'][2][1] * z1 + t['coef'][2][2]
    l2 = t['coef'][2][0] * z2 ** 2 + t['coef'][2][1] * z2 + t['coef'][2][2]

    # Find intersect between each parametric curve and L at points l1 and l2
    # Unsure why it is the smallest solution, but seems to work
    q = quadratic(t['coef'][0][0], t['coef'][0][1], t['coef'][0][2] - l1)
    x1 = np.abs(q).min()

    q = quadratic(t['coef'][0][0], t['coef'][0][1], t['coef'][0][2] - l2)
    x2 = np.abs(q).min()

    q = quadratic(t['coef'][1][0], t['coef'][1][1], t['coef'][1][2] - l1)
    y1 = np.abs(q).min()

    q = quadratic(t['coef'][1][0], t['coef'][1][1], t['coef'][1][2] - l2)
    y2 = np.abs(q).min()

    u = 1. / 3
    v = 2. / 3

    a = 3 * (1 - u) * (1 - u) * u
    b = 3 * (1 - u) * u * u
    c = 3 * (1 - v) * (1 - v) * v
    d = 3 * (1 - v) * v * v

    det = a * d - b * c

    p0x = (t['flp'][0] - t['boundary'][0]) / (t['boundary'][1] - t['boundary'][0])
    p0y = (t['flp'][1] - t['boundary'][2]) / (t['boundary'][3] - t['boundary'][2])
    p0z = (t['flp'][2] - t['boundary'][4]) / (t['boundary'][5] - t['boundary'][4])
    p3x = (t['flp'][3] - t['boundary'][0]) / (t['boundary'][1] - t['boundary'][0])
    p3y = (t['flp'][4] - t['boundary'][2]) / (t['boundary'][3] - t['boundary'][2])
    p3z = (t['flp'][5] - t['boundary'][4]) / (t['boundary'][5] - t['boundary'][4])

    q1x = x1 - ((1 - u) * (1 - u) * (1 - u) * p0x + u * u * u * p3x)
    q1y = y1 - ((1 - u) * (1 - u) * (1 - u) * p0y + u * u * u * p3y)
    q1z = z1 - ((1 - u) * (1 - u) * (1 - u) * p0z + u * u * u * p3z)

    q2x = x2 - ((1 - v) * (1 - v) * (1 - v) * p0x + v * v * v * p3x)
    q2y = y2 - ((1 - v) * (1 - v) * (1 - v) * p0y + v * v * v * p3y)
    q2z = z2 - ((1 - v) * (1 - v) * (1 - v) * p0z + v * v * v * p3z)

    p1x = d * q1x - b * q2x
    p1y = d * q1y - b * q2y
    p1z = d * q1z - b * q2z
    p1x /= det
    p1y /= det
    p1z /= det

    p2x = (-c) * q1x + a * q2x
    p2y = (-c) * q1y + a * q2y
    p2z = (-c) * q1z + a * q2z
    p2x /= det
    p2y /= det
    p2z /= det

    x_ = np.array([p0x, p1x, p2x, p3x]) * (t['boundary'][1] - t['boundary'][0]) + t['boundary'][0]
    y_ = np.array([p0y, p1y, p2y, p3y]) * (t['boundary'][3] - t['boundary'][2]) + t['boundary'][2]
    z_ = np.array([p0z, p1z, p2z, p3z]) * (t['boundary'][5] - t['boundary'][4]) + t['boundary'][4]

    control = [[x_[0], y_[0], z_[0]], [x_[1], y_[1], z_[1]], [x_[2], y_[2], z_[2]], [x_[3], y_[3], z_[3]]]
    return control


def st_prep_first(data, i=0, j0=1):
    out = []
    j = j0
    while (i + j) <= data.shape[0]:
        j = j0
        p = True
        while p and (i + j < data.shape[0]):
            d0 = data.ix[i:(i + j), ].copy()
            o = fit_first(d0)
            p = np.all([i < 100 for i in o['p']])
            j += 1
        else:
            if j == 2:
                d0 = data.ix[i:(i + j - 1), ].copy()
            else:
                d0 = data.ix[i:(i + j - 2), ].copy()

            o = fit_first(d0)
            p = np.all([i < 100 for i in o['p']])
            if p:
                if j == 2:
                    oo = {'model': o, 'i': i, 'ij': i + j - 1}
                    i += j - 1
                else:
                    oo = {'model': o, 'i': i, 'ij': i + j - 2}
                    i += j - 2
                out.append(oo)

                j = j0
            else:
                i += 1
                j = j0

            if (i + j) >= data.shape[0]:
                break
    h = pd.DataFrame(out)
    from shapely.geometry import LineString
    g = [(i['pts']['from'], i['pts']['to']) for i in h.model]
    geometry = [LineString(i) for i in g]
    l = [i.wkt for i in geometry]
    h['geometry'] = l
    return h


def get_metro_ingestid(msa='boston'):
    engine = sa.create_engine("crate://db.world.io:4200")
    q = f"""select datasetid, metadata['table'] as table_name, ingestid, major, minor
              from dataset
             where datasetid like 'tracking/%/{msa}'
             order by datasetid, major desc, minor desc"""
    df = pd.read_sql(q, engine)
    ingestid = df.loc[df.major == df.major.max()].loc[df.minor == df.minor.max()]['ingestid'][0]
    return ingestid


def query_device_ids(ingestid):
    engine = sa.create_engine("crate://world.spatially.co:4200")
    q = f"""select count(*) as count
              from cuebiq
             where ingestid = '{ingestid}'"""
    z = pd.read_sql(q, engine)

    q = f"""select data['t_deviceid'] as deviceid, count(*) as count
              from cuebiq
             where ingestid = '{ingestid}'
             group by data['t_deviceid']
            having count(*) > 500
             limit {z['count'][0]}"""
    df = pd.read_sql(q, engine)

    # t = tuple(df["data['t_deviceid']"])
    # out = {'IngID': IngID, 'deviceID': t}
    return df


def get_phone_data(ingestid, device_id, co, acc):
    engine = sa.create_engine("crate://db.world.io:4200")

    q = f"""select data['i_devicetime']+data['i_tzoffset'] as local_time, data['i_accuracy'] as accuracy, point
              from cuebiq
             where ingestid = '{ingestid}'
               and data['t_deviceid'] = '{device_id}'
               and data['i_accuracy'] < {acc}
             limit {co}"""

    df = pd.read_sql(q, engine)

    df.columns = ['deviceTime', 'accuracy', 'point']
    df['key'] = df['deviceTime'].apply(str) + df.point.apply(str)
    df = df.drop_duplicates(subset=['key'])
    del df['key']
    p = df['point'].str
    df['x'] = p[0]
    df['y'] = p[1]
    del df['point']
    df = df.sort_values(by='deviceTime')
    return df
