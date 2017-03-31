import logging
import sys, os
import warnings
from enum import Enum

import numpy as np
import pandas as pd


class Order(Enum):
    First = 1
    Second = 2


err = sys.stderr.write


def _read_sql():
    """

    :return: a function that executes a query and returns a dataframe.
    """
    import sqlalchemy as sa

    logging.basicConfig()
    logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

    engine = sa.create_engine("crate://db.world.io:4200")

    def reader(q):
        return pd.read_sql(q, engine)

    return reader


read_sql = _read_sql()


def curve_fitter(data, order=Order.Second):
    """

    :param data:
    :param order:
    :return:
    """
    from numpy import linalg
    import scipy.stats as stats

    if order not in Order:
        err("Invalid order")
        return None

    mi_x = data.x.min()
    mi_y = data.y.min()
    mi_z = data.local_time.min()
    ma_x = data.x.max()
    ma_y = data.y.max()
    ma_z = data.local_time.max()

    x = np.array((data.x - mi_x) / (ma_x - mi_x))
    x[np.isnan(x)] = 0.
    y = np.array((data.y - mi_y) / (ma_y - mi_y))
    y[np.isnan(y)] = 0.
    z = np.array((data.local_time - mi_z) / (ma_z - mi_z))
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


D = (np.pi / 180) * 6378000


def fit_first(data, e=100):
    """

    :param data:
    :param e:
    :return:
    """
    warnings.simplefilter('ignore', np.RankWarning)

    px = np.polyfit(data.local_time, data.x, deg=1)
    py = np.polyfit(data.local_time, data.y, deg=1)

    xhat = np.array([px[0] * v + px[1] for v in data.local_time])
    yhat = np.array([py[0] * v + py[1] for v in data.local_time])

    max_ex = np.max(abs(xhat - data.x)) * D
    max_ey = np.sum(abs(yhat - data.y)) * D

    out = {
        'pts': {
            'from': [
                px[0] * data.local_time.min() + px[1],
                py[0] * data.local_time.min() + py[1],
                data.local_time.min(),
            ],
            'to': [
                px[0] * data.local_time.max() + px[1],
                py[0] * data.local_time.max() + py[1],
                data.local_time.max(),
            ],
        },
        'p': [max_ex, max_ey],
    }
    return out


def coord_parser(p):
    """expects p = '[lat, lon]' then returns [lat, lon]"""
    return eval(p)


def st_prep(data, i=0, j=4):
    """

    :param data:
    :param i:
    :param j:
    :return:
    """
    out = []
    while (i + j) <= data.shape[0]:
        p = True
        while p and (i + j < data.shape[0]):
            d0 = data.ix[i:(i + j), ].copy()
            o = curve_fitter(d0, order=Order.First)
            p = np.all([x > 0.8 for x in o['pVals']])
            j += 1
        else:
            ij = i + j - 2
            if j == 5:
                ij += 1

            d0 = data.ix[i:ij, ].copy()

            flp = [d0.iloc[0]['x'], d0.iloc[0]['y'], d0.iloc[0]['local_time'],
                   d0.iloc[-1]['x'], d0.iloc[-1]['y'], d0.iloc[-1]['local_time']]

            o = curve_fitter(d0, order=Order.First)
            p = np.all([x > 0.8 for x in o['pVals']])
            if p:
                oo = {
                    'coef': o['coefficients'], 'pval': o['pVals'],
                    'boundary': o['boundaries'], 'i': i, 'ij': ij, 'flp': flp,
                }
                out.append(oo)

                i = ij
                j = 4
            else:
                i += 1
                j = 4
    helmets = pd.DataFrame(out)
    return helmets


def quadratic(a, b, c):
    """

    :param a:
    :param b:
    :param c:
    :return:
    """
    d = (b ** 2) - (4 * a * c)
    # find two solutions
    sol1 = (-b - np.sqrt(d)) / (2 * a)
    sol2 = (-b + np.sqrt(d)) / (2 * a)
    out = np.array((sol1, sol2))
    return out


def bezier_solver(t):
    """

    :param t:
    :return:
    """
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

    pf = lambda e, f, g, h: (t['flp'][e] - t['boundary'][f]) / (t['boundary'][g] - t['boundary'][h])

    p0x = pf(0, 0, 1, 0)
    p0y = pf(1, 2, 3, 2)
    p0z = pf(2, 4, 5, 4)
    p3x = pf(3, 0, 1, 0)
    p3y = pf(4, 2, 3, 2)
    p3z = pf(5, 4, 5, 4)

    qf = lambda e, f, g, h: e - ((1 - f) * (1 - f) * (1 - f) * g + f * f * f * h)

    q1x = qf(x1, u, p0x, p3x)
    q1y = qf(y1, u, p0y, p3y)
    q1z = qf(z1, u, p0z, p3z)

    q2x = qf(x2, v, p0x, p3x)
    q2y = qf(y2, v, p0y, p3y)
    q2z = qf(z2, v, p0z, p3z)

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

    lf = lambda e, f, g: (t['boundary'][e] - t['boundary'][f]) + t['boundary'][g]

    x_ = np.array([p0x, p1x, p2x, p3x]) * lf(1, 0, 0)
    y_ = np.array([p0y, p1y, p2y, p3y]) * lf(3, 2, 2)
    z_ = np.array([p0z, p1z, p2z, p3z]) * lf(5, 4, 4)

    control = [[x_[i], y_[i], z_[i]] for i in range(0, 4)]
    return control


def st_prep_first(data, i=0, j0=1):
    """

    :param data:
    :param i:
    :param j0:
    :return:
    """
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
            ij = i + j - 2
            if j == 2:
                ij += 1

            d0 = data.ix[i:ij, ].copy()
            print(i)

            o = fit_first(d0)
            p = np.all([i < 100 for i in o['p']])
            if p:
                oo = {'model': o, 'i': i, 'ij': ij}
                out.append(oo)

                i = ij
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


def query_metro_ingestid(msa='boston'):
    """

    :param msa:
    :return:
    """
    q = f"""select datasetid, metadata['table'] as table_name, ingestid, major, minor
              from dataset
             where datasetid like 'tracking/%/{msa}'
             order by datasetid, major desc, minor desc"""
    df = read_sql(q)
    ingestid = df.loc[df.major == df.major.max()].loc[df.minor == df.minor.max()]['ingestid'][0]
    return ingestid


def query_deviceids(ingestid, acc):
    """

    :param ingestid:
    :return:
    """
    q = f"""select count(*) as count
              from cuebiq
             where ingestid = '{ingestid}'"""
    z = read_sql(q)

    limit = os.environ.get('LIMIT', z['count'][0])

    q = f"""select data['t_deviceid'] as deviceid, count(*) as count
              from cuebiq
             where ingestid = '{ingestid}'
               and data['i_accuracy'] < {acc}
             group by data['t_deviceid']
            having count(*) > 500
             limit {limit}"""
    df = read_sql(q)

    # t = tuple(df["data['t_deviceid']"])
    # out = {'IngID': IngID, 'deviceID': t}
    return df


def query_phone_data(ingestid, deviceid, count, accuracy):
    """

    :param ingestid:
    :param deviceid:
    :param count:
    :param accuracy:
    :return:
    """
    q = f"""select data['i_devicetime']+data['i_tzoffset'] as local_time, data['i_accuracy'] as accuracy, point
              from cuebiq
             where ingestid = '{ingestid}'
               and data['t_deviceid'] = '{deviceid}'
               and data['i_accuracy'] < {accuracy}
             limit {count}"""

    df = read_sql(q)

    df.columns = ['local_time', 'accuracy', 'point']
    df['key'] = df['local_time'].apply(str) + df.point.apply(str)
    df = df.drop_duplicates(subset=['key'])
    del df['key']
    p = df['point'].str
    df['x'] = p[0]
    df['y'] = p[1]
    del df['point']
    df = df.sort_values(by='local_time')
    df = df.drop_duplicates(subset='local_time')
    df = df.reset_index(drop=True)
    return df
