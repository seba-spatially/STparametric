import numpy as np

from functions import curve_fitter, quadratic


def bezierTest(t):
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
    return (control)


def main():
    t = h.iloc[10]
    t
    Dg = dd.ix[t['i']:t['ij'], ]
    Xg = Dg.x
    Yg = Dg.y
    Zg = Dg.local_time
    z1 = 1. / 3
    z2 = 2. / 3

    l1 = t['coef'][2][0] * z1 ** 2 + t['coef'][2][1] * z1 + t['coef'][2][2]
    l2 = t['coef'][2][0] * z2 ** 2 + t['coef'][2][1] * z2 + t['coef'][2][2]

    curve_fitter(Dg)

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

    tt = np.linspace(0, 1, 20)

    bez = []
    for t in tt:
        x = (1 - t) ** 3 * x_[0] + 3 * (1 - t) ** 2 * t * x_[1] + 3 * (1 - t) * t ** 2 * x_[2] + t ** 3 * x_[3]
        y = (1 - t) ** 3 * y_[0] + 3 * (1 - t) ** 2 * t * y_[1] + 3 * (1 - t) * t ** 2 * y_[2] + t ** 3 * y_[3]
        z = (1 - t) ** 3 * z_[0] + 3 * (1 - t) ** 2 * t * z_[1] + 3 * (1 - t) * t ** 2 * z_[2] + t ** 3 * z_[3]
        bez.append([x, y, z])
    bez = pd.DataFrame(bez, columns=['x', 'y', 'z'])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xg, Yg, Zg, c='r')
    ax.scatter(x_, y_, z_, c='b')
    ax.scatter(bez.x, bez.y, bez.z, c='g')

    ##############################
    data = dd.ix[16:21, ]
    X = data.x
    Y = data.y
    Z = data.local_time

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    xmi = X.min()
    xma = X.max()
    ymi = Y.min()
    yma = Y.max()
    zmi = Z.min()
    zma = Z.max()

    X = (X - xmi) / (xma - xmi)
    Y = (Y - ymi) / (yma - ymi)
    Z = (Z - zmi) / (zma - zmi)

    L = np.array([list(range(len(Z)))]).T
    # independent fits
    # av**2 + bv +c
    Ax = np.array([X ** 2, X, np.ones(len(X))]).T
    Ay = np.array([Y ** 2, Y, np.ones(len(Y))]).T
    Az = np.array([Z ** 2, Z, np.ones(len(Z))]).T

    # Z as a function of the index
    cz = np.matmul(linalg.pinv(Az), L)
    cx = np.matmul(linalg.pinv(Ax), L)
    cy = np.matmul(linalg.pinv(Ay), L)

    # predict L for Z = [1/3,2/3]
    z1 = 1 / 3
    z2 = 2 / 3

    l1 = cz[0] * z1 ** 2 + cz[1] * z1 + cz[2]
    l2 = cz[0] * z2 ** 2 + cz[1] * z2 + cz[2]

    # Find intersect between each parametric curve and L at points l1 and l2
    # Unsure why it is the smallest solution, but seems to work
    q = quadratic(cx[0], cx[1], cx[2] - l1)
    x1 = np.array([x for x in q if x > 0]).min()

    q = quadratic(cx[0], cx[1], cx[2] - l2)
    x2 = np.array([x for x in q if x > 0]).min()

    q = quadratic(cy[0], cy[1], cy[2] - l1)
    y1 = np.array([x for x in q if x > 0]).min()

    q = quadratic(cy[0], cy[1], cy[2] - l2)
    y2 = np.array([x for x in q if x > 0]).min()

    u = 1 / 3
    v = 2 / 3

    a = 3 * (1 - u) * (1 - u) * u
    b = 3 * (1 - u) * u * u
    c = 3 * (1 - v) * (1 - v) * v
    d = 3 * (1 - v) * v * v

    det = a * d - b * c

    p0x = X[0]
    p0y = Y[0]
    p0z = Z[0]
    p3x = X[-1]
    p3y = Y[-1]
    p3z = Z[-1]

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

    x_ = np.array([p0x, p1x, p2x, p3x]) * (xma - xmi) + xmi
    y_ = np.array([p0y, p1y, p2y, p3y]) * (yma - ymi) + ymi
    z_ = np.array([p0z, p1z, p2z, p3z]) * (zma - zmi) + zmi

    control = [[x_[0], y_[0], z_[0]], [x_[1], y_[1], z_[1]], [x_[2], y_[2], z_[2]], [x_[3], y_[3], z_[3]]]

    tt = np.linspace(0, 1, 10)

    bez = []
    for t in tt:
        x = (1 - t) ** 3 * p0x + 3 * (1 - t) ** 2 * t * p1x + 3 * (1 - t) * t ** 2 * p2x + t ** 3 * p3x
        y = (1 - t) ** 3 * p0y + 3 * (1 - t) ** 2 * t * p1y + 3 * (1 - t) * t ** 2 * p2y + t ** 3 * p3y
        z = (1 - t) ** 3 * p0z + 3 * (1 - t) ** 2 * t * p1z + 3 * (1 - t) * t ** 2 * p2z + t ** 3 * p3z
        bez.append([x, y, z])

    bez = pd.DataFrame(bez, columns=['x', 'y', 'z'])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='r')
    ax.scatter(bez.x, bez.y, bez.z, c='g')
    ax.scatter([p0x, p1x, p2x, p3x], [p0y, p1y, p2y, p3y], [p0z, p1z, p2z, p3z], c='b')

    ax.scatter(x_, y_, z_, c='b')


if __name__ == "__main__":
    main()
