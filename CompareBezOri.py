##Connection to PG

def compare(pid = 0):
    import psycopg2
    conn = psycopg2.connect(host = 'localhost', port = '5434' ,
                            dbname="SpatioTemporal", user="postgres", password="geomatics")
    cur = conn.cursor()
    q = "SELECT st_astext(st_geometryfromtext), i, ij from public.beziergeometry;"
    cur.execute(q)
    records = cur.fetchall()

    import pandas as pd

    df = pd.DataFrame(records)
    df.columns = ['geometry', 'i', 'ij']
    df.head()

    #get original points
    ori = pd.read_csv('oneDeviceOneday.tsv', sep='\t')
    ori.head()

    x0 = ori.x[int(df.i[pid]):int(df.ij[pid])]
    y0 = ori.y[int(df.i[pid]):int(df.ij[pid])]

    from shapely.wkt import dumps, loads
    from shapely.geometry import LineString
    g = loads(df.geometry[pid])

    co = list(g.coords)
    x = [i[0] for i in co]
    y = [i[1] for i in co]
    z = [i[2] for i in co]

    import numpy as np
    tt = np.linspace(0,1,100)

    bez = []
    for t in tt:
        x_ = (1-t)**3 * x[0] + 3*(1-t)**2*t*x[1] + 3*(1-t)*t**2*x[2] + t**3*x[3]
        y_ = (1-t)**3 * y[0] + 3*(1-t)**2*t*y[1] + 3*(1-t)*t**2*y[2] + t**3*y[3]
        z_ = (1-t)**3 * z[0] + 3*(1-t)**2*t*z[1] + 3*(1-t)*t**2*z[2] + t**3*z[3]
        bez.append([x_,y_,z_])
    bez = pd.DataFrame(bez,columns=['x','y','z'])

    crs = {'init' :'epsg:4326'}
    import mplleaflet
    import matplotlib.pyplot as plt
    plt.hold(True)
    plt.plot(x0,y0,'go')
    plt.plot(x,y,'r--')
    plt.plot(bez.x, bez.y, 'b')
    mplleaflet.show()


compare(4)
