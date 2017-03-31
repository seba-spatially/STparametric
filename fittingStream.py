from functions import get_metro_ingestid, query_device_ids, get_phone_data, st_prep_first


def main():
    # Get the ingest ID for the phone data in a given MSA
    ingestid = get_metro_ingestid('boston')

    # Find deviceIDs in that MSA
    f = query_device_ids(ingestid)
    f = f.ix[0:100]
    f.head()
    no_p = f["count"].sum()
    print("100 deviceIDs have originally {} points".format(no_p))

    from sqlalchemy import create_engine
    engine = create_engine('postgresql://postgres:postgres@localhost:5434/SpatioTemporal')
    c = []
    for ind, row in f.iterrows():
        print(ind)
        # Get the Phone data for a deviceID in a MSA
        data = get_phone_data(ingestid=ingestid, device_id=row["deviceid"], co=row["count"], acc=200)
        ###prepare the parametric shapes
        h = st_prep_first(data)
        prep = h[['i', 'ij', 'geometry']]
        prep['deviceid'] = row["deviceid"]
        prep.to_sql("data", engine, schema='staging')
        c.append(prep.shape[0])


def test_with_quadratic():
    """
    THIS BLOCK IS A TEST WITH A QUADRATIC FUNCTION FIT
    """
    from functions import coord_parser, bezier_solver, st_prep
    import pandas as pd

    data = pd.read_csv('oneDeviceOneDay.tsv', sep='\t')
    data = data[['local_time', 'lastSeen', 'point', 'accuracy']]
    p = [coord_parser(x) for x in data.point]
    x = [float(i[0]) for i in p]
    y = [float(i[1]) for i in p]
    data['x'] = x
    data['y'] = y
    del data['point']
    print(data.head())

    # sort data by local_time
    dd = data.sort_values(by='local_time', axis=0)

    h = st_prep(dd)
    print(h.head())

    control = []
    for index, row in h.iterrows():
        c = bezier_solver(row)
        control.append(c)

    print('done')

    h['bezContrPts'] = control
    from shapely.geometry import LineString

    h.head()

    lines = [LineString(i) for i in h['bezContrPts']]
    from geojson import dumps
    l = [dumps(i) for i in lines]
    print(l[0])
    print(lines[0])
    h['Linestring'] = lines

    print(h.head())
    h.to_csv('bezierControlPoints.tsv', index=False, sep='\t')

    geometry = h[['Linestring', 'i', 'ij']]
    print(geometry.head())
    geometry.to_csv('beziergeometry.csv', index=False, sep='|', header=None)
    line = LineString(h.iloc[0]['bezContrPts'])


if __name__ == "__main__":
    main()
