import pandas as pd
import sqlalchemy as sa


def main():
  engine = sa.create_engine("crate://db.world.io:4200")

  bostonBB = ("POLYGON(("
              "-71.70501708984375 42.99579622431779,-70.27679443359375 42.99579622431779,"
              "-70.27679443359375 41.77867279318799,-71.70501708984375 41.77867279318799,"
              "-71.70501708984375 42.99579622431779))")
  print(bostonBB)

  q = f"""
  select data['t_deviceid'], count(1)
    from cuebiq
   where MATCH(shape, '{bostonBB}') using within
     and (data['i_devicetime'] > 1473897600
     and data['i_devicetime'] < 1474070400)
     and data['i_accuracy'] < 200
   group by data['t_deviceid']
   order by count(1) DESC
   limit 5
   """

  df = pd.read_sql(q, engine)
  df.head()

  q = f"""
  select data['t_deviceid'], data['i_devicetime'], data['t_lastseen'],
         data['i_tzoffset'], data['i_accuracy'], point
    from cuebiq
   where data['t_deviceid'] = '{df["data['t_deviceid']"][0]}'
     and data['i_accuracy'] < 200
   limit {df['count(1)'][0]}"""

  cue = pd.read_sql(q, engine)
  cue.columns = ['deviceID', 'deviceTime', 'lastSeen', 'tzOffset', 'accuracy', 'point']
  cue['key'] = cue['deviceTime'].apply(str) + cue['lastSeen'].apply(str)

  cue = cue.drop_duplicates(subset=['key'])
  p = cue['point'].str
  cue['x'] = p[0]
  cue['y'] = p[1]
  cue['day'] = (cue['deviceTime'] + cue['tzOffset']) // 86400
  print("counts", cue.day.value_counts())

  extract = cue.loc[cue.day == 17060]
  extract = extract.sort_values(by='deviceTime')
  extract['rank'] = range(extract.shape[0])
  extract.to_csv('oneDeviceOneday.tsv', index=False, sep='\t')

  print("head", cue.head())
  print("shape", extract.shape)


if __name__ == "__main__":
  main()
