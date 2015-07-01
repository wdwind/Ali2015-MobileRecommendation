from datetime import datetime

dt0 = datetime.strptime('2014-11-18 00', '%Y-%m-%d %H')
#dt1 = datetime.strptime('2014-12-18 00', '%Y-%m-%d %H')

duration_hours = lambda x, y: int((x - y).total_seconds() / 3600)

#time_thresh = duration_hours(dt1, dt0)