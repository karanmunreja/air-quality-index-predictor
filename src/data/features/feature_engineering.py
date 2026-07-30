from datetime import datetime
def add_time_features(weather):
    timestamp=weather.get('last_updated') or weather.get('time')
    if not timestamp:
        raise ValueError('Timestamp Not Found')
    if " " in timestamp:
        dt=datetime.strptime(timestamp,'%Y-%m-%d %H:%M')
    else:
        dt=datetime.strptime(timestamp, "%Y-%m-%dT%H:%M")
    weather['year']=dt.year
    weather['month']=dt.month
    weather['day']=dt.day
    weather['hour']=dt.hour
    weather['minute']=dt.minute
    weather['weekday']=dt.strftime('%A')
    return weather
