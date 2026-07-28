from datetime import datetime
def add_time_features(weather):
    last_updated=weather.get('last_updated')
    dt=datetime.strptime(last_updated,'%Y-%m-%d %H:%M')
    weather['year']=dt.year
    weather['month']=dt.month
    weather['day']=dt.day
    weather['hour']=dt.hour
    weather['minute']=dt.minute
    weather['weekday']=dt.strftime('%A')
    return weather