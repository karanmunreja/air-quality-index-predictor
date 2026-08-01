def merge_data(weather_data, aqi_data):
    weather_hourly=weather_data['hourly']
    aqi_hourly=aqi_data['hourly']
    merged_data=[]
    num_records=len(weather_hourly['time'])
    for i in range(num_records):
        record={}
        for key,values in weather_hourly.items():
            record[key] = values[i]
        for key,values in aqi_hourly.items():
            record[key]=values[i]
        merged_data.append(record)
    return merged_data
