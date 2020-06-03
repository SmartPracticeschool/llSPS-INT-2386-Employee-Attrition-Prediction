
import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Age':32,'DailyRate':26, 'DistanceFromHome':12, 'MonthlyIncome':2985, 'OverTime':0})

print(r.json())
