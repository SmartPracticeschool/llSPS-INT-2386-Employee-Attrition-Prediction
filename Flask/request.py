
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':32,'Daily Rate':26, 'Distance From Home':29, 'Monthly Income':2985, 'Over Time':0})

print(r.json())
