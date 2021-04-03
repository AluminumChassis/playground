import requests

token = 'pk_690c3bca132946b0a2d36d3ba1c0a518'
url = 'https://cloud.iexapis.com/stable/stock/AAPL/financials?token=' + token + '&period=annual'

headers = {'Content-Type': 'application/json'}

response = requests.request('GET', url)

print(response.text)