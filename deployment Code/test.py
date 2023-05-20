import requests

resp = requests.post("https://getprediction-bhdhvw323q-el.a.run.app", files={'message':"RED LEHENGA WITH EMBROIDERED BLOUSE"})

print(resp.json()['urls'])