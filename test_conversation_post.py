import requests

url = "http://127.0.0.1:8000/conversation"
payload = {"prompt": "I have combination skin."}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print("Status:", response.status_code)
print("Response:", response.text)
