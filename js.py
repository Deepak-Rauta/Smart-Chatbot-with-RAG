import requests

url = "http://127.0.0.1:8000/chatbot"
data = {"query": "What is AI?"}

response = requests.post(url, json=data)
print(response.json())  # Should return the chatbot response
