from ollama import Client

try:
    client = Client(host='http://localhost:11434')
    print("Connected to Ollama.")
    models = client.list()
    print(models)
    for m in models['models']:
        print(m)
except Exception as e:
    print(f"Error: {e}")
