from ollama import Client

def get_ollama_models():
    client = Client(host='http://localhost:11434')
    models = client.list()
    return models
