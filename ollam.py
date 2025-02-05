import requests

OLLAMA = 'http://localhost:11434/api/generate'

def query_ollama(model_name, prompt):
    headers= { "Content-Type": "application/json" }
    payload = {"model": model_name, "prompt": prompt}

    response = requests.post(OLLAMA, 
                             json=payload, 
                             headers=headers)
    print("Raw json content", response.text)

    if response.status_code == 200:
        return response.json().get('response', 'No response')
    else:
        return 'Error: ' + response.text
    
print("Ollama Demo")

model = "qwen2.5-coder:0.5b"

user_prompt = input("Enter your prompt: ")

print(query_ollama(model, user_prompt))