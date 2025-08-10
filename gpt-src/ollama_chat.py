import requests

API_URL = "http://localhost:11434/api/chat"
MODEL = "gpt-oss:20b"  # 確保已經 pull 下來

BASE_URL = "http://localhost:11434"

def get_api_url(endpoint):
    return f"{BASE_URL}{endpoint}"



def chat_with_ollama(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False  # True 時會邊生成邊傳
    }
    res = requests.post(API_URL, json=payload)
    return res.json()

if __name__ == "__main__":
    reply = chat_with_ollama("用繁體中文簡短介紹你自己")
    print(reply["message"]["content"])
