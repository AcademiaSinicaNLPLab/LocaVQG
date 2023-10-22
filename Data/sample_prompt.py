import requests
import json
import tqdm
import time

API_KEY = YOUR_API_KEY
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

def generate_chat_completion(messages, model="gpt-4", temperature=0.7, max_tokens=None, presence_penalty=0.1):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    response = requests.post(
        API_ENDPOINT, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

prompt = f"Here are some descriptions of your surroundings:\n\n"

address_prompt = f"You are currently driving in 232 First Avenue, Pittsburgh, PA 15222, USA.\n\n"
caption_prompt = f"On your north, an empty building with cars parked outside of it.\nOn your east, a city street with buildings and a fire hydrant.\nOn your south, a man walking in front of a brick building.\nOn your west, a city street with cars parked on the side of a street.\n\n\n"

postfix_prompt = "Based on those descriptions, please ask 10 engaging questions about it"

message = prompt + address_prompt + caption_prompt + postfix_prompt

messages = [
    {"role": "system", "content": "You are a tour guide and you are currently driving in a car with your tourists. You want to engage with them with any kind of information you have around you."},
    {"role": "user", "content": message}
]

response_text = generate_chat_completion(messages)


print(response_text)