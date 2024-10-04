from flask import Flask, render_template, Response, request
from concurrent.futures import ThreadPoolExecutor
import json
import requests
import sseclient  # pip install sseclient-py
import pyttsx3  # pip install pyttsx3
import threading
import queue
import re
import os

app = Flask(__name__)

# API endpoint
url = "http://127.0.0.1:7690/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize an empty list to keep track of the conversation history
history = []
history2 = []
# Create a queue for TTS requests
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:  # Exit the thread if None is received
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

# Start the TTS worker thread
threading.Thread(target=tts_worker, daemon=True).start()

def load_history():
    """Load conversation history from data.json."""
    if os.path.exists('data2.json'):
        with open('data2.json', 'r') as f:
            return json.load(f)
    return []

def save_history():
    """Save current conversation history to data.json."""
    with open('data2.json', 'w') as f:
        json.dump(history, f, indent=4)

def format_code_start(text, count):
    if count == 1:
        return re.sub(r'```', r"<br><pre id='codesnap' class='all-pre'><code>", text)
    elif count == 2:
        return re.sub(r'```', r"</code></pre>", text)

def get_stream_response():
    data = {
        "messages": history2,
        "mode": "instruct",
        "character": "sudi",
        "instruction_template": "Alpaca",
        "max_tokens": 1000,
        "temperature": 1,
        "top_p": 0.9,
        "seed": 10,
        "stream": True,
    }

    complete_text = ""
    
    try:
        stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
        stream_response.raise_for_status()
        client = sseclient.SSEClient(stream_response)

        for event in client.events():
            payload = json.loads(event.data)
            if 'choices' in payload and len(payload['choices']) > 0:
                response_text = payload['choices'][0]['delta'].get('content', '')
                if response_text:
                    complete_text += response_text
                    yield response_text  # Stream the formatted text response

        # After streaming is done, append complete response to history
        if complete_text:
            assistant_response = {"role": "assistant", "content": complete_text}
            history.append(assistant_response)
            save_history()  # Save updated history to file
            tts_queue.put(complete_text)  # Queue the complete response for TTS
            
    except requests.exceptions.RequestException as e:
        yield f"An error occurred: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream', methods=['POST'])
def stream():
    user_input = request.form['user_input']
    
    # Append the user's input to history
    # history.append({"role": "user", "content": user_input})
    history2.append({"role": "user", "content": user_input})
    # save_history()  # Save updated history to file

    # Check for matching previous user inputs
    for i in range(len(history) - 1):  # Exclude the latest user input
        if history[i]["role"] == "user" and history[i]["content"].lower() == user_input.lower():
            # If we find a matching user input, return the next assistant response
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                response_text = history[i + 1]["content"]
                # Append this interaction to history
                # history.append({"role": "user", "content": user_input})
                # history.append({"role": "assistant", "content": response_text})
                history2.append({"role": "assistant", "content": response_text})
                # save_history()  # Save updated history to file
                return Response(response_text, mimetype='text/event-stream')

    # If no match found, generate a new response
    if user_input.lower() in ["hi", "hallow"]:
        response_text = "Hallow, how can I assist today?"
        # history.append({"role": "user", "content": user_input})
        # history.append({"role": "assistant", "content": response_text})
        # save_history()  # Save updated history to file
        return Response(response_text, mimetype='text/event-stream')
    else:
        history.append({"role": "user", "content": user_input})
        save_history()
        return Response(get_stream_response(), mimetype='text/event-stream')

@app.route('/clear', methods=['POST'])
def clear_history():
    global history2
    history2 = []  # Clear the conversation history
    return Response("Clearing history is disabled.", mimetype='text/event-stream')

if __name__ == '__main__':
    history = load_history()  # Load history at the start
    app.run(debug=True)


























































































































































































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3
# import threading
# import queue
# import re
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []
# history2 = []

# # Create a queue for TTS requests
# tts_queue = queue.Queue()

# def tts_worker():
#     while True:
#         text = tts_queue.get()
#         if text is None:  # Exit the thread if None is received
#             break
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# # Start the TTS worker thread
# threading.Thread(target=tts_worker, daemon=True).start()

# def load_history():
#     """Load conversation history from data.json."""
#     if os.path.exists('data.json'):
#         with open('data.json', 'r') as f:
#             return json.load(f)
#     return []

# def save_history():
#     """Save current conversation history to data.json with pretty print."""
#     with open('data.json', 'w') as f:
#         json.dump(history, f, indent=4)

# def find_similar_response(user_input):
#     """Find a similar response from the history based on user input."""
#     if not history:
#         return None
    
#     # Extract previous user messages
#     previous_user_inputs = [entry["content"] for entry in history if entry["role"] == "user"]

#     # Append the new user input to the list
#     previous_user_inputs.append(user_input)

#     # Create a TF-IDF vectorizer and fit_transform the user inputs
#     vectorizer = TfidfVectorizer().fit_transform(previous_user_inputs)
#     vectors = vectorizer.toarray()

#     # Compute cosine similarity
#     cosine_matrix = cosine_similarity(vectors)
#     similar_indices = cosine_matrix[-1][:-1]  # Ignore the last row (the new input itself)

#     # If any similarity is above a threshold, return the corresponding assistant response
#     if max(similar_indices) > 0.5:  # You can adjust this threshold
#         # Find the index of the most similar previous input
#         most_similar_index = similar_indices.argmax()
#         # Find the corresponding assistant response
#         if most_similar_index + 1 < len(history) and history[most_similar_index + 1]["role"] == "assistant":
#             return history[most_similar_index + 1]["content"]
    
#     return None

# def get_stream_response():
#     data = {
#         "messages": history2,
#         "mode": "instruct",
#         "character": "sudi",
#         "instruction_template": "Alpaca",
#         "max_tokens": 1000,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     complete_text = ""
    
#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         for event in client.events():
#             payload = json.loads(event.data)
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the formatted text response

#         # After streaming is done, append complete response to history
#         if complete_text:
#             assistant_response = {"role": "assistant", "content": complete_text}
#             history.append(assistant_response)
#             save_history()  # Save updated history to file
#             tts_queue.put(complete_text)  # Queue the complete response for TTS
            
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     # Check for similar previous responses
#     similar_response = find_similar_response(user_input)
#     if similar_response:
#         # If a similar response is found, return it
#         return Response(similar_response, mimetype='text/event-stream')

#     # Append the user's input to history
#     history.append({"role": "user", "content": user_input})
#     history2.append({"role": "user", "content": user_input})
#     save_history()  # Save updated history to file

#     # If no match found, generate a new response
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         history2.append({"role": "assistant", "content": response_text})
#         save_history()  # Save updated history to file
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     history2 = []  # Clear the conversation history
#     return Response("Clearing history is disabled.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     history = load_history()  # Load history at the start
#     app.run(debug=True)

































































































































































































































































































# ----------------the best one----------------------------


# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3
# import threading
# import queue
# import re
# import os

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []
# history2 = []
# # Create a queue for TTS requests
# tts_queue = queue.Queue()

# def tts_worker():
#     while True:
#         text = tts_queue.get()
#         if text is None:  # Exit the thread if None is received
#             break
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# # Start the TTS worker thread
# threading.Thread(target=tts_worker, daemon=True).start()

# def load_history():
#     """Load conversation history from data.json."""
#     if os.path.exists('data.json'):
#         with open('data.json', 'r') as f:
#             return json.load(f)
#     return []

# def save_history():
#     """Save current conversation history to data.json."""
#     with open('data.json', 'w') as f:
#         json.dump(history, f)

# def format_code_start(text, count):
#     if count == 1:
#         return re.sub(r'```', r"<br><pre id='codesnap' class='all-pre'><code>", text)
#     elif count == 2:
#         return re.sub(r'```', r"</code></pre>", text)

# def get_stream_response():
#     data = {
#         "messages": history2,
#         "mode": "instruct",
#         "character": "sudi",
#         "instruction_template": "Alpaca",
#         "max_tokens": 1000,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     complete_text = ""
    
#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         for event in client.events():
#             payload = json.loads(event.data)
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the formatted text response

#         # After streaming is done, append complete response to history
#         if complete_text:
#             assistant_response = {"role": "assistant", "content": complete_text}
#             history.append(assistant_response)
#             save_history()  # Save updated history to file
#             tts_queue.put(complete_text)  # Queue the complete response for TTS
            
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     # Append the user's input to history
#     history.append({"role": "user", "content": user_input})
#     history2.append({"role": "user", "content": user_input})
#     save_history()  # Save updated history to file

#     # Check for matching previous user inputs
#     for i in range(len(history) - 1):  # Exclude the latest user input
#         if history[i]["role"] == "user" and history[i]["content"].lower() == user_input.lower():
#             # If we find a matching user input, return the next assistant response
#             if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
#                 response_text = history[i + 1]["content"]
#                 # Append this interaction to history
#                 history.append({"role": "assistant", "content": response_text})
#                 history2.append({"role": "assistant", "content": response_text})
#                 save_history()  # Save updated history to file
#                 return Response(response_text, mimetype='text/event-stream')

#     # If no match found, generate a new response
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         save_history()  # Save updated history to file
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history2
#     history2 = []  # Clear the conversation history
#     return Response("Clearing history is disabled.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     history = load_history()  # Load history at the start
#     app.run(debug=True)



































































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3
# import threading
# import queue
# import re
# import os

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the current conversation history
# history = []
# # Create a queue for TTS requests
# tts_queue = queue.Queue()

# def tts_worker():
#     while True:
#         text = tts_queue.get()
#         if text is None:  # Exit the thread if None is received
#             break
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# # Start the TTS worker thread
# threading.Thread(target=tts_worker, daemon=True).start()

# def load_history():
#     """Load conversation history from data.json."""
#     if os.path.exists('data.json'):
#         with open('data.json', 'r') as f:
#             return json.load(f)
#     return []

# def save_history():
#     """Save current conversation history to data.json."""
#     with open('data.json', 'w') as f:
#         json.dump(history, f)

# def format_code_start(text, count):
#     if count == 1:
#         return re.sub(r'```', r"<br><pre id='codesnap' class='all-pre'><code>", text)
#     elif count == 2:
#         return re.sub(r'```', r"</code></pre>", text)

# def get_stream_response():
#     data = {
#         "messages": history,
#         "mode": "instruct",
#         "character": "sudi",
#         "instruction_template": "Alpaca",
#         "max_tokens": 1000,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     complete_text = ""
    
#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         for event in client.events():
#             payload = json.loads(event.data)
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the formatted text response

#         # After streaming is done, append complete response to history
#         if complete_text:
#             assistant_response = {"role": "assistant", "content": complete_text}
#             history.append(assistant_response)
#             save_history()  # Save updated history to file
#             tts_queue.put(complete_text)  # Queue the complete response for TTS
            
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     global history
#     history = []  # Clear the current conversation history
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     history.append({"role": "user", "content": user_input})
#     save_history()  # Save updated history to file

#     # Check for matching previous responses
#     for entry in history:
#         if entry["role"] == "assistant" and entry["content"].lower() == user_input.lower():
#             response_text = entry["content"]
#             # Ensure to add this interaction to the history again if needed
#             history.append({"role": "assistant", "content": response_text})
#             save_history()  # Save updated history to file
#             return Response(response_text, mimetype='text/event-stream')

#     # If no match found, generate a new response
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         save_history()  # Save updated history to file
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history
#     history = []  # Clear the current conversation history
#     save_history()  # Save cleared history to file
#     return Response("Conversation history cleared.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     history = load_history()  # Load history at the start
#     app.run(debug=True)





































































































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3
# import threading
# import queue
# import re
# import os

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the current conversation history
# history = []
# # Create a queue for TTS requests
# tts_queue = queue.Queue()

# def tts_worker():
#     while True:
#         text = tts_queue.get()
#         if text is None:  # Exit the thread if None is received
#             break
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# # Start the TTS worker thread
# threading.Thread(target=tts_worker, daemon=True).start()

# def load_history():
#     """Load conversation history from data.json."""
#     if os.path.exists('data.json'):
#         with open('data.json', 'r') as f:
#             return json.load(f)
#     return []

# def save_history():
#     """Save current conversation history to data.json."""
#     with open('data.json', 'w') as f:
#         json.dump(history, f)

# def format_code_start(text, count):
#     if count == 1:
#         return re.sub(r'```', r"<br><pre id='codesnap' class='all-pre'><code>", text)
#     elif count == 2:
#         return re.sub(r'```', r"</code></pre>", text)

# def get_stream_response():
#     data = {
#         "messages": history,
#         "mode": "instruct",
#         "character": "sudi",
#         "instruction_template": "Alpaca",
#         "max_tokens": 1000,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the formatted text response

#         # Queue the complete response for TTS
#         if complete_text:
#             tts_queue.put(complete_text)
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     global history
#     history = []  # Clear the current conversation history
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     history.append({"role": "user", "content": user_input})
#     save_history()  # Save updated history to file

#     # Check for matching previous responses
#     for entry in history:
#         if entry["role"] == "assistant" and entry["content"].lower() == user_input.lower():
#             response_text = entry["content"]
#             # Ensure to add this interaction to the history again if needed
#             history.append({"role": "assistant", "content": response_text})
#             save_history()  # Save updated history to file
#             return Response(response_text, mimetype='text/event-stream')

#     # If no match found, generate a new response
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         save_history()  # Save updated history to file
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history
#     history = []  # Clear the current conversation history
#     save_history()  # Save cleared history to file
#     return Response("Conversation history cleared.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     history = load_history()  # Load history at the start
#     app.run(debug=True)


















































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3
# import threading
# import queue
# import re
# import os

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []
# # Create a queue for TTS requests
# tts_queue = queue.Queue()

# def tts_worker():
#     while True:
#         text = tts_queue.get()
#         if text is None:  # Exit the thread if None is received
#             break
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# # Start the TTS worker thread
# threading.Thread(target=tts_worker, daemon=True).start()

# def load_history():
#     """Load conversation history from data.json."""
#     global history
#     if os.path.exists('data.json'):
#         with open('data.json', 'r') as f:
#             history = json.load(f)

# def save_history():
#     """Save conversation history to data.json."""
#     with open('data.json', 'w') as f:
#         json.dump(history, f)

# def format_code_start(text,count):
#     if(count==1):
#         return re.sub(r'```', r"<br><pre id='codesnap' class='all-pre'><code>", text)
#     elif(count==2):
#         return re.sub(r'```', r"</code></pre>", text)

# def get_stream_response():
#     data = {
#         "messages": history,
#         "mode": "instruct",
#         "character": "sudi",
#         "instruction_template": "Alpaca",
#         "max_tokens": 1000,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the formatted text response

#         # Queue the complete response for TTS
#         if complete_text:
#             tts_queue.put(complete_text)
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     global history
#     save_history()  # Save empty history to file
#     history = []  # Clear the conversation history
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     history.append({"role": "user", "content": user_input})
#     save_history()  # Save updated history to file
    
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         save_history()  # Save updated history to file
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history
#     history = []  # Clear the conversation history
#     save_history()  # Save cleared history to file
#     return Response("Conversation history cleared.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     load_history()  # Load history at the start
#     app.run(debug=True)











































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3
# import threading
# import queue
# import re

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []
# # Create a queue for TTS requests
# tts_queue = queue.Queue()

# def tts_worker():
#     while True:
#         text = tts_queue.get()
#         if text is None:  # Exit the thread if None is received
#             break
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# # Start the TTS worker thread
# threading.Thread(target=tts_worker, daemon=True).start()

# def format_code_start(text,count):
#     if(count==1):
#         return re.sub(r'```', r"<br><pre id='codesnap' class='all-pre'><code>", text)
#     elif(count==2):
#         return re.sub(r'```', r"</code></pre>", text)
# def get_stream_response():
#     data = {
#         "messages": history,
#         "mode": "instruct",
#         "character": "sudi",
#         "instruction_template": "Alpaca",
#         "max_tokens": 1000,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text # Stream the formatted text response

#         # Queue the complete response for TTS
#         if complete_text:
#             tts_queue.put(complete_text)
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     global history
#     history = []  # Clear the conversation history
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     history.append({"role": "user", "content": user_input})
    
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history
#     history = []  # Clear the conversation history
#     return Response("Conversation history cleared.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True)


























































































































































































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3
# import threading
# import queue

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []
# # Create a queue for TTS requests
# tts_queue = queue.Queue()

# def tts_worker():
#     while True:
#         text = tts_queue.get()
#         if text is None:  # Exit the thread if None is received
#             break
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# # Start the TTS worker thread
# threading.Thread(target=tts_worker, daemon=True).start()

# def get_stream_response():
#     data = {
#         "messages": history,
#         "mode": "instruct",
#         "character": "sudi",
#         "instruction_template": "Alpaca",
#         "max_tokens": 200,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the text response

#         # Queue the complete response for TTS
#         if complete_text:
#             tts_queue.put(complete_text)
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     global history
#     history = []  # Clear the conversation history
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     history.append({"role": "user", "content": user_input})
    
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history
#     history = []  # Clear the conversation history
#     return Response("Conversation history cleared.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True)



























































































































































































































































































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3
# import threading
# import queue

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []














# # # Function to handle TTS in a separate thread
# # def speak_text(text):
# #     engine.say(text)
# #     engine.runAndWait()

# # def get_stream_response():
# #     data = {
# #         "messages": history,
# #         "mode": "instruct",
# #         "instruction_template": "Alpaca",
# #         "max_tokens": 515,
# #         "temperature": 1,
# #         "top_p": 0.9,
# #         "seed": 10,
# #         "stream": True,
# #     }

# #     try:
# #         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
# #         stream_response.raise_for_status()
# #         client = sseclient.SSEClient(stream_response)

# #         complete_text = ""
# #         for event in client.events():
# #             payload = json.loads(event.data)
# #             print("Payload:", payload)  # Debugging line
            
# #             if 'choices' in payload and len(payload['choices']) > 0:
# #                 response_text = payload['choices'][0]['delta'].get('content', '')
# #                 if response_text:
# #                     complete_text += response_text
# #                     yield response_text  # Stream the text response

# #         # Speak the complete response only once, after streaming
# #         if complete_text:
# #             threading.Thread(target=speak_text, args=(complete_text,)).start()
# #     except requests.exceptions.RequestException as e:
# #         yield f"An error occurred: {e}"





























# # Create a queue for TTS requests
# tts_queue = queue.Queue()

# def tts_worker():
#     while True:
#         text = tts_queue.get()
#         if text is None:  # Exit the thread if None is received
#             break
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# # Start the TTS worker thread
# threading.Thread(target=tts_worker, daemon=True).start()

# def get_stream_response():
#     data = {
#         "messages": history,
#         # "mode": "chat",
#         "mode": "instruct",
#         "character": "sudi",
#         "instruction_template": "Alpaca",
#         "max_tokens": 200,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             print("Payload:", payload)  # Debugging line
            
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the text response

#         # Queue the complete response for TTS
#         if complete_text:
#             tts_queue.put(complete_text)
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"





























# @app.route('/')
# def index():
#     global history
#     history = []  # Clear the conversation history
#     return render_template('index.html')
    

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     history.append({"role": "user", "content": user_input})
    
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history
#     history = []  # Clear the conversation history
#     return Response("Conversation history cleared.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True)





















































































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []

# def get_stream_response():
#     data = {
#         "messages": history,
#         "mode": "instruct",
#         "instruction_template": "Alpaca",
#         "max_tokens": 515,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             print("Payload:", payload)  # Debugging line
            
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the text response

#         # Speak the complete response only once, after streaming
#         if complete_text:
#             engine.say(complete_text)
#             engine.runAndWait()
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     history.append({"role": "user", "content": user_input})
    
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history
#     history = []  # Clear the conversation history
#     return Response("Conversation history cleared.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True)





























































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []

# def get_stream_response():
#     data = {
#         "messages": history,
#         "mode": "instruct",
#         "instruction_template": "Alpaca",
#         "max_tokens": 515,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             print("Payload:", payload)  # Debugging line
            
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text
#             else:
#                 yield "No valid response from the API."

#         engine.say(complete_text)
#         engine.runAndWait()
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     history.append({"role": "user", "content": user_input})
    
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     global history
#     history = []  # Clear the conversation history
#     return Response("Conversation history cleared.", mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True)

















































































# import json
# import requests
# import sseclient  # pip install sseclient-py

# url = "http://127.0.0.1:7690/v1/completions"

# headers = {
#     "Content-Type": "application/json"
# }

# data = {
#     "prompt": "This is a cake recipe:\n\n1.",
#     "max_tokens": 200,
#     "temperature": 1,
#     "top_p": 0.9,
#     "seed": 10,
#     "stream": True,
# }

# stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
# client = sseclient.SSEClient(stream_response)

# print(data['prompt'], end='')
# for event in client.events():
#     payload = json.loads(event.data)
#     print(payload['choices'][0]['text'], end='')

# print()
































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []

# def get_stream_response():
#     data = {
#         "messages": history,  # Send the message history to the API
#         "mode": "instruct",  # Ensure mode is set
#         "instruction_template": "Alpaca",  # Specify instruction template
#         "max_tokens": 515,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             response_text = payload['choices'][0]['message']['content']
#             complete_text += response_text
#             yield response_text  # Stream the text response

#         # Speak the complete response
#         engine.say(complete_text)
#         engine.runAndWait()
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     # Append the user's message to the history
#     history.append({"role": "user", "content": user_input})
    
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True)




































































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize an empty list to keep track of the conversation history
# history = []

# def get_stream_response():
#     data = {
#         "messages": history,  # Send the message history to the API
#         "mode": "instruct",  # Ensure mode is set
#         "instruction_template": "Alpaca",  # Specify instruction template
#         "max_tokens": 515,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             print("Payload:", payload)  # Debugging line
            
#             if 'choices' in payload and len(payload['choices']) > 0:
#                 # Access the content from the delta
#                 response_text = payload['choices'][0]['delta'].get('content', '')
#                 if response_text:
#                     complete_text += response_text
#                     yield response_text  # Stream the text response
#             else:
#                 yield "No valid response from the API."

#         # Speak the complete response
#         engine.say(complete_text)
#         engine.runAndWait()
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"



# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
    
#     # Append the user's message to the history
#     history.append({"role": "user", "content": user_input})
    
#     if user_input.lower() in ["hi", "hallow"]:
#         response_text = "Hallow, how can I assist today?"
#         history.append({"role": "assistant", "content": response_text})
#         return Response(response_text, mimetype='text/event-stream')
#     else:
#         return Response(get_stream_response(), mimetype='text/event-stream')
    
# @app.route('/clear', methods=['POST'])
# def stream():
#     history = []

# if __name__ == '__main__':
#     app.run(debug=True)









































































# from flask import Flask, render_template, Response, request
# import json
# import requests
# import sseclient  # pip install sseclient-py
# import pyttsx3  # pip install pyttsx3

# app = Flask(__name__)

# # API endpoint
# url = "http://127.0.0.1:7690/v1/completions"
# headers = {
#     "Content-Type": "application/json"
# }

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# def get_stream_response(prompt):
#     data = {
#         "prompt": prompt,
#         "max_tokens": 515,
#         "temperature": 1,
#         "top_p": 0.9,
#         "seed": 10,
#         "stream": True,
#     }

#     try:
#         stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
#         stream_response.raise_for_status()
#         client = sseclient.SSEClient(stream_response)

#         complete_text = ""
#         for event in client.events():
#             payload = json.loads(event.data)
#             response_text = payload['choices'][0]['text']
#             complete_text += response_text
#             yield response_text  # Stream the text response

#         # Speak the complete response
#         engine.say(complete_text)
#         engine.runAndWait()
#     except requests.exceptions.RequestException as e:
#         yield f"An error occurred: {e}"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/stream', methods=['POST'])
# def stream():
#     user_input = request.form['user_input']
#     if(user_input=="hi"):
#         return "hallow how can i assist tuday"
#     elif(user_input=="hallow"):
#         return "hallow how can i assist tuday" 
#     else:
#         return Response(get_stream_response(user_input), mimetype='text/event-stream')

# if __name__ == '__main__':
#     app.run(debug=True)