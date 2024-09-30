from flask import Flask, render_template, Response, request
import json
import requests
import sseclient  # pip install sseclient-py
import pyttsx3  # pip install pyttsx3

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

def get_stream_response():
    data = {
        "messages": history,
        "mode": "instruct",
        "instruction_template": "Alpaca",
        "max_tokens": 515,
        "temperature": 1,
        "top_p": 0.9,
        "seed": 10,
        "stream": True,
    }

    try:
        stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
        stream_response.raise_for_status()
        client = sseclient.SSEClient(stream_response)

        complete_text = ""
        for event in client.events():
            payload = json.loads(event.data)
            print("Payload:", payload)  # Debugging line
            
            if 'choices' in payload and len(payload['choices']) > 0:
                response_text = payload['choices'][0]['delta'].get('content', '')
                if response_text:
                    complete_text += response_text
                    yield response_text  # Stream the text response

        # Speak the complete response only once, after streaming
        if complete_text:
            engine.say(complete_text)
            engine.runAndWait()
    except requests.exceptions.RequestException as e:
        yield f"An error occurred: {e}"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream', methods=['POST'])
def stream():
    user_input = request.form['user_input']
    
    history.append({"role": "user", "content": user_input})
    
    if user_input.lower() in ["hi", "hallow"]:
        response_text = "Hallow, how can I assist today?"
        history.append({"role": "assistant", "content": response_text})
        return Response(response_text, mimetype='text/event-stream')
    else:
        return Response(get_stream_response(), mimetype='text/event-stream')

@app.route('/clear', methods=['POST'])
def clear_history():
    global history
    history = []  # Clear the conversation history
    return Response("Conversation history cleared.", mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)





























































































































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