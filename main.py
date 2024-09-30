from flask import Flask, render_template, Response, request
import json
import requests
import sseclient  # pip install sseclient-py
import pyttsx3  # pip install pyttsx3

app = Flask(__name__)

# API endpoint
url = "http://127.0.0.1:7690/v1/completions"
headers = {
    "Content-Type": "application/json"
}

# Initialize text-to-speech engine
engine = pyttsx3.init()

def get_stream_response(prompt):
    data = {
        "prompt": prompt,
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
            response_text = payload['choices'][0]['text']
            complete_text += response_text
            yield response_text  # Stream the text response

        # Speak the complete response
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
    return Response(get_stream_response(user_input), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)