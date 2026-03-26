import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ["YOUR_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 10,
  "max_output_tokens": 1,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service.\n\n    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.\n    {'track_name': 'My City (Bonus Track)', 'track_artist': 'Canon', 'track_album': 'Mad Haven'} - Not Skipped\n\n\n    Will the user skip the next track in the sequence? Here is the next track:\n    {'track_name': 'Start Over (feat. NF)', 'track_artist': 'Flame', 'track_album': 'Royal Flush'}\n\n    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.\n\n    Answer:")

print(response.text)