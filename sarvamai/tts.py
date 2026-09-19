from sarvamai import SarvamAI
from sarvamai.play import save  
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise ValueError("SARVAM_API_KEY not found in .env")

client = SarvamAI(api_subscription_key=api_key)


text = "hello,it's me Paramjit Saikia from Jorhat but live in guwahati for educational purpose . now i'm testing you sarvam ai to generate audio from text" 
try:
    audio = client.text_to_speech.convert(
        text=text,
        target_language_code="en-IN", 
        model="bulbul:v3", 
        speaker="shubh"        
    )

    save(audio, "output_tts.wav")
    print("Audio saved as output_tts.wav – play it!")

except Exception as e:
    print("TTS Error:", str(e))