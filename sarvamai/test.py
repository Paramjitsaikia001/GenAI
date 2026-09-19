from sarvamai import SarvamAI
from dotenv import load_dotenv
import os

load_dotenv()  # This loads .env into os.environ

# Get the key from env (safe & still uses .env)
api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError("SARVAM_API_KEY not found in .env file or environment")

client = SarvamAI(
    api_subscription_key=api_key   # ← this is the correct parameter name
)

# Rest of your code...
response = client.speech_to_text.transcribe(
    file=open("test.mp3", "rb"),  # note: if it's .m4a (common Apple format), rename to .m4a if needed
    model="saaras:v3",
    mode="transcribe",
    # language_code="as-IN"  # if Assamese/Nagaland dialect, try forcing it
)

print("Transcription:", response)