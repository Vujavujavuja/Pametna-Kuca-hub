import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import os

load_dotenv("keys.env")
speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_REGION")
print(f"Speech Key: {speech_key}")
print(f"Service Region: {service_region}")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language="sr-RS"
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

print("Speak into your microphone.")
result = speech_recognizer.recognize_once()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("Recognized:", result.text)
elif result.reason == speechsdk.ResultReason.NoMatch:
    print("No speech could be recognized.")
elif result.reason == speechsdk.ResultReason.Canceled:
    print("Speech Recognition canceled:", result.cancellation_details.reason)
