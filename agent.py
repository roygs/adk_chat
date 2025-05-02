from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents import Agent
from google.adk.tools import ToolContext, FunctionTool
from google.genai import types
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    ToolCodeExecution,
    VertexAISearch,
)
from google.adk.tools import google_search

from pydantic import BaseModel, Field
import pathlib
from pathlib import Path
import contextlib
import wave
import base64
import mimetypes
import io
import requests

from playsound import playsound
import mimetypes
import os
import threading
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
APP_NAME = "code_pipeline_app"
USER_ID = "dev_user_01"
SESSION_ID = "pipeline_session_01"
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17" 
INDIAN_LANG_CODES = ['en','hi','bn','gu','kn','ml','mr','od','pa','ta','te']
TEMP_FOLDER = Path("temp") 
TEMP_FOLDER.mkdir(exist_ok=True)
OUTPUT_WAVEFILE = "answer.wav"

api_key = os.environ['SARVAM_API_KEY']

# #input schema for the tool
class InfoAnswer(BaseModel):
    language: str = Field(description="The language in which the user asked the query")
    retreived_answer: str = Field(description="Answer obtained by querying the document")
    hasSpoken: bool = Field(description="Whether the user has typed the answer or not")

@contextlib.contextmanager
def wave_file(filename, channels=1, rate= 8000, #24000
               sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

def get_lang_code(language:str):
    if len(language) == 2:
        return language
    lang_code = {
        "english": "en",
        "hindi": "hi",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "marathi": "mr",
        "punjabi": "pa",
        "bengali": "bn",
        "gujarati": "gu",
        "odia": "or",
        "urdu": "ur",
        "assamese": "as",
        "maithili": "mai",
        "sanskrit": "sa",
        "sindhi": "sd",
        "nepali": "ne",
        "french": "fr",
        "german": "de",
        "greek": "el",
        "gujarati": "gu",
        "japanese": "ja",
        "korean": "ko",
        "marathi": "mr",
        "russian": "ru",
        "tamil": "ta",
        "telugu": "te",
        "urdu": "ur",
        "vietnamese": "vi",
        "chinese": "zh",
        "hindi": "hi",
        "bengali": "bn",
        "gujarati": "gu",
        "kannada": "kn",
        "malayalam": "ml",
        "marathi": "mr",
        "odia": "or",
        "punjabi": "pa",
        "tamil": "ta",
        "telugu": "te",
        "urdu": "ur"
    }
    return lang_code.get(language.lower())

#helper function
def get_translation(text, language):
    """
    Translate the text to the given language
    Returns: translated text
    """
    target_lang_code = get_lang_code(language)
    if target_lang_code not in INDIAN_LANG_CODES:
        translated_text="User has asked question in a language which is not supported."
        target_lang_code = "en-IN"
    else:
        target_lang_code = target_lang_code + "-IN"
        if target_lang_code != "en-IN":
            # if target is not english then translate
            url = "https://api.sarvam.ai/translate"

            payload = {
                "input": text,
                "source_language_code": "en-IN",
                "target_language_code": target_lang_code,
                "speaker_gender": "Female",
                "mode": "formal",
                "model": "mayura:v1",
                "enable_preprocessing": False,
                "output_script": "spoken-form-in-native",
                "numerals_format": "international"
            }
            headers = {"Content-Type": "application/json",
                    'api-subscription-key': api_key
                        }

            response = requests.request("POST", url, json=payload, headers=headers)
            translated_text = response.json()["translated_text"]
        else:
            translated_text = text

    return translated_text, target_lang_code


def speak(text, lang_code):
    """
    Converts text to speech
    Returns: wav file path
    """
    import requests
    out_wav_filename = os.path.join(TEMP_FOLDER, OUTPUT_WAVEFILE)

    #Convert the answer to speech
    #Use SARVAM API
    
    url = "https://api.sarvam.ai/text-to-speech"

    payload = {
    "inputs": [text],
    "speaker": "meera",
    "pitch": 0,
    "pace": 1.25,
    "loudness": 2,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v1",
    "target_language_code": lang_code
    }
    headers = {
        'api-subscription-key': api_key,
        "Content-Type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()
    #print("Request successful")

    audio_data = response.json()['audios']
    delimiter = ""
    audio_string = delimiter.join(audio_data)

    with wave_file(out_wav_filename) as wav:
        audio_bytes = base64.b64decode(audio_string)
        wav.writeframes(audio_bytes)

    #threading.Thread(target=playsound, args=(out_wav_filename,)).start()
    print("Speaking ..")
    playsound(out_wav_filename)

# Tool function
def translate_and_speak(a_dict: InfoAnswer, tool_context:ToolContext) :
    """
        Translate the text and convert to speech.
        Returns: translated_answer
    """
    print(a_dict.get("hasSpoken"))
    translated_answer, lang_code = get_translation(a_dict.get("retreived_answer"), a_dict.get("language"))
    if a_dict.get("hasSpoken"):
        speak(translated_answer, lang_code)
    return translated_answer


translate_and_speak_tool = FunctionTool(func = translate_and_speak)

bhasha_chat_agent = Agent(
    name="bhasha_chat_agent",
    model=GEMINI_MODEL,
    description=(
        "Agent to transcribe audio to text and perform content retreival and answer in the language used by user"
    ),
    instruction=(
    """
    You are a helpful agent. 
    If a query.wav file is provided then user has verbally asked so
        transcribe the query.wav file, 
        determine the language spoken in this file and translate the transcribed text to english. 
    else if no query.wav file provided then user has typed the query so
      determine the language of the text query and translate the text to english.
      After the above step do the following
       If a file has been provided then query the other file with the translated text and generate an answer. The answer
      should be less than 500 characters. 
           
       Finally Pass language used, whether user has spoken and the answer to the translate_and_speak_tool.
      
     """),
    tools=[ translate_and_speak_tool]
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=bhasha_chat_agent, app_name=APP_NAME, session_service=session_service)

#root_agent = bhasha_chat_agent

# Agent Interaction
def call_agent(data_file, query_file, user_prompt=""):
    print(f"datafile {data_file}")
    print(f"queryfile {query_file}")
    final_response="error"
    parts=[]
    if data_file != "":

        mime_type, encoding = mimetypes.guess_type(data_file)
        # add the data file 
        parts=[ 
            types.Part.from_bytes(
            data=Path(data_file).read_bytes(),mime_type=mime_type,)]
    
    if (query_file != ""):
        # Case 1: user has recorded his query.
   
        # ignore user_prompt if user has recorded query
        prompt=""
        parts.append( types.Part.from_bytes(
            data=Path(query_file).read_bytes(), mime_type='audio/wav',))
       
    else:
        # Case 2: query.wav not present, i.e. user has typed his query.
        if user_prompt == "":
            # Case 3: Neither audio nor text query present
            final_response = "Please type or ask a question."
            return final_response
        else:
            prompt=user_prompt
        
    parts.append(types.Part(text=prompt))
    content = types.Content(role='user',parts=parts )
        
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
           
    
    return final_response

# FILES_FOLDER=".\\files"
# df = os.path.join(FILES_FOLDER, "Principal-Sample-Life-Insurance-Policy.pdf")
# qf = os.path.join(FILES_FOLDER,"q_en.wav")

# r = call_agent(df, "", "summarize this document")
# #r = call_agent(df, "", "sfsgss")
# #r = call_agent(df, "", "how are you today")
# print (f"Agent Response: \n {r}")

# query="हिताधिकारी को क्या मिलेगा"

#df = os.path.join(FILES_FOLDER, "Principal-Sample-Life-Insurance-Policy.pdf")
# qf = os.path.join(FILES_FOLDER,"q_hi.wav")

# r = call_agent(df, qf)
# #r = call_agent(df, "", "summarize this document")
# r = call_agent(df, "", "what will beneficiary get")
# # #r = call_agent("",  qf)
# # #r = call_agent(df, qf,"how are you") #"how are you"
# # #r = call_agent(df, "","tum kaise ho")
# # #r = call_agent(df, qf,"how are you")
# #r = call_agent(df, "", query)
# print (f"Agent Response: \n {r}")