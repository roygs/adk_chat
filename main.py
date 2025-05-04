from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import os
import shutil
from pathlib import Path
from agent import call_agent
#from agent import get_audio_answer
import constants

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Create directories if they don't exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
#os.makedirs(os.path.join(STATIC_DIR, "audio"), exist_ok=True)

# Create upload directory if it doesn't exist
TEMP_DIR = Path(constants.TEMP_FOLDER) 
TEMP_DIR.mkdir(exist_ok=True)

out_wav_filename = os.path.join(TEMP_DIR, constants.OUTPUT_WAVEFILE)
#out_wav_filename = "/static/answer.wav"

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # code to execute when app is loading
    print("loading app")
    yield
    # code to execute when app is shutting down
    print("closing")

app = FastAPI(lifespan=app_lifespan)


# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount temp files directory
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "audioAnswerURL": ""})

@app.post("/submit_form", response_class=HTMLResponse)
async def process_form(
    request: Request,
    text_input: str = Form(None),
    file: UploadFile = File(None),
    voice_recorded: bool = Form(False)
):
    result = ""
    data_file=""
    query_file=""
    message=""
    
    if text_input:
        #result += f"Text input received: {text_input}"
        message += f"{text_input} "
    elif voice_recorded:
        message += "Voice input received \n"
        query_file = os.path.join(TEMP_DIR,constants.QUERY_FILE)
    else:
        message = "No input provided"
    if file and file.filename:
        # Save the uploaded file
        file_path = TEMP_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        message += f"File uploaded: {file.filename}\n"
        data_file = os.path.join(TEMP_DIR, file.filename)
    

    result += call_agent(data_file, query_file, text_input)
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "result": result, "message":message, "audioAnswerURL":"temp/answer.wav"}
    )

@app.post("/save-audio")
async def save_audio(request: Request):
    # Read the raw audio data from the request body
    audio_data = await request.body()
    
    # Save it to the upload directory as query.wav
    with open(TEMP_DIR / constants.QUERY_FILE, "wb") as f:
        f.write(audio_data)
    
    return {"success": True}

@app.get("/clear-files")
def clear_files( request: Request):
    print("clearing")
    # Clear the contents of the upload directory
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@app.get("/answer-audio")
def answer_audio():
    print("answering audio")
    #audioAnswerURL = URL.createObjectURL(out_wav_filename)
    return FileResponse(out_wav_filename, media_type="audio/wav")

# Add a route to serve the templates directory
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)