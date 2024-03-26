from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import librosa
import json
import requests

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")

model = tf.keras.models.load_model('my_model.keras')
SAMPLE_RATE = 22050
TRACK_DURATION = 30
DATA_PATH = "./features_data.json"

CLIENT_ID = 'b2e110f2fc4043e2b1e16d11f7d5d175'
CLIENT_SECRET = '866ff56c9541421a8e1747c5bf9ad1db'

class PredictionResult(BaseModel):
    predicted_label: str

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    z = np.array(data['mapping'])
    return X, y, z

def get_access_token():
    token_url = 'https://accounts.spotify.com/api/token'
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(token_url, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
    token_data = response.json()
    return token_data['access_token']

@app.get("/")
async def read_form(request: Request, predicted_label: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "predicted_label": predicted_label})

@app.post("/predict/")
async def predict_genre(request: Request, file: UploadFile = File(...)):
    X, y, z = load_data(DATA_PATH)
    with open('temp.wav', 'wb') as f:
        f.write(file.file.read())
    signal, sample_rate = librosa.load('temp.wav', sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    mfcc = mfcc[np.newaxis, ..., np.newaxis] 
    mfcc = mfcc[:, :87, :, :]
    prediction = model.predict(mfcc)
    predicted_label = z[np.argmax(prediction)]
    return templates.TemplateResponse("index.html", {"request": request, "predicted_label": predicted_label})

@app.post("/recommends/")
async def recommends(request: Request, predicted_label: str = Form(...)):
    print('aaa: '+predicted_label)
    access_token = get_access_token()
    print(access_token)
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    search_url = f"https://api.spotify.com/v1/search?q=genre:{predicted_label}&type=track&market=TH&limit=50"
    print(search_url)
    response = requests.get(search_url, headers=headers)
    tracks_data = response.json().get("tracks", {}).get("items", [])
    
    
    tracks = [{
        "name": track["name"],
        "artist": track["artists"][0]["name"],
        "cover_image": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
        "preview_url": track["preview_url"],
        "popularity": track["popularity"],
        "spotify_link": track["external_urls"]["spotify"]
    } for track in tracks_data]
    sorted_tracks = sorted(tracks, key=lambda x: x["popularity"], reverse=True)[:5]
    return templates.TemplateResponse("index.html", {"request": request, "predicted_label": predicted_label, "tracks": sorted_tracks})

@app.post("/predictapi/")
async def predict_genre_api(request: Request, file: UploadFile = File(...)):
    X, y, z = load_data(DATA_PATH)
    with open('temp.wav', 'wb') as f:
        f.write(file.file.read())
        print('99999')
    signal, sample_rate = librosa.load('temp.wav', sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    mfcc = mfcc[np.newaxis, ..., np.newaxis] 
    mfcc = mfcc[:, :87, :, :]
    prediction = model.predict(mfcc)
    predicted_label = z[np.argmax(prediction)]
    json_predicted_label = {
        "predicted_label":predicted_label
    }
    json_compatible_item_data = jsonable_encoder(json_predicted_label)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/recommendsapi/")
async def recommends_api(request: Request, predicted_label: str = Form(...)):
    print('aaa: '+predicted_label)
    access_token = get_access_token()
    print(access_token)
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    search_url = f"https://api.spotify.com/v1/search?q=genre:{predicted_label}&type=track&market=TH&limit=50"
    print(search_url)
    response = requests.get(search_url, headers=headers)
    tracks_data = response.json().get("tracks", {}).get("items", [])
    
    tracks = [{
        "name": track["name"],
        "artist": track["artists"][0]["name"],
        "cover_image": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
        "preview_url": track["preview_url"],
        "popularity": track["popularity"],
        "spotify_link": track["external_urls"]["spotify"]
    } for track in tracks_data]
    sorted_tracks = sorted(tracks, key=lambda x: x["popularity"], reverse=True)[:5]
    json_sorted_tracks = {
        "sorted_tracks":sorted_tracks
    }
    json_compatible_item_data = jsonable_encoder(json_sorted_tracks)
    return JSONResponse(content=json_compatible_item_data)