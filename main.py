from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import insightface

app = FastAPI()

# Habilita CORS para permitir peticiones desde tu frontend (Django, React, etc)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia por ["http://localhost:8000"] si solo usas Django local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

def cargar_modelo():
    global model
    if model is None:
        model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(640, 640))
    return model

@app.post("/generar_embedding/")
async def generar_embedding_api(file: UploadFile = File(...)):
    # Recibe una imagen y retorna el embedding
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "No se pudo decodificar la imagen."}, status_code=400)
    model = cargar_modelo()
    faces = model.get(img)
    if not faces:
        return JSONResponse({"error": "No se detectó rostro."}, status_code=400)
    embedding = faces[0]['embedding'].tolist()
    return {"embedding": embedding}

@app.post("/match_faces/")
async def match_faces_api(
    file: UploadFile = File(...),
    embeddings: str = Form(...)
):
    # embeddings debe ser una lista de listas JSON (embeddings de estudiantes)
    import json
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "No se pudo decodificar la imagen."}, status_code=400)
    model = cargar_modelo()
    faces = model.get(img)
    if not faces:
        return JSONResponse({"error": "No se detectó rostro."}, status_code=400)
    embedding_input = faces[0]['embedding']
    try:
        embeddings_list = json.loads(embeddings)
    except Exception as e:
        return JSONResponse({"error": f"Embeddings JSON inválido: {str(e)}"}, status_code=400)
    # Matching con todos los embeddings (ArcFace: coseno > 0.5 ~ match)
    resultados = []
    for idx, emb in enumerate(embeddings_list):
        emb = np.array(emb, dtype=np.float32)
        sim = np.dot(emb, embedding_input) / (np.linalg.norm(emb) * np.linalg.norm(embedding_input))
        resultados.append({"idx": idx, "similarity": float(sim)})
    return {"resultados": resultados}
