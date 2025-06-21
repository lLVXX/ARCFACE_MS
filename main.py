import psycopg2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from io import BytesIO
from PIL import Image
from collections import defaultdict

# CONFIG
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "12345678"
PG_DB = "SCOUT_DB"

PG_QUERY = """
    SELECT estudiante_id, embedding
    FROM personas_estudiantefoto
    WHERE embedding IS NOT NULL
"""

THRESHOLD = 0.5  # <--- Ajusta aquí (0.5 recomendado)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_multi_embeddings_from_db():
    conn = psycopg2.connect(
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
    )
    cur = conn.cursor()
    cur.execute(PG_QUERY)
    data = cur.fetchall()
    cur.close()
    conn.close()
    embeddings_dict = defaultdict(list)
    for est_id, emb_bin in data:
        emb = np.frombuffer(emb_bin, dtype=np.float32)
        embeddings_dict[est_id].append(emb)
    return embeddings_dict

@app.post("/match_faces/")
async def match_faces(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(img)
    faces = face_app.get(np_img)
    if not faces:
        return {"ok": False, "msg": "No se detectaron rostros"}

    # --- Matching Multi-Embedding ---
    embeddings_dict = get_multi_embeddings_from_db()
    results = []
    for face in faces:
        query_emb = face.embedding
        best_score = -1
        best_est_id = None

        for est_id, emb_list in embeddings_dict.items():
            # Max similarity among all embeddings of this estudiante
            max_sim = max(
                np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
                for emb in emb_list
            )
            if max_sim > best_score:
                best_score = max_sim
                best_est_id = est_id

        results.append({
            "face_box": face.bbox.tolist(),
            "estudiante_id": int(best_est_id) if best_score > THRESHOLD else None,
            "similarity": float(best_score),
            "match": bool(best_score > THRESHOLD)
        })

    return {
        "ok": True,
        "num_faces": len(faces),
        "results": results,
    }



@app.post("/generar_embedding/")
async def generar_embedding(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(img)
    faces = face_app.get(np_img)
    if not faces:
        return {"ok": False, "msg": "No se detectaron rostros"}
    # Retorna solo el embedding del rostro más grande (o el primero)
    embedding = faces[0].embedding.tolist()
    return {"ok": True, "embedding": embedding}



@app.get("/")
def healthcheck():
    return {"status": "ok"}
