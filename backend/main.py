from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, shutil, pickle, threading
import numpy as np
import faiss
import cv2
from deepface import DeepFace

ADMIN_PASSWORD   = "08420842"
UPLOAD_DIR       = "storage/uploads"
FAISS_INDEX_FILE = "storage/faiss.index"
FILENAMES_FILE   = "storage/filenames.pkl"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL         = "Facenet512"
DETECTOR      = "opencv"
EMBEDDING_DIM = 512
THRESHOLD     = 0.45
MAX_NEIGHBORS = 500

app = FastAPI(title="Wedding Face Recognition API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/storage", StaticFiles(directory="storage"), name="storage")
index_lock = threading.Lock()

def enhance_image(img_path: str):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
        enhanced_path = img_path + "_enh.jpg"
        cv2.imwrite(enhanced_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return enhanced_path
    except Exception:
        return None

def normalize(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def extract_from_path(path: str, min_conf: float = 0.50):
    try:
        results = DeepFace.represent(
            img_path=path,
            model_name=MODEL,
            enforce_detection=False,
            detector_backend=DETECTOR,
        )
        embs = []
        for r in results:
            emb  = r.get("embedding")
            conf = r.get("face_confidence", 1.0)
            if emb and conf >= min_conf:
                embs.append(normalize(np.array(emb, dtype=np.float32)))
        return embs
    except Exception as e:
        print(f"[extract error] {e}")
        return []

def get_embeddings(img_path: str):
    embeddings = extract_from_path(img_path, min_conf=0.60)
    if not embeddings:
        enhanced_path = enhance_image(img_path)
        if enhanced_path:
            try:
                embeddings = extract_from_path(enhanced_path, min_conf=0.40)
            finally:
                try:
                    os.remove(enhanced_path)
                except Exception:
                    pass
    print(f"[embeddings] {os.path.basename(img_path)} → {len(embeddings)} face(s)")
    return embeddings

def load_index():
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FILENAMES_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FILENAMES_FILE, "rb") as f:
            filenames = pickle.load(f)
        return index, filenames
    return faiss.IndexFlatIP(EMBEDDING_DIM), []

def save_index(index, filenames):
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FILENAMES_FILE, "wb") as f:
        pickle.dump(filenames, f)

def already_indexed(filename: str) -> bool:
    """Check if this filename is already in the index."""
    _, filenames = load_index()
    return filename in filenames

@app.get("/")
def root():
    return {"message": "Wedding Face Recognition API is running."}

@app.post("/upload_dataset/")
async def upload_dataset(
    file: UploadFile = File(...),
    password: str = Header(None),
):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Unauthorized")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Skip if already indexed — prevents duplicates on re-upload
    with index_lock:
        _, existing = load_index()
        if file.filename in existing:
            return {"message": f"'{file.filename}' already in dataset.", "faces_found": 0}

    embeddings = get_embeddings(save_path)

    if not embeddings:
        return {"message": f"'{file.filename}' uploaded but no face detected.", "faces_found": 0}
    with index_lock:
        index, filenames = load_index()
        for emb in embeddings:
            index.add(np.array([emb], dtype=np.float32))
            filenames.append(file.filename)
        save_index(index, filenames)

    print(f"[upload] {file.filename} → {len(embeddings)} face(s) indexed")
    return {"message": f"'{file.filename}' uploaded.", "faces_found": len(embeddings)}
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Unauthorized")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embeddings = get_embeddings(save_path)

    if not embeddings:
        return {"message": f"'{file.filename}' uploaded but no face detected.", "faces_found": 0}

    with index_lock:
        index, filenames = load_index()
        for emb in embeddings:
            index.add(np.array([emb], dtype=np.float32))
            filenames.append(file.filename)
        save_index(index, filenames)

    print(f"[upload] {file.filename} → {len(embeddings)} face(s) indexed")
    return {"message": f"'{file.filename}' uploaded.", "faces_found": len(embeddings)}

@app.post("/search/")
async def search(file: UploadFile = File(...)):
    query_path = os.path.join(UPLOAD_DIR, "_query_" + file.filename)
    with open(query_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_embeddings = get_embeddings(query_path)

    try:
        os.remove(query_path)
    except OSError:
        pass

    if not query_embeddings:
        return {"result": "No face detected. Please upload a clear photo.", "count": 0, "photos": []}

    with index_lock:
        index, filenames = load_index()

    if index.ntotal == 0:
        return {"result": "No picture detected", "count": 0, "photos": []}

    print(f"[search] {len(query_embeddings)} query face(s), {index.ntotal} indexed faces")

    # Best score per filename — no duplicates, no noise
    best_scores: dict = {}

    for q_emb in query_embeddings:
        q = np.array([q_emb], dtype=np.float32)
        k = min(MAX_NEIGHBORS, index.ntotal)
        distances, indices = index.search(q, k)
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            fname = filenames[idx]
            score = float(dist)
            if fname not in best_scores or score > best_scores[fname]:
                best_scores[fname] = score

    sorted_scores = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"[search] top scores: {[(f, round(s,4)) for f,s in sorted_scores[:10]]}")

    matched = [fname for fname, score in best_scores.items() if score >= THRESHOLD]
    print(f"[search] matched at threshold {THRESHOLD}: {matched}")

    if not matched:
        return {"result": "No picture detected", "count": 0, "photos": []}

    unique_matched = list(dict.fromkeys(matched))
    photos = ["/storage/uploads/" + f for f in unique_matched]
    return {"result": "Photos found", "count": len(photos), "photos": photos}

@app.get("/dataset/count")
def dataset_count():
    files = [
        f for f in os.listdir(UPLOAD_DIR)
        if not f.startswith("_query_") and
        f.lower().rsplit(".", 1)[-1] in {"jpg", "jpeg", "png", "webp", "bmp"}
    ]
    return {"count": len(files)}

@app.delete("/dataset/clear")
async def clear_dataset(password: str = Header(None)):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Unauthorized")
    with index_lock:
        for f in os.listdir(UPLOAD_DIR):
            if not f.startswith("_query_"):
                try:
                    os.remove(os.path.join(UPLOAD_DIR, f))
                except Exception:
                    pass
        for fpath in [FAISS_INDEX_FILE, FILENAMES_FILE]:
            if os.path.exists(fpath):
                os.remove(fpath)
    return {"message": "Dataset cleared successfully."}