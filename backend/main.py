from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, pickle, threading, shutil
import numpy as np
import faiss
import cv2
import cloudinary
import cloudinary.uploader
import cloudinary.api
from deepface import DeepFace

ADMIN_PASSWORD = "08420842"

cloudinary.config(
    cloud_name = "dffkk8ano",
    api_key    = "637784848439562",
    api_secret = "YUfhmVRX6TcQQoDRlkrgVvqJrfo"
)

STORAGE_DIR      = "storage"
FAISS_INDEX_FILE = "storage/faiss.index"
FILENAMES_FILE   = "storage/filenames.pkl"
os.makedirs(STORAGE_DIR, exist_ok=True)

MODEL         = "Facenet512"
DETECTOR      = "opencv"
EMBEDDING_DIM = 512
THRESHOLD     = 0.45
MAX_NEIGHBORS = 500

app = FastAPI(title="WedSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
index_lock = threading.Lock()

def enhance_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_AREA)
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

def extract_from_path(path, min_conf=0.50):
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

def get_embeddings(img_path):
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
            data = pickle.load(f)
        return index, data["filenames"], data["urls"]
    return faiss.IndexFlatIP(EMBEDDING_DIM), [], {}

def save_index(index, filenames, urls):
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FILENAMES_FILE, "wb") as f:
        pickle.dump({"filenames": filenames, "urls": urls}, f)

@app.get("/")
def root():
    return {"message": "WedSnap API is running."}

@app.post("/upload_dataset/")
async def upload_dataset(
    file: UploadFile = File(...),
    password: str = Header(None),
):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Unauthorized")

    tmp_path = os.path.join(STORAGE_DIR, "_tmp_" + file.filename)
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with index_lock:
        _, existing_filenames, existing_urls = load_index()
        if file.filename in existing_filenames:
            os.remove(tmp_path)
            return {"message": f"'{file.filename}' already in dataset.", "faces_found": 0}

    embeddings = get_embeddings(tmp_path)

    if not embeddings:
        os.remove(tmp_path)
        return {"message": f"'{file.filename}' uploaded but no face detected.", "faces_found": 0}

    try:
        result = cloudinary.uploader.upload(
            tmp_path,
            folder="wedsnap",
            public_id=file.filename.rsplit(".", 1)[0],
            overwrite=True
        )
        photo_url = result["secure_url"]
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Cloudinary upload failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    with index_lock:
        index, filenames, urls = load_index()
        for emb in embeddings:
            index.add(np.array([emb], dtype=np.float32))
            filenames.append(file.filename)
        urls[file.filename] = photo_url
        save_index(index, filenames, urls)

    print(f"[upload] {file.filename} → {len(embeddings)} face(s) → {photo_url}")
    return {"message": f"'{file.filename}' uploaded.", "faces_found": len(embeddings)}

@app.post("/search/")
async def search(file: UploadFile = File(...)):
    tmp_path = os.path.join(STORAGE_DIR, "_query_" + file.filename)
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_embeddings = get_embeddings(tmp_path)

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    if not query_embeddings:
        return {"result": "No face detected. Please upload a clear photo.",
                "count": 0, "photos": []}

    with index_lock:
        index, filenames, urls = load_index()

    if index.ntotal == 0:
        return {"result": "No picture detected", "count": 0, "photos": []}

    best_scores = {}
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

    matched = [fname for fname, score in best_scores.items() if score >= THRESHOLD]
    unique_matched = list(dict.fromkeys(matched))

    if not unique_matched:
        return {"result": "No picture detected", "count": 0, "photos": []}

    photos = [urls[f] for f in unique_matched if f in urls]
    return {"result": "Photos found", "count": len(photos), "photos": photos}

@app.get("/dataset/count")
def dataset_count():
    with index_lock:
        _, filenames, _ = load_index()
    return {"count": len(set(filenames))}

@app.delete("/dataset/clear")
async def clear_dataset(password: str = Header(None)):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        cloudinary.api.delete_resources_by_prefix("wedsnap/")
    except Exception as e:
        print(f"[clear] Cloudinary error: {e}")
    with index_lock:
        for fpath in [FAISS_INDEX_FILE, FILENAMES_FILE]:
            if os.path.exists(fpath):
                os.remove(fpath)
    return {"message": "Dataset cleared successfully."}