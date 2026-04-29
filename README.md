<<<<<<< HEAD
# Wedding Face Recognition 📸

A face recognition system for wedding events. Guests upload a selfie and instantly find every event photo they appear in — no app download required.

---

## Features

- **Admin panel** — password-protected upload of event photos to the dataset
- **Guest search** — upload a selfie and get back every matching photo
- **Download button** on every matched photo
- **QR code** auto-generated so guests can scan and open on their phones
- **Connection status** indicator — always shows if the backend is reachable

---

## Project Structure

```
wedding-face-recognition/
├── backend/
│   ├── main.py              # FastAPI backend (face detection + matching)
│   ├── requirements.txt     # Python dependencies
│   └── storage/
│       └── uploads/         # Event photos are stored here
│
└── frontend/
    └── index.html           # Complete frontend (HTML + CSS + JS in one file)
```

---

## Setup & Running

### Step 1 — Create a virtual environment (one time only)

```powershell
cd C:\path\to\wedding-face-recognition
python -m venv venv
```

### Step 2 — Activate the virtual environment

```powershell
venv\Scripts\activate
```

### Step 3 — Install dependencies (one time only)

```powershell
pip install -r backend/requirements.txt
```

### Step 4 — Start the backend

Open a terminal and run:

```powershell
python -m uvicorn main:app --port 8001 --app-dir backend
```

You should see:
```
Uvicorn running on http://127.0.0.1:8001
```

Keep this terminal open while using the app.

### Step 5 — Start the frontend

Open a **second** terminal and run:

```powershell
cd frontend
python -m http.server 5500
```

### Step 6 — Open in browser

```
http://127.0.0.1:5500
```

---

## How to Use

### As Admin (uploading event photos)

1. Click the **Admin 🔒** tab
2. Enter the password: `wedding123`
3. Click the upload zone and select all event photos (you can select many at once)
4. Wait for the "Uploaded successfully" message

### As a Guest (finding your photos)

1. Click **Find My Photos** tab
2. Upload a clear selfie (front-facing, good lighting works best)
3. Click **Search My Photos**
4. Your matching photos appear — click **Download** on any to save it

---

## Configuration

### Change the admin password

Edit **both** files:

`backend/main.py` — line 8:
```python
ADMIN_PASSWORD = "your-new-password"
```

`frontend/index.html` — near the top of the `<script>` block:
```javascript
const ADMIN_PASSWORD = 'your-new-password';
```

---

## How Face Matching Works

The backend uses **OpenCV** with a Haar Cascade classifier to detect faces, then compares them using histogram correlation. A match is recorded when the correlation score exceeds **0.7** (70%).

**Tips for best results:**
- Upload front-facing, well-lit photos to the dataset
- Guest selfies should be clear and front-facing
- The same person photographed in very different lighting or angles may not match

---

## Deploying for Real Events (guests on their phones)

`127.0.0.1` only works on your own machine. For guests on other devices, you need to deploy the backend online. Free options:

| Platform | Steps |
|----------|-------|
| [Render](https://render.com) | Connect GitHub repo → New Web Service → set start command |
| [Railway](https://railway.app) | Connect GitHub repo → Deploy → set start command |

Start command for both:
```
uvicorn main:app --host 0.0.0.0 --port $PORT --app-dir backend
```

After deploying, open the frontend **Settings ⚙️** tab and update the Backend URL to your deployed URL.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for the API |
| `uvicorn` | ASGI server to run FastAPI |
| `python-multipart` | Required for file uploads |
| `opencv-python` | Face detection and comparison |
| `numpy` | Array operations (used by OpenCV) |

---

## License

MIT — free to use and modify for personal or commercial projects.
=======
# Wed_Snap
>>>>>>> 89721dd336800729bf69a1a62c34111d1f757529
