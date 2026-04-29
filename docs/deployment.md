# Deployment

The Streamlit app at `app/streamlit_app.py` is the demo-able form of RankForge. Three options for hosting it.

---

## Option 1 — Streamlit Cloud (recommended, free, zero infra)

Streamlit Community Cloud hosts public apps from GitHub for free. About **3 minutes** of clicking once the repo is set up.

### Prereqs
- Repo must be public on GitHub. ✓ ([avszoom/RankForge](https://github.com/avszoom/RankForge))
- Model artifacts must be in the repo. ✓ (committed in this phase: `models/bm25.pkl`, `models/faiss.index`, etc.)
- `requirements.txt` at the repo root. ✓
- Entry-point script. ✓ — `app/streamlit_app.py`

### Steps
1. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) with your GitHub account.
2. Click **New app**.
3. Pick the repo (`avszoom/RankForge`), branch (`main`), and main-file path:
   ```
   app/streamlit_app.py
   ```
4. *(Optional)* set a custom subdomain like `rankforge`.
5. Click **Deploy**.

First boot takes ~2 min:
- ~30 s container start
- ~60 s `pip install` (sentence-transformers + torch are the heavy items)
- ~30 s downloading the cross-encoder model from Hugging Face (~80 MB)
- ~10 s initial Python imports

After that, queries take ~5 ms (BM25/FAISS only) up to ~1.5 s (with cross-encoder).

### Updating
Push to `main` → Streamlit Cloud auto-redeploys within a minute. No CLI needed.

### Resource limits (free tier)
- 1 GB RAM (we use ~600 MB at peak — fits)
- 1 CPU core
- Sleeps after inactivity (cold-start on next visit ~30 s)

---

## Option 2 — Run locally

Quickest way to try the app on your machine.

```bash
git clone https://github.com/avszoom/RankForge.git
cd RankForge

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) put your OPENAI_API_KEY in .env if you want to regenerate the corpus
cp .env.example .env

streamlit run app/streamlit_app.py
```

Opens [http://localhost:8501](http://localhost:8501) automatically. The first query loads the cross-encoder (~30 s download from Hugging Face); subsequent queries are fast.

---

## Option 3 — Docker (self-host anywhere)

Containerized version for hosting on Fly.io, Render, Cloud Run, your own VPS, etc.

### Dockerfile

Create `Dockerfile` at the repo root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for faiss + lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Healthcheck so the platform knows when the app is ready.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### Build + run locally

```bash
docker build -t rankforge .
docker run -p 8501:8501 rankforge
```

### Deploy to Fly.io

```bash
fly launch          # answer "N" to "create a postgres database"
fly deploy
```

### Deploy to Render

1. Push the Dockerfile.
2. On Render dashboard: *New → Web Service → connect repo → Runtime: Docker*.
3. Default port `8501`. Free instance type is enough.

### Deploy to Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT/rankforge
gcloud run deploy rankforge --image gcr.io/PROJECT/rankforge --port 8501 --memory 2Gi
```

The 2 GB memory is needed because the cross-encoder model peaks around ~600 MB at inference.

---

## A note on cross-encoder cold-start

Wherever you deploy, the first query that uses the cross-encoder triggers a one-time model download (~80 MB) from Hugging Face. Two ways to make this disappear from the user-facing latency:

1. **Pre-warm on container start.** Add an idempotent `model.predict([("warmup", "warmup")])` inside `load_resources()` in `streamlit_app.py`. Subsequent queries hit the cached model.
2. **Bake the model into the Docker image** to avoid the download:
   ```dockerfile
   RUN python -c "from sentence_transformers import CrossEncoder; \
                  CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   ```
   Adds ~80 MB to the image size in exchange for instant first-query latency.

For the Streamlit Cloud free tier, option 1 is enough.
