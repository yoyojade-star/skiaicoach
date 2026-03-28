# SkiAI Coach — Android (Kotlin + Jetpack Compose)

Minimal client for the SkiAI FastAPI backend: pick a video, `POST /upload`, poll `GET /result/{job_id}`, play `GET /uploads/{job_id}_analyzed.mp4` with ExoPlayer.

## Open in Android Studio

1. Install [Android Studio](https://developer.android.com/studio) (Ladybug or newer recommended).
2. **File → Open** and select this `android` folder (not the repo root).
3. Let Gradle sync finish; if the wrapper is missing, use **File → Settings → Build → Gradle** and ensure Gradle uses the wrapper, or run **File → Sync Project with Gradle Files**.

## Run

1. Start the Python API (`python main.py` on port **8001**).
2. **Android Emulator**: default base URL is `http://10.0.2.2:8001` (already set in `BuildConfig.DEFAULT_BASE_URL`). The emulator maps `10.0.2.2` to your host machine.
3. **Physical device**: use your PC’s LAN IP, e.g. `http://192.168.1.50:8001`, and ensure the phone and PC are on the same Wi‑Fi; the backend must listen on `0.0.0.0` (as `uvicorn` does in `main.py`).

Cleartext HTTP is allowed only for `10.0.2.2`, `localhost`, and `127.0.0.1` via `network_security_config.xml`. For production, serve the API over **HTTPS** and tighten the network config.

## Agent backend

This app targets the **standard** backend (`main.py`). The agent API (`mainagent.py`) expects the same upload shape plus optional `agent_skills`; extending the client is straightforward (add another multipart field in `SkiApiService` + `MultipartBody.Part`).
