# Solar Dashboard Router

This project now includes a local web app that routes between separate notebook-backed dashboards for ETL and PVK.

## What the app does

- Opens a sidebar selector for ETL or PVK
- Keeps the ETL dashboard logic in its own module
- Keeps the PVK dashboard logic in its own module
- Loads data from `Data_file.xlsx` or an uploaded `.xlsx`
- Lets users set model/test/input values from controls instead of editing notebook cells
- Shows metrics, point predictions, full curves, and actual-vs-predicted comparisons for each dashboard

## Run locally

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start app:

```bash
streamlit run app.py
```

4. Open the local URL shown by Streamlit (usually `http://localhost:8501`) and use the sidebar to switch between ETL and PVK.

## Notes

- Default data source is `Data_file.xlsx` in the project root.
- The router uses the ETL dashboard as the template pattern and keeps PVK code separate in its own module.
- If your file or sheet name differs, upload a matching file for the dashboard you want to use.

## Deploying to Render (recommended)

This repository includes a `Dockerfile` so you can deploy easily to Render as a Web Service.

Steps:

1. Commit and push your repository to GitHub.

2. Create a new Web Service on Render and connect your GitHub repo.

3. Choose "Docker" as the environment (Render will detect the `Dockerfile`).

4. Build and deploy. Render sets the `$PORT` environment variable automatically; the included `Dockerfile` uses it so Streamlit will bind to the correct port.

Local test commands (Docker):

```bash
docker build -t solar-dashboard:latest .
docker run -p 8501:8501 -e PORT=8501 solar-dashboard:latest
```

Alternative (without Docker): in Render's Web Service settings you can set the start command to:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

If your app reads local Excel files, either commit them (not recommended for large/private data) or configure an object storage / secrets and update paths in the code.

