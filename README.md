# Marketplace Assistant (MVP)

Lightweight decision-support scaffolding for the Marketplace® simulation.

Quick start (using Poetry):

```bash
cd marketplace
poetry install
poetry run python -m marketplace_assistant.cli import-csv --help
poetry run streamlit run marketplace_assistant/dashboard/streamlit_app.py
```

If you prefer pip/venv, create a virtualenv and `pip install -r requirements.txt` (not included).

This repo writes generated Markdown into your Obsidian vault (the `Marketplace/` folder).
