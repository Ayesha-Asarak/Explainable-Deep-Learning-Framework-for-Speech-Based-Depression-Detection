# Project layout (clean)

## Keep at root — deployed system

| Item | Role |
|------|------|
| `server.py` | FastAPI web server |
| `start_server.sh` | Helper to start server |
| `train_official_acoustic.py` | Train deployed Random Forest |
| `requirements.txt` | Python dependencies |
| `src/` | Core code (data, features, predict, explain, …) |
| `frontend/` | Web UI |
| `models/` | Saved models (`.pkl`, metadata) |
| `depressed/` · `non depressed/` | DAIC-WOZ audio |
| `patient_records/` | Saved analyses |
| `*_AVEC2017.csv` · `full_test_split.csv` | Official labels/splits |

**Run**
```bash
python3 train_official_acoustic.py
python3 -m uvicorn server:app --host 127.0.0.1 --port 8765
```

## Organized folders

| Folder | Contents |
|--------|----------|
| `docs/01_Writing_and_Viva/` | Thesis markdown, viva guide, interview notes |
| `docs/02_Thesis_Figures/` | Architecture / flowchart PNG+PDF, UI techniques map |
| `docs/03_Thesis_PDFs_and_Word/` | Full thesis PDF/DOCX drafts |
| `experiments/01_Extra_Training_Scripts/` | CNN/SSL/other train scripts (not deployed) |
| `experiments/02_Legacy_Apps/` | Old Streamlit `app.py` |
| `archive/01_Text_Extracts/` | Extracted text from old PDFs |
| `archive/02_Tools/` | One-off helpers (`generate_pdf.py`) |

## Process flow

Audio → participant speech → segments → 23 features → aggregate → RF → label → explanations → Web UI
