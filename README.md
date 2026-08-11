# Procurement Optimizer Web App

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Features
- Upload supplier Excel/CSV/PDF buying manuals
- Normalize supplier catalogs using the supplied ingestion module
- Enter required demand
- Run the supplied MILP solver
- View supplier-level purchasing results
- Download the optimized basket as CSV
