# DataXFormer (Web Tables)

Replication-oriented reimplementation of the **web table transformation discovery pipeline** from  
*DataXFormer: A Robust Transformation Discovery System* (Abedjan et al.).

Developed as part of the **Large Scale Data Integration (LSDI)** course at **TU Berlin**.

---

## Overview

This project replicates the **web table component** of the DataXFormer architecture. The system learns data transformation rules (e.g., mappings, normalization, structural changes) from example pairs and applies them to unseen data.

**Scope:**
- ✅ Web table transformation discovery (indexing, querying, joining, ranking)
- ✅ Large-scale optimizations
- ❌ Web form discovery
- ❌ Crowd / HIT implementation
- ❌ Complementary subsystems

The focus is on **reproducibility, modularity, and systematic experimentation**.

---

## Tech Stack

- **Language:** Python ≥ 3.10
- **Libraries:** `pandas`, `numpy`, `torch`, `nltk`, `tabulate`, `orjson`
- **Database:** `vertica_python`, `python-dotenv`
- **Frontend:** `streamlit`

Dependencies are managed via `pyproject.toml`.

---

## Environment Setup

The system requires access to a **Vertica database**.

Create a `.env` file in the project root:

```env
VERTICA_HOST=xxx
VERTICA_PORT=xxx
VERTICA_USER=xxx
VERTICA_PASSWORD=xxx
VERTICA_DATABASE=xxx
```

The database schema is expected to follow the web table setup described in the paper. Table and column names can be configured via `VerticaConfig` in `src/config.py`.

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd DataXFormer

# Using uv (recommended)
uv sync

# Or via pip
pip install -r requirements.txt
```

---

## Usage

### Streamlit Demo
```bash
streamlit run src/Frontend.py
```

### Pipeline Execution (No UI)
```bash
python main.py
```

---

## Project Structure

```text
DataXFormer/
├── main.py                 # Pipeline entry point
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   ├── Frontend.py         # Streamlit UI
│   ├── config.py           # Configuration (VerticaConfig)
│   ├── pipeline.py
│   ├── database/
│   │   ├── vertica_client.py
│   │   ├── query_factory.py
│   │   └── direct_FD.py
│   └── web_tables/
│       ├── indexing.py
│       ├── querying.py
│       ├── joining.py
│       └── ranking.py
├── tests/
├── test_notebooks/
└── fd_verifier_cache/
```

---

## Notes & Troubleshooting

- Ensure `.env` exists before execution.
- Verify database schema compatibility.
- Check Python version if dependency issues occur.

---

## Authors

Group project developed as part of the LSDI course at TU Berlin.

**Contributors:**
- Christoph Halberstadt
- Lucas Reinken

---

## License

Intended for academic and educational use only.  
Licensing follows the requirements of the LSDI course at TU Berlin.

---

## Reference

Abedjan et al.  
*DataXFormer: A Robust Transformation Discovery System*
```
