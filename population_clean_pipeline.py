# population_clean_pipeline.py
# Stage 1: Clean and aggregate population_state.csv
# Produces a clean, aggregated CSV suitable as a base for modular analysis
# Author: [Your Name]
# Date: 2025-09-17

import pandas as pd
from pathlib import Path
import sys

# -------------------- Paths --------------------
DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POP_FILE = DATA_DIR / "population_state.csv"
OUT_FILE = OUT_DIR / "population_clean.csv"

# -------------------- Config --------------------
# Target states and years (adjustable)
TARGET_STATES = ['Selangor','Johor','Penang','Perak','Kelantan','Sabah','Sarawak','Kuala Lumpur']
TARGET_YEARS = list(range(2000, 2020))  # inclusive
AGG_COLS = ['state','year','sex']       # columns to group by
FILL_ZERO = True                         # fill missing population with 0

# -------------------- Helpers --------------------
def safe_read_csv(fp):
    if not fp.exists():
        print(f"ERROR: file not found: {fp}")
        sys.exit(1)
    try:
        return pd.read_csv(fp)
    except UnicodeDecodeError:
        return pd.read_csv(fp, encoding="latin1")

def normalize_state(s):
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    for prefix in ['wilayah ','wp ','w.p. ','w.p ','pulau ']:
        s = s.replace(prefix,'')
    return s.title()

def normalize_sex(x):
    if pd.isna(x):
        return x
    x = str(x).strip().lower()
    if x in ['male','m','lelaki','l']:
        return 'Male'
    if x in ['female','f','perempuan','p']:
        return 'Female'
    if 'both' in x or 'total' in x:
        return 'Both'
    return x.title()

def detect_population_col(df):
    candidates = ['population','pop','value','jumlah','jumlah_population','jumlah_pop']
    for c in candidates:
        if c in df.columns:
            return c
    nums = df.select_dtypes(include='number').columns.tolist()
    nums = [c for c in nums if 'year' not in c.lower()]
    if nums:
        return nums[0]
    return None

# -------------------- Load --------------------
print("Loading population data:", POP_FILE)
pop = safe_read_csv(POP_FILE)
print("Columns detected:", pop.columns.tolist())

# -------------------- Preprocess --------------------
# Detect year
if 'date' in pop.columns:
    pop['date'] = pd.to_datetime(pop['date'], errors='coerce')
    pop['year'] = pop['date'].dt.year
elif 'year' in pop.columns:
    pop['year'] = pd.to_numeric(pop['year'], errors='coerce')
else:
    # fallback: first column with 4-digit integers
    for c in pop.columns:
        if pop[c].astype(str).str.match(r'^\d{4}$').any():
            pop['year'] = pd.to_numeric(pop[c], errors='coerce')
            break
    else:
        raise KeyError("No year column found in population CSV.")

# Detect population count
pop_count_col = detect_population_col(pop)
if pop_count_col is None:
    raise KeyError("Population count column not found. Columns: " + ", ".join(pop.columns))
print("Using population column:", pop_count_col)

# Normalize state and sex
pop['state'] = pop['state'].astype(str).apply(normalize_state)
pop['sex'] = pop['sex'].astype(str).apply(normalize_sex)
pop['population'] = pd.to_numeric(pop[pop_count_col], errors='coerce')

# -------------------- Filter --------------------
pop = pop[pop['state'].isin(TARGET_STATES) & pop['year'].isin(TARGET_YEARS)]

# -------------------- Aggregate --------------------
pop_agg = pop.groupby(AGG_COLS, as_index=False)['population'].sum()

# -------------------- Ensure full grid --------------------
# Fill missing combinations
full_index = pd.MultiIndex.from_product([TARGET_STATES, TARGET_YEARS, ['Male','Female']], names=AGG_COLS)
pop_full = pd.DataFrame(index=full_index).reset_index()
pop_full = pop_full.merge(pop_agg, on=AGG_COLS, how='left')

if FILL_ZERO:
    pop_full['population'] = pop_full['population'].fillna(0)

# -------------------- Output --------------------
pop_full.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
print("Cleaned population dataset saved to:", OUT_FILE)
print("Total rows:", len(pop_full))
