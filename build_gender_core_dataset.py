# build_gender_core_dataset_final.py
# Final, robust script to build gender_core_dataset.csv (ready for Power BI)
# Put your raw CSVs in data/raw/
# Files expected (names can be adjusted below):
#   - data/raw/population_state.csv
#   - data/raw/birth_sex_ethnic_state.csv
#   - data/raw/death_sex_ethnic_state.csv
#   - data/raw/primary_students.csv

import pandas as pd
from pathlib import Path
import sys

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# file names (adjust if your files use different filenames)
POP_F = DATA_DIR / "population_state.csv"
BIRTH_F = DATA_DIR / "birth_sex_ethnic_state.csv"
DEATH_F = DATA_DIR / "death_sex_ethnic_state.csv"
STUD_F = DATA_DIR / "primary_students.csv"

# --- helper functions ------------------------------------------------------
def safe_read_csv(fp):
    if not fp.exists():
        print(f"ERROR: file not found: {fp}")
        sys.exit(1)
    # try a couple of encodings
    try:
        return pd.read_csv(fp)
    except UnicodeDecodeError:
        return pd.read_csv(fp, encoding="latin1")

def normalize_state(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s_low = s.lower()
    # simple fixes
    s_low = s_low.replace('wilayah ', '').replace('wp ', '')
    s_low = s_low.replace('w.p. ', '').replace('w.p ', '')
    s_low = s_low.replace('pulau ', '')
    # common names to title case
    return s_low.title()

def normalize_sex(x):
    if pd.isna(x):
        return x
    x = str(x).strip().lower()
    if x in ['male','m','lelaki','l']:
        return 'Male'
    if x in ['female','f','perempuan','p']:
        return 'Female'
    if 'both' in x or 'total' in x or x in ['both sexes','both sexes (male & female)','both sexes (male & female)']:
        return 'Both'
    # fallback: title-case whatever remains
    return x.title()

# --- 1) Population ---------------------------------------------------------
print("\nLoading population data:", POP_F)
pop = safe_read_csv(POP_F)
print("Population columns:", pop.columns.tolist())

# attempt to find date/year column
if 'date' in pop.columns:
    pop['date'] = pd.to_datetime(pop['date'], errors='coerce')
    pop['year'] = pop['date'].dt.year
elif 'year' in pop.columns:
    pop['year'] = pd.to_numeric(pop['year'], errors='coerce')
else:
    # try to find any column that looks like year
    for c in pop.columns:
        if 'year' in c.lower():
            pop['year'] = pd.to_numeric(pop[c], errors='coerce')
            break
    else:
        raise KeyError("No date/year column found in population CSV.")

# population count column detection
pop_count_col = None
for candidate in ['population','value','pop','jumlah','jumlah_pop','jumlah_population']:
    if candidate in pop.columns:
        pop_count_col = candidate
        break
# fallback: find numeric column besides year
if pop_count_col is None:
    numeric_cols = pop.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in ['year']]
    if numeric_cols:
        pop_count_col = numeric_cols[0]
if pop_count_col is None:
    raise KeyError("Population count column not found in population CSV. Columns: " + ", ".join(pop.columns))

# normalize columns we will use
pop['state'] = pop['state'].astype(str).apply(normalize_state)
pop['sex'] = pop['sex'].astype(str).apply(normalize_sex)
pop['population'] = pd.to_numeric(pop[pop_count_col], errors='coerce')

print("Using population count column:", pop_count_col)
pop_agg = pop.groupby(['state','year','sex'], as_index=False)['population'].sum()

# --- 2) Births -------------------------------------------------------------
print("\nLoading births data:", BIRTH_F)
births = safe_read_csv(BIRTH_F)
print("Births columns:", births.columns.tolist())

# date/year
if 'date' in births.columns:
    births['date'] = pd.to_datetime(births['date'], errors='coerce')
    births['year'] = births['date'].dt.year
elif 'year' in births.columns:
    births['year'] = pd.to_numeric(births['year'], errors='coerce')
else:
    # try to find any year-like column
    for c in births.columns:
        if 'year' in c.lower():
            births['year'] = pd.to_numeric(births[c], errors='coerce')
            break
# count col detection -- many DOSM files use 'abs'
birth_col = None
for candidate in ['live_births','Live Births','abs','value','jumlah','births','jumlah_kelahiran','kelahiran']:
    if candidate in births.columns:
        birth_col = candidate
        break
if birth_col is None:
    # look for first numeric column (excluding year)
    nums = births.select_dtypes(include='number').columns.tolist()
    nums = [c for c in nums if c.lower() not in ['year']]
    if nums:
        birth_col = nums[0]
if birth_col is None:
    raise KeyError("Births count column not found in births CSV. Columns: " + ", ".join(births.columns))

births['state'] = births['state'].astype(str).apply(normalize_state)
births['sex'] = births['sex'].astype(str).apply(normalize_sex)
births['live_births'] = pd.to_numeric(births[birth_col], errors='coerce')
births_agg = births.groupby(['state','year','sex'], as_index=False)['live_births'].sum()

print("Using births count column:", birth_col)

# --- 3) Deaths -------------------------------------------------------------
print("\nLoading deaths data:", DEATH_F)
deaths = safe_read_csv(DEATH_F)
print("Deaths columns:", deaths.columns.tolist())

if 'date' in deaths.columns:
    deaths['date'] = pd.to_datetime(deaths['date'], errors='coerce')
    deaths['year'] = deaths['date'].dt.year
elif 'year' in deaths.columns:
    deaths['year'] = pd.to_numeric(deaths['year'], errors='coerce')
else:
    for c in deaths.columns:
        if 'year' in c.lower():
            deaths['year'] = pd.to_numeric(deaths[c], errors='coerce')
            break

death_col = None
for candidate in ['deaths','Deaths','abs','value','jumlah','kematian','death']:
    if candidate in deaths.columns:
        death_col = candidate
        break
if death_col is None:
    nums = deaths.select_dtypes(include='number').columns.tolist()
    nums = [c for c in nums if c.lower() not in ['year']]
    if nums:
        death_col = nums[0]
if death_col is None:
    raise KeyError("Deaths count column not found in deaths CSV. Columns: " + ", ".join(deaths.columns))

deaths['state'] = deaths['state'].astype(str).apply(normalize_state)
deaths['sex'] = deaths['sex'].astype(str).apply(normalize_sex)
deaths['deaths'] = pd.to_numeric(deaths[death_col], errors='coerce')
deaths_agg = deaths.groupby(['state','year','sex'], as_index=False)['deaths'].sum()

print("Using deaths count column:", death_col)

# --- 4) Students -----------------------------------------------------------
print("\nLoading students data:", STUD_F)
students = safe_read_csv(STUD_F)
print("Students columns:", students.columns.tolist())

# try to standardize known names
students_cols_map = {}
for c in students.columns:
    low = c.lower()
    if 'year' in low:
        students_cols_map[c] = 'year'
    elif low in ['sex','gender']:
        students_cols_map[c] = 'sex'
    elif 'state' in low:
        students_cols_map[c] = 'state'
    elif 'student' in low or 'number' in low or 'count' in low or 'value' in low:
        students_cols_map[c] = 'students'

students = students.rename(columns=students_cols_map)
# ensure columns exist
if 'year' not in students.columns:
    # try to detect numeric column that looks like year
    for c in students.columns:
        if students[c].astype(str).str.match(r'^\d{4}$').any():
            students.rename(columns={c:'year'}, inplace=True)
            break
if 'students' not in students.columns:
    # if not found, try 'value' as fallback
    if 'value' in students.columns:
        students.rename(columns={'value':'students'}, inplace=True)

# normalize
if 'year' in students.columns:
    students['year'] = pd.to_numeric(students['year'], errors='coerce')
students['state'] = students['state'].astype(str).apply(normalize_state)
if 'sex' in students.columns:
    students['sex'] = students['sex'].astype(str).apply(normalize_sex)

# if students count column exists, convert
if 'students' in students.columns:
    students['students'] = pd.to_numeric(students['students'], errors='coerce')
else:
    # if no students column found, create zero column and warn
    print("Warning: no students count column found. 'students' will be set to 0.")
    students['students'] = 0
    students['sex'] = students.get('sex', 'Both')  # ensure column exists

students_agg = students.groupby(['state','year','sex'], as_index=False)['students'].sum()

# --- 5) Filtering config ---------------------------------------------------
# default selection (change as you like)
states = ['Selangor','Johor','Penang','Perak','Kelantan','Sabah','Sarawak','Kuala Lumpur']
years = list(range(2000, 2020))  # 2000-2019 inclusive
sexes = ['Male','Female']

pop_sel = pop_agg[
    (pop_agg['state'].isin(states)) &
    (pop_agg['year'].isin(years)) &
    (pop_agg['sex'].isin(sexes))
].copy()

# --- 6) Merge ----------------------------------------------------------------
df = pop_sel.merge(births_agg, on=['state','year','sex'], how='left')
df = df.merge(deaths_agg, on=['state','year','sex'], how='left')
df = df.merge(students_agg, on=['state','year','sex'], how='left')

# replace NaN numerics with 0 (births/deaths/students), preserve population NaNs if any
for col in ['live_births','deaths','students']:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
    else:
        df[col] = 0

# --- 7) Derived metrics ------------------------------------------------------
df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
df['births_per_1000'] = df.apply(lambda r: (r['live_births'] / r['population'] * 1000) if r['population']>0 else None, axis=1)
df['deaths_per_1000'] = df.apply(lambda r: (r['deaths'] / r['population'] * 1000) if r['population']>0 else None, axis=1)

# --- 8) QC prints ------------------------------------------------------------
expected = len(states) * len(years) * len(sexes)
print(f"\nExpected rows: {expected}; actual rows after filtering: {len(df)}")

# report missing merges
missing_births = df['live_births'].isna().sum() if 'live_births' in df.columns else 0
missing_deaths = df['deaths'].isna().sum() if 'deaths' in df.columns else 0
print(f"Rows with zero births (or missing handled as 0): { (df['live_births']==0).sum() }")
print(f"Rows with zero deaths (or missing handled as 0): { (df['deaths']==0).sum() }")

print("\nSample rows:")
print(df.head(10).to_string(index=False))

# --- 9) Save -----------------------------------------------------------------
out_fp = OUT_DIR / "gender_core_dataset.csv"
df.to_csv(out_fp, index=False, encoding='utf-8-sig')
print("\nSaved merged dataset to:", out_fp)


