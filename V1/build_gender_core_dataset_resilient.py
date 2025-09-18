# build_gender_core_dataset_resilient.py
# Robust pipeline: left-join base population, ensure full matrix, optional split of "Both" counts.
# Put your raw CSVs in data/raw/
# Files (default names; edit if different):
#  - data/raw/population_state.csv
#  - data/raw/birth_sex_ethnic_state.csv
#  - data/raw/death_sex_ethnic_state.csv
# Output:
#  - data/processed/gender_core_dataset_full.csv

import pandas as pd
from pathlib import Path
import sys
import math

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POP_F = DATA_DIR / "population_state.csv"
BIRTH_F = DATA_DIR / "birth_sex_ethnic_state.csv"
DEATH_F = DATA_DIR / "death_sex_ethnic_state.csv"

# CONFIG: choose states, years, sexes and splitting behavior
STATES = ['Selangor','Johor','Penang','Perak','Kelantan','Sabah','Sarawak','Kuala Lumpur']
YEARS = list(range(2000, 2020))   # inclusive 2000-2019
SEXES = ['Male','Female']         # target sexes to guarantee (Both may also exist)
SPLIT_BOTH = True                 # if True, split 'Both' counts into Male/Female by population proportions
FALLBACK_SEXRATIO_M_PER_100_F = 105.0  # fallback sex ratio if population proportion unavailable

# ----------------- helpers -----------------
def safe_read_csv(fp):
    if not fp.exists():
        print(f"ERROR: file not found: {fp}")
        sys.exit(1)
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.read_csv(fp, encoding='latin1')

def normalize_state(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s_low = s.lower()
    s_low = s_low.replace('wilayah ', '').replace('wp ', '').replace('w.p. ', '').replace('w.p ', '')
    s_low = s_low.replace('pulau ', '')
    return s_low.title()

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

def detect_count_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first numeric column not 'year'
    nums = df.select_dtypes(include='number').columns.tolist()
    nums = [c for c in nums if 'year' not in c.lower()]
    return nums[0] if nums else None

# ----------------- load population (base) -----------------
print("Loading population:", POP_F)
pop = safe_read_csv(POP_F)
print("Population columns:", pop.columns.tolist())

# create year
if 'date' in pop.columns:
    pop['date'] = pd.to_datetime(pop['date'], errors='coerce')
    pop['year'] = pop['date'].dt.year
elif 'year' in pop.columns:
    pop['year'] = pd.to_numeric(pop['year'], errors='coerce')
else:
    # try to find year-like col
    for c in pop.columns:
        if 'year' in c.lower():
            pop['year'] = pd.to_numeric(pop[c], errors='coerce')
            break

# population count detection
pop_count_col = detect_count_col(pop, ['population','value','pop','jumlah','jumlah_population','jumlah_pop'])
if pop_count_col is None:
    raise KeyError("Could not find population count column. Columns: " + ", ".join(pop.columns))
print("Using population count column:", pop_count_col)

# normalize
pop['state'] = pop['state'].astype(str).apply(normalize_state)
pop['sex'] = pop['sex'].astype(str).apply(normalize_sex)
pop['population'] = pd.to_numeric(pop[pop_count_col], errors='coerce')

# aggregate to overall age/eth if duplicates
pop_agg = pop.groupby(['state','year','sex'], as_index=False)['population'].sum()

# ----------------- load births -----------------
print("\nLoading births:", BIRTH_F)
births = safe_read_csv(BIRTH_F)
print("Births columns:", births.columns.tolist())

# year
if 'date' in births.columns:
    births['date'] = pd.to_datetime(births['date'], errors='coerce')
    births['year'] = births['date'].dt.year
elif 'year' in births.columns:
    births['year'] = pd.to_numeric(births['year'], errors='coerce')

birth_count_col = detect_count_col(births, ['live_births','Live Births','abs','value','jumlah','kelahiran','live births'])
if birth_count_col is None:
    raise KeyError("Could not find births count column. Columns: " + ", ".join(births.columns))
print("Using births count column:", birth_count_col)

births['state'] = births['state'].astype(str).apply(normalize_state)
births['sex'] = births['sex'].astype(str).apply(normalize_sex)
births['count_raw'] = pd.to_numeric(births[birth_count_col], errors='coerce')

# ----------------- load deaths -----------------
print("\nLoading deaths:", DEATH_F)
deaths = safe_read_csv(DEATH_F)
print("Deaths columns:", deaths.columns.tolist())

if 'date' in deaths.columns:
    deaths['date'] = pd.to_datetime(deaths['date'], errors='coerce')
    deaths['year'] = deaths['date'].dt.year
elif 'year' in deaths.columns:
    deaths['year'] = pd.to_numeric(deaths['year'], errors='coerce')

death_count_col = detect_count_col(deaths, ['deaths','Deaths','abs','value','jumlah','kematian'])
if death_count_col is None:
    raise KeyError("Could not find deaths count column. Columns: " + ", ".join(deaths.columns))
print("Using deaths count column:", death_count_col)

deaths['state'] = deaths['state'].astype(str).apply(normalize_state)
deaths['sex'] = deaths['sex'].astype(str).apply(normalize_sex)
deaths['count_raw'] = pd.to_numeric(deaths[death_count_col], errors='coerce')

# ----------------- prepare full grid from population base -----------------
# Base grid: ensure population has all states/years/sexes we want
base_grid = pd.MultiIndex.from_product([STATES, YEARS, SEXES], names=['state','year','sex']).to_frame(index=False)
base_grid['state'] = base_grid['state'].astype(str).apply(normalize_state)
base_grid['sex'] = base_grid['sex'].astype(str).apply(normalize_sex)

# Merge population into base grid using left join on state-year-sex from pop_agg
df = base_grid.merge(pop_agg, on=['state','year','sex'], how='left')

# If population missing for some combos, attempt to fill from 'Both' or leave NaN
# Build quick pivot of population for proportions if needed
pop_pivot = pop_agg.pivot_table(index=['state','year'], columns='sex', values='population').reset_index()

# ----------------- function to allocate "Both" counts -----------------
def allocate_both(df_counts, kind='births'):
    """
    df_counts: DataFrame with columns state, year, sex, count_raw
    kind: for messages only
    Returns: DataFrame with columns state, year, sex, count (Male/Female)
    """
    # separate Both rows and Male/Female rows
    df_counts = df_counts[['state','year','sex','count_raw']].copy()
    df_counts['sex'] = df_counts['sex'].astype(str)
    both = df_counts[df_counts['sex']=='Both'].copy()
    mf = df_counts[df_counts['sex'].isin(['Male','Female'])].copy()
    # aggregate existing male/female counts
    mf_agg = mf.groupby(['state','year','sex'], as_index=False)['count_raw'].sum()
    # prepare allocation results
    alloc_rows = []
    # iterate both rows and allocate
    for idx, r in both.iterrows():
        s, y, total = r['state'], r['year'], r['count_raw']
        # find population proportions for same state-year
        try:
            p = pop_pivot[(pop_pivot['state']==s) & (pop_pivot['year']==y)]
            if not p.empty and not (pd.isna(p.get('Male').iloc[0]) or pd.isna(p.get('Female').iloc[0])):
                male_pop = p.get('Male').iloc[0] or 0
                female_pop = p.get('Female').iloc[0] or 0
                if male_pop+female_pop > 0:
                    pm = male_pop/(male_pop+female_pop)
                    pf = female_pop/(male_pop+female_pop)
                else:
                    pm = 0.5125  # fallback from fallback ratio 105/205
                    pf = 1 - pm
            else:
                # fallback from global sex ratio
                pm = FALLBACK_SEXRATIO_M_PER_100_F / (FALLBACK_SEXRATIO_M_PER_100_F + 100)
                pf = 1 - pm
        except Exception:
            pm = FALLBACK_SEXRATIO_M_PER_100_F / (FALLBACK_SEXRATIO_M_PER_100_F + 100)
            pf = 1 - pm
        male_alloc = int(round(total * pm))
        female_alloc = int(total - male_alloc)  # preserve total
        alloc_rows.append({'state': s, 'year': y, 'sex': 'Male', 'count_raw': male_alloc})
        alloc_rows.append({'state': s, 'year': y, 'sex': 'Female', 'count_raw': female_alloc})
    # combine existing mf_agg with alloc_rows
    alloc_df = pd.concat([mf_agg, pd.DataFrame(alloc_rows)], ignore_index=True, sort=False)
    # now aggregate so multiple entries for same state-year-sex sum up
    alloc_df = alloc_df.groupby(['state','year','sex'], as_index=False)['count_raw'].sum()
    return alloc_df

# ----------------- process births: allocate 'Both' or keep -----------------
births_subset = births[['state','year','sex','count_raw']].copy()
# group as necessary
births_subset = births_subset.groupby(['state','year','sex'], as_index=False)['count_raw'].sum()

if SPLIT_BOTH:
    births_alloc = allocate_both(births_subset, kind='births')
else:
    births_alloc = births_subset.copy()

# same for deaths
deaths_subset = deaths[['state','year','sex','count_raw']].copy()
deaths_subset = deaths_subset.groupby(['state','year','sex'], as_index=False)['count_raw'].sum()

if SPLIT_BOTH:
    deaths_alloc = allocate_both(deaths_subset, kind='deaths')
else:
    deaths_alloc = deaths_subset.copy()

# ----------------- merge counts into df (left join so we keep all base rows) -----------------
df = df.merge(births_alloc.rename(columns={'count_raw':'live_births'}), on=['state','year','sex'], how='left')
df = df.merge(deaths_alloc.rename(columns={'count_raw':'deaths'}), on=['state','year','sex'], how='left')

# fill NaNs with 0 for counts (you can choose to leave NaNs instead)
df['live_births'] = df['live_births'].fillna(0).astype(int)
df['deaths'] = df['deaths'].fillna(0).astype(int)

# fill population NaNs with 0 (or leave NaN if you prefer)
df['population'] = df['population'].fillna(0)

# derived metrics
df['births_per_1000'] = df.apply(lambda r: (r['live_births'] / r['population'] * 1000) if r['population']>0 else None, axis=1)
df['deaths_per_1000'] = df.apply(lambda r: (r['deaths'] / r['population'] * 1000) if r['population']>0 else None, axis=1)
df['natural_increase'] = df['live_births'] - df['deaths']

# ----------------- QC & logging -----------------
expected_rows = len(STATES) * len(YEARS) * len(SEXES)
actual_rows = len(df)
print("\nExpected base rows (states*years*sexes):", expected_rows)
print("Actual rows produced:", actual_rows)
if actual_rows < 300:
    print("Warning: actual rows < 300. Consider adding more states/years or keeping 'Both' sex rows.")

# find missing population entries
missing_pop = df[df['population']==0].shape[0]
print("Rows with missing population (population==0):", missing_pop)
# show some missing combos
missing_combos = df[(df['population']==0) | (df['population'].isna())][['state','year','sex']].drop_duplicates().head(20)
if not missing_combos.empty:
    print("Sample missing population combos (first 20):")
    print(missing_combos.to_string(index=False))

# Save a CSV and also a log file with counts per state-year-sex
out_fp = OUT_DIR / "gender_core_dataset_full.csv"
df.to_csv(out_fp, index=False, encoding='utf-8-sig')
print("\nSaved merged dataset to:", out_fp)

# Save a short diagnostics CSV
diag = df.groupby(['state','year']).agg(
    pop_total = ('population','sum'),
    births_total = ('live_births','sum'),
    deaths_total = ('deaths','sum'),
    rows = ('sex','count')
).reset_index()
diag.to_csv(OUT_DIR / "merge_diagnostics_by_state_year.csv", index=False)
print("Saved diagnostics by state-year.")
