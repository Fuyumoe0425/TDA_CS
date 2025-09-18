# build_gender_core_dataset_stage2.py
# Stage 2: Merge births and deaths into cleaned population dataset
# Input: population_clean.csv, birth_sex_ethnic_state.csv, death_sex_ethnic_state.csv
# Output: gender_core_dataset.csv

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")  # cleaned population already here
RAW_DIR = Path("data/raw")         # births/deaths raw files
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POP_FILE = DATA_DIR / "population_clean.csv"
BIRTH_FILE = RAW_DIR / "birth_sex_ethnic_state.csv"
DEATH_FILE = RAW_DIR / "death_sex_ethnic_state.csv"
OUT_FILE = OUT_DIR / "gender_core_dataset.csv"

STATES = ['Selangor','Johor','Penang','Perak','Kelantan','Sabah','Sarawak','Kuala Lumpur']
YEARS = list(range(2000, 2020))
SEXES = ['Male','Female']
FALLBACK_SEXRATIO_M_PER_100_F = 105.0  # used if population proportions not available

# helper functions
def safe_read_csv(fp):
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found")
    try:
        return pd.read_csv(fp)
    except:
        return pd.read_csv(fp, encoding='latin1')

def normalize_state(s):
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = s.replace('wilayah ','').replace('wp ','').replace('w.p. ','').replace('w.p ','').replace('pulau ','')
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

def detect_numeric_col(df):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    return numeric_cols[0] if numeric_cols else None

def allocate_both(df_counts, pop_df, fallback_ratio=FALLBACK_SEXRATIO_M_PER_100_F):
    both = df_counts[df_counts['sex']=='Both'].copy()
    mf = df_counts[df_counts['sex'].isin(['Male','Female'])].copy()
    mf_agg = mf.groupby(['state','year','sex'], as_index=False)['count'].sum()
    alloc_rows = []

    pop_pivot = pop_df.pivot_table(index=['state','year'], columns='sex', values='population').reset_index()

    for _, row in both.iterrows():
        s, y, total = row['state'], row['year'], row['count']
        pop_row = pop_pivot[(pop_pivot['state']==s) & (pop_pivot['year']==y)]
        if not pop_row.empty and not pd.isna(pop_row.get('Male').iloc[0]) and not pd.isna(pop_row.get('Female').iloc[0]):
            male_pop = pop_row['Male'].iloc[0]
            female_pop = pop_row['Female'].iloc[0]
            if male_pop + female_pop > 0:
                pm = male_pop / (male_pop + female_pop)
            else:
                pm = fallback_ratio / (fallback_ratio + 100)
        else:
            pm = fallback_ratio / (fallback_ratio + 100)
        male_count = int(round(total * pm))
        female_count = total - male_count
        alloc_rows.append({'state': s, 'year': y, 'sex': 'Male', 'count': male_count})
        alloc_rows.append({'state': s, 'year': y, 'sex': 'Female', 'count': female_count})

    alloc_df = pd.concat([mf_agg, pd.DataFrame(alloc_rows)], ignore_index=True)
    alloc_df = alloc_df.groupby(['state','year','sex'], as_index=False)['count'].sum()
    return alloc_df

# load base population
pop = safe_read_csv(POP_FILE)
pop['state'] = pop['state'].apply(normalize_state)
pop['sex'] = pop['sex'].apply(normalize_sex)
pop = pop[pop['state'].isin(STATES) & pop['year'].isin(YEARS) & pop['sex'].isin(SEXES)]
pop_agg = pop.groupby(['state','year','sex'], as_index=False)['population'].sum()

# load births
births = safe_read_csv(BIRTH_FILE)
if 'date' in births.columns:
    births['year'] = pd.to_datetime(births['date'], errors='coerce').dt.year
births['state'] = births['state'].apply(normalize_state)
births['sex'] = births['sex'].apply(normalize_sex)
birth_count_col = detect_numeric_col(births)
births['count'] = pd.to_numeric(births[birth_count_col], errors='coerce')
births_agg = births.groupby(['state','year','sex'], as_index=False)['count'].sum()
births_final = allocate_both(births_agg, pop_agg)

# load deaths
deaths = safe_read_csv(DEATH_FILE)
if 'date' in deaths.columns:
    deaths['year'] = pd.to_datetime(deaths['date'], errors='coerce').dt.year
deaths['state'] = deaths['state'].apply(normalize_state)
deaths['sex'] = deaths['sex'].apply(normalize_sex)
death_count_col = detect_numeric_col(deaths)
deaths['count'] = pd.to_numeric(deaths[death_count_col], errors='coerce')
deaths_agg = deaths.groupby(['state','year','sex'], as_index=False)['count'].sum()
deaths_final = allocate_both(deaths_agg, pop_agg)

# prepare full grid
full_grid = pd.MultiIndex.from_product([STATES, YEARS, SEXES], names=['state','year','sex']).to_frame(index=False)
df = full_grid.merge(pop_agg, on=['state','year','sex'], how='left')
df = df.merge(births_final.rename(columns={'count':'live_births'}), on=['state','year','sex'], how='left')
df = df.merge(deaths_final.rename(columns={'count':'deaths'}), on=['state','year','sex'], how='left')

# fill NaNs
df['population'] = df['population'].fillna(0)
df['live_births'] = df['live_births'].fillna(0).astype(int)
df['deaths'] = df['deaths'].fillna(0).astype(int)

# derived metrics
df['births_per_1000'] = df.apply(lambda r: (r['live_births']/r['population']*1000) if r['population']>0 else None, axis=1)
df['deaths_per_1000'] = df.apply(lambda r: (r['deaths']/r['population']*1000) if r['population']>0 else None, axis=1)
df['natural_increase'] = df['live_births'] - df['deaths']

# save final dataset
df.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
print("Saved final gender_core_dataset.csv with", len(df), "rows")

