import pandas as pd

population = pd.read_csv("data/processed/population_core.csv")

births = pd.read_csv("data/raw/birth_sex_ethnic_state.csv")
births['sex'] = births['sex'].str.strip().str.lower()
births['ethnicity'] = births['ethnicity'].str.strip().str.lower()
births = births[(births['sex'] == 'both') & (births['ethnicity'] == 'overall')]
births['year'] = pd.to_datetime(births['date']).dt.year
births = births[['state','year','abs']].rename(columns={'abs':'live_births'})

deaths = pd.read_csv("data/raw/death_sex_ethnic_state.csv")
deaths['sex'] = deaths['sex'].str.strip().str.lower()
deaths['ethnicity'] = deaths['ethnicity'].str.strip().str.lower()
deaths = deaths[(deaths['sex'] == 'both') & (deaths['ethnicity'] == 'overall')]
deaths['year'] = pd.to_datetime(deaths['date']).dt.year
deaths = deaths[['state','year','abs']].rename(columns={'abs':'deaths'})

merged = population.merge(births, on=['state','year'], how='left')
merged = merged.merge(deaths, on=['state','year'], how='left')

merged['birth_rate'] = merged['live_births'] / merged['population'] * 1000
merged['death_rate'] = merged['deaths'] / merged['population'] * 1000
merged['natural_increase'] = merged['live_births'] - merged['deaths']

output_path = "data/processed/population_births_deaths.csv"
merged.to_csv(output_path, index=False)

print("Final dataset saved to:", output_path)
print("Shape:", merged.shape)
print(merged.head(10))
