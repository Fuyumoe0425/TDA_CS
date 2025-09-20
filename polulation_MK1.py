import pandas as pd


population = pd.read_csv("data/raw/population_state.csv")


population['sex'] = population['sex'].str.strip().str.lower()
population['age'] = population['age'].str.strip().str.lower()
population['ethnicity'] = population['ethnicity'].str.strip().str.lower()


pop_core = population[
    (population['sex'] == 'both') &
    (population['age'] == 'overall') &
    (population['ethnicity'] == 'overall')
]


pop_core['year'] = pd.to_datetime(pop_core['date']).dt.year


YEARS = list(range(2000, 2020))   
pop_core = pop_core[pop_core['year'].isin(YEARS)]

pop_core = pop_core[['state','year','population']]

output_path = "data/processed/population_core.csv"
pop_core.to_csv(output_path, index=False)

print("Final dataset shape:", pop_core.shape)
print(pop_core.head())

