# Project-1-Relational-Model
# House Elections Dataset

## Overview
A longitudinal dataset combining US House election returns and 
ACS demographic data for congressional districts from 2012 to 2022.

## Data Sources
- MIT Election Data and Science Lab:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
- American Community Survey 5-Year Estimates (Census Bureau API):
  https://api.census.gov

## Repository Structure
- `data/raw/` — original downloaded source files
- `data/processed/` — cleaned and merged output files
- `code/` — scripts to reproduce the dataset in order

## To Reproduce
1. Download raw MIT data into `data/raw/`
2. Run `code/01_clean_election_returns.py`
3. Run `code/02_pull_acs_data.py`
4. Run `code/03_merge_datasets.py`
5. Final dataset will be saved to `data/processed/combined_dataset.csv`

## Dataset Description
| File | Rows | Description |
|---|---|---|
| election_returns_clean.csv | 2,681 | House election results 2012-2024 by district |
| acs_demographics_clean.csv | 2,625 | ACS demographic estimates 2012-2022 by district |
| combined_dataset.csv | TBD | Merged dataset on district_id and year |
