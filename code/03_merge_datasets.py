import logging
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("03_merge_datasets.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Load both cleaned datasets
try:
    election = pd.read_csv("election_returns_clean.csv")
    log.info(f"Election returns loaded: {election.shape}")
except FileNotFoundError:
    log.error("File not found: election_returns_clean.csv -- run 01_clean_election_returns.py first")
    sys.exit(1)
except Exception as e:
    log.error(f"Failed to load election returns: {e}")
    sys.exit(1)

try:
    acs = pd.read_csv("acs_demographics_clean.csv")
    log.info(f"ACS demographics loaded: {acs.shape}")
except FileNotFoundError:
    log.error("File not found: acs_demographics_clean.csv -- run 02_pull_acs_data.py first")
    sys.exit(1)
except Exception as e:
    log.error(f"Failed to load ACS demographics: {e}")
    sys.exit(1)

# Drop rows with missing state_po (DC and territories)
acs = acs.dropna(subset=['state_po', 'district_id'])

# ACS 5-year estimates lag by ~2 years so we align years as follows:
# ACS 2012 -> Election 2012
# ACS 2014 -> Election 2014
# ACS 2016 -> Election 2016
# ACS 2018 -> Election 2018
# ACS 2020 -> Election 2020
# ACS 2022 -> Election 2022
# Note: 2024 election will not have a matching ACS year yet

# Merge on district_id and year
try:
    combined = pd.merge(
        election,
        acs,
        on=['year', 'district_id'],
        how='inner'
    )
    log.info(f"Combined shape after inner merge: {combined.shape}")
except Exception as e:
    log.error(f"Merge failed: {e}")
    sys.exit(1)

# Drop duplicate state column from ACS
combined = combined.drop(columns=['state_po_y'])
combined = combined.rename(columns={'state_po_x': 'state_po'})

# Inspect
print(combined.shape)
print(f"\nMissing values:")
print(combined.isnull().sum())
print(f"\nRows per year:")
print(combined.groupby('year').size())
print(f"\nCompetitive districts per year:")
print(combined.groupby('year')['competitive'].sum())
print(f"\nSample rows:")
print(combined.head())

# Save
try:
    combined.to_csv("combined_dataset.csv", index=False)
    log.info("Saved to combined_dataset.csv")
    print("\nSaved to combined_dataset.csv")
except Exception as e:
    log.error(f"Failed to save output: {e}")
    sys.exit(1)
