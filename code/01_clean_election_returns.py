import logging
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("01_clean_election_returns.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Load raw MIT MEDSL house returns file
try:
    df = pd.read_csv("1976-2022-house.csv")
except FileNotFoundError:
    log.error("File not found: 1976-2022-house.csv -- make sure it is in the working directory")
    sys.exit(1)
except Exception as e:
    log.error(f"Failed to load CSV: {e}")
    sys.exit(1)

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df['year'].value_counts().sort_index())

# Inspect key filter columns before applying filters
print("Stage values:")
print(df['stage'].value_counts())

print("\nRunoff values:")
print(df['runoff'].value_counts())

print("\nSpecial values:")
print(df['special'].value_counts())

print("\nParty values (top 20):")
print(df['party'].value_counts().head(20))

# Check specifically what 2012 looks like unfiltered
df_2012 = df[df['year'] == 2012]
print("\n2012 sample:")
print(df_2012[['year', 'stage', 'runoff', 'special', 'party', 'candidate']].head(10))

# Filter to 2012-2024 general elections only
# Exclude runoffs, special elections, and write-ins
# Handle NaN in runoff column with fillna
df_filtered = df[
    (df['year'] >= 2012) &
    (df['stage'] == 'GEN') &
    (df['runoff'].fillna(False) == False) &
    (df['special'] == False) &
    (df['writein'] == False)
]

# Keep only major party candidates
df_filtered = df_filtered[df_filtered['party'].isin(['DEMOCRAT', 'REPUBLICAN'])].copy()
log.info(f"Filtered to {len(df_filtered):,} candidate-level rows")

# Calculate each candidate's share of total votes in their race
df_filtered['vote_share'] = df_filtered['candidatevotes'] / df_filtered['totalvotes']

# Pivot from candidate level to race level (one row per district-year)
try:
    race_level = df_filtered.groupby(
        ['year', 'state', 'state_po', 'district', 'party']
    )['vote_share'].sum().unstack('party').reset_index()
    race_level.columns = ['year', 'state', 'state_po', 'district', 'dem_share', 'rep_share']
    log.info(f"Pivoted to race level: {len(race_level):,} rows")
except Exception as e:
    log.error(f"Failed to pivot to race level: {e}")
    sys.exit(1)

# Compute margin, winner, and competitiveness flag
race_level['margin'] = race_level['rep_share'] - race_level['dem_share']
race_level['winner'] = race_level['margin'].apply(lambda x: 'R' if x > 0 else 'D')
race_level['margin_abs'] = race_level['margin'].abs()

# Competitive = won by less than 10 percentage points
race_level['competitive'] = race_level['margin_abs'] < 0.10

# Fix at-large districts (district 0 -> district 1)
race_level['district'] = race_level['district'].replace(0, 1)

# Create shared district_id key for merging with ACS data
race_level['district_id'] = race_level['state_po'] + '-' + race_level['district'].astype(str).str.zfill(2)

# Drop uncontested races where one party had no candidate
race_level = race_level.dropna(subset=['dem_share', 'rep_share'])
log.info(f"Final shape after dropping uncontested races: {race_level.shape}")

# Inspect final output
print(race_level.shape)
print(f"\nCompetitive districts per year:")
print(race_level.groupby('year')['competitive'].sum())
print(f"\nSample rows:")
print(race_level.head())

# Save
try:
    race_level.to_csv("election_returns_clean.csv", index=False)
    log.info("Saved to election_returns_clean.csv")
    print("\nSaved to election_returns_clean.csv")
except Exception as e:
    log.error(f"Failed to save output: {e}")
    sys.exit(1)
