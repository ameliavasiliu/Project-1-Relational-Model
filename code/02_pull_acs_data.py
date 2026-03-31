import logging
import sys
import requests
import pandas as pd
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("02_pull_acs_data.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Test keyless Census API access before pulling all years
try:
    test_url = "https://api.census.gov/data/2022/acs/acs5?get=NAME,B15003_001E,B15003_022E,B15003_023E,B15003_024E,B15003_025E&for=congressional%20district:*"
    response = requests.get(test_url, timeout=20)
    print(response.status_code)
    print(response.text[:500])
except Exception as e:
    log.error(f"Census API test request failed: {e}")
    sys.exit(1)

# ACS 5-year estimates available at congressional district level for these years
years = [2012, 2014, 2016, 2018, 2020, 2022]

# Variables to pull:
# Educational attainment (B15003)
# B15003_001E = total population 25+
# B15003_022E = bachelor's degree
# B15003_023E = master's degree
# B15003_024E = professional degree
# B15003_025E = doctorate degree
# B15003_017E = high school diploma
# B15003_018E = GED

# Age (B01001)
# B01001_001E = total population

# Race (B02001)
# B02001_001E = total
# B02001_002E = white alone
# B02001_003E = black alone
# B02001_005E = asian alone

# Hispanic (B03001)
# B03001_003E = hispanic or latino

# Population (B01003)
# B01003_001E = total population

# Median household income (B19013)
# B19013_001E = median household income

VARIABLES = [
    "B15003_001E",  # total pop 25+ (education denominator)
    "B15003_017E",  # high school diploma
    "B15003_018E",  # GED
    "B15003_022E",  # bachelor's
    "B15003_023E",  # master's
    "B15003_024E",  # professional degree
    "B15003_025E",  # doctorate
    "B01001_001E",  # total population
    "B02001_001E",  # race total
    "B02001_002E",  # white alone
    "B02001_003E",  # black alone
    "B02001_005E",  # asian alone
    "B03001_003E",  # hispanic or latino
    "B19013_001E",  # median household income
]

def pull_acs_year(year, variables):
    vars_str = ",".join(["NAME"] + variables)
    url = f"https://api.census.gov/data/{year}/acs/acs5?get={vars_str}&for=congressional%20district:*"
    try:
        response = requests.get(url, timeout=20)
    except Exception as e:
        log.error(f"Request failed for {year}: {e}")
        return None
    if response.status_code != 200:
        print(f"Failed for {year}: {response.status_code}")
        return None
    try:
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df['year'] = year
        print(f"Pulled {year}: {df.shape[0]} districts")
        return df
    except Exception as e:
        log.error(f"Failed to parse response for {year}: {e}")
        return None

# Pull all years
all_years = []
for year in years:
    df_year = pull_acs_year(year, VARIABLES)
    if df_year is not None:
        all_years.append(df_year)
    time.sleep(1)  # be polite to the API

if not all_years:
    log.error("No ACS data was successfully pulled -- exiting")
    sys.exit(1)

# Combine
acs_raw = pd.concat(all_years, ignore_index=True)
print(f"\nTotal rows: {acs_raw.shape}")
print(acs_raw.head())

# Convert all numeric columns to numeric type
numeric_cols = [
    "B15003_001E", "B15003_017E", "B15003_018E", "B15003_022E",
    "B15003_023E", "B15003_024E", "B15003_025E", "B01001_001E",
    "B02001_001E", "B02001_002E", "B02001_003E", "B02001_005E",
    "B03001_003E", "B19013_001E"
]

for col in numeric_cols:
    acs_raw[col] = pd.to_numeric(acs_raw[col], errors='coerce')

# Replace negative values with NaN (Census uses -666666666 for missing)
acs_raw[numeric_cols] = acs_raw[numeric_cols].where(acs_raw[numeric_cols] >= 0, np.nan)

# Compute derived variables
acs_clean = acs_raw.copy()

# Education
acs_clean['pct_college'] = (
    acs_clean['B15003_022E'] + acs_clean['B15003_023E'] +
    acs_clean['B15003_024E'] + acs_clean['B15003_025E']
) / acs_clean['B15003_001E']

acs_clean['pct_hs_only'] = (
    acs_clean['B15003_017E'] + acs_clean['B15003_018E']
) / acs_clean['B15003_001E']

# Race and ethnicity
acs_clean['pct_white'] = acs_clean['B02001_002E'] / acs_clean['B02001_001E']
acs_clean['pct_black'] = acs_clean['B02001_003E'] / acs_clean['B02001_001E']
acs_clean['pct_asian'] = acs_clean['B02001_005E'] / acs_clean['B02001_001E']
acs_clean['pct_hispanic'] = acs_clean['B03001_003E'] / acs_clean['B01001_001E']
acs_clean['pct_nonwhite'] = 1 - acs_clean['pct_white']

# Income
acs_clean['median_income'] = acs_clean['B19013_001E']

# Total population
acs_clean['total_population'] = acs_clean['B01001_001E']

# Create district ID to match election returns
acs_clean['state_fips'] = acs_clean['state'].astype(str).str.zfill(2)
acs_clean['district_num'] = acs_clean['congressional district'].astype(str).str.zfill(2)

# Map state fips to state abbreviation
fips_to_abbr = {
    '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT',
    '10':'DE','11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL',
    '18':'IN','19':'IA','20':'KS','21':'KY','22':'LA','23':'ME','24':'MD',
    '25':'MA','26':'MI','27':'MN','28':'MS','29':'MO','30':'MT','31':'NE',
    '32':'NV','33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND',
    '39':'OH','40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD',
    '47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA','54':'WV',
    '55':'WI','56':'WY'
}

acs_clean['state_po'] = acs_clean['state_fips'].map(fips_to_abbr)
acs_clean['district_id'] = acs_clean['state_po'] + '-' + acs_clean['district_num']

# Keep only the columns we need
acs_final = acs_clean[[
    'year', 'state_po', 'district_id',
    'total_population', 'median_income',
    'pct_college', 'pct_hs_only',
    'pct_white', 'pct_black', 'pct_asian', 'pct_hispanic', 'pct_nonwhite'
]].copy()

# Inspect
print(acs_final.shape)
print(f"\nMissing values:")
print(acs_final.isnull().sum())
print(f"\nSample rows:")
print(acs_final.head())
print(f"\nYear counts:")
print(acs_final.groupby('year').size())

# Save
try:
    acs_final.to_csv("acs_demographics_clean.csv", index=False)
    log.info("Saved to acs_demographics_clean.csv")
    print("\nSaved to acs_demographics_clean.csv")
except Exception as e:
    log.error(f"Failed to save output: {e}")
    sys.exit(1)
