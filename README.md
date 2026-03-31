# DS 4320 Project 1: Predicting Competitive US House District Outcomes Using Demographic and Structural Features

**Executive Summary**

This repository contains a fully constructed secondary dataset covering US House of Representatives elections from 2012 through 2022, built using the relational model. The dataset combines district-level electoral returns from the MIT Election Data and Science Lab with demographic estimates from the US Census Bureau American Community Survey, national economic and political context variables, and a suite of derived predictive features, including lagged margins, party flip indicators, and competitiveness scores. The data is structured as five relational tables stored in parquet format and linked by shared district and year keys, totaling over 1 MB of data across 2,256 district-year observations and 45 features. A gradient boosting model trained on the dataset predicts party seat flips with a cross-validated ROC-AUC of 0.96 and a held-out 2022 test AUC of 0.98, demonstrating that demographic and structural features contain a strong predictive signal about competitive House outcomes.

**Name:** Amelia Vasiliu

**NetID:** ega9cw

**DOI:** 10.5281/zenodo.19342839

**Press Release:** [Congressional Races Are Decided by Demographics, Not Polls](https://github.com/ameliavasiliu/Project-1-Relational-Model/blob/main/Press_Release.md)

**Data:** [UVA OneDrive Data Folder](https://myuva-my.sharepoint.com/:f:/r/personal/ega9cw_virginia_edu/Documents/Project-1-Relational-Model?csf=1&web=1&e=O6hjyP)

**Pipeline:** [Link to Pipeline folder in GitHub](https://github.com/ameliavasiliu/Project-1-Relational-Model/tree/main/Pipeline)

**License:** MIT License --  [LICENSE]([LICENSE](https://github.com/ameliavasiliu/Project-1-Relational-Model/blob/main/LICENSE))

---

## Problem Definition

**General Problem:** Predicting election results.

**Refined Specific Problem:** Existing forecasting efforts rely heavily on horse-race polling, which at the US House district level is sparse, expensive, and historically unreliable. No assembled dataset systematically combines Census-derived demographic change, historical electoral returns, incumbency context, and national macroeconomic conditions across all House districts over multiple election cycles. The refined problem is: can a multivariate secondary dataset combining demographic shifts, structural election features, and national context enable more accurate prediction of party seat flips in US House districts than polling-based approaches?

**Rationale for Refinement**

The move from general election prediction to US House district seat flips is motivated by two converging gaps. National-level forecasting is a mature space, but House races remain comparatively underserved despite determining which party controls Congress. More importantly, demographic change is one of the most structurally meaningful but analytically underused signals available. Shifts in a district's age composition, educational attainment, racial makeup, and economic conditions between Census cycles have shown strong relationships with partisan swing in the political science literature, yet most forecasting models treat demographics as a static control rather than a dynamic predictor. Combining American Community Survey data with historical electoral returns, incumbency context, and national macroeconomic indicators creates a longitudinal dataset that treats demographic change as a first-class signal. The party flip target was selected over continuous margin prediction because flips are the discrete outcome that determines which party controls the chamber, making them the most practically relevant unit of prediction for campaigns, journalists, and researchers.

**Motivation**

Congressional elections shape nearly every dimension of American policy, yet the analytical infrastructure for predicting their outcomes at the district level remains underdeveloped. The over-reliance on sparse polling has produced well-documented forecasting failures, and campaigns, journalists, and citizens alike struggle to distinguish genuinely competitive races from those that only appear close. This project is motivated by the belief that a more structurally grounded dataset can do better, not by replacing polling, but by contextualizing it within the demographic and behavioral trends that explain why a district is moving in the first place. Competitive races attract more attention, more turnout, and more genuine representation, so improving how we identify them has value beyond prediction alone. Building the right data infrastructure is the first step toward forecasting that is not only more accurate, but more honest about what it knows and does not know.

**Press Release Headline**

[Congressional Races Are Decided by Demographics, Not Polls; A New Dataset Is Built Around That Fact](#press-release)

---

## Domain Exposition

**Terminology**

| Term | Definition |
|------|------------|
| Congressional district | One of 435 geographic divisions of the US, each electing one representative to the House every two years |
| Competitive district | A district won by less than 10 percentage points, where the outcome is genuinely uncertain |
| Partisan swing | The shift in a party's vote share from one election cycle to the next in a given district |
| Margin of victory | The difference in vote share between the winning and losing candidate |
| Party flip | A change in which party wins a district from one cycle to the next |
| Generic ballot | A national poll asking which party voters prefer for Congress, used as a national environment indicator |
| Lagged margin | The margin of victory in a district from the prior election cycle, one of the strongest predictors of future outcomes |
| ACS | American Community Survey, an annual Census Bureau survey providing district-level demographic estimates |
| CVAP | Citizen Voting Age Population, the Census-derived count of citizens aged 18 and older used as the turnout denominator |
| MIT MEDSL | MIT Election Data and Science Lab, which maintains publicly available district-level electoral returns |
| Redistricting | The redrawing of congressional district boundaries after each decennial Census |
| Incumbency effect | The structural advantage held by a sitting representative seeking re-election |
| Two-party vote share | Vote share calculated using only Democratic and Republican votes as the denominator |
| NCHS urban-rural code | A six-level classification from the National Center for Health Statistics, ranging from large central metro (1) to noncore rural (6) |
| Gradient Boosting | A machine learning ensemble method that iteratively builds decision trees to minimize prediction error |
| ROC-AUC | Area Under the Receiver Operating Characteristic Curve, a summary metric for binary classifier performance ranging from 0.5 (random) to 1.0 (perfect) |

**Domain Overview**

This project sits at the intersection of political science, demography, and data science, three fields that have converged steadily as large administrative datasets and computational methods have become more accessible to researchers. The core domain is electoral forecasting, which uses observable signals to estimate likely election outcomes before votes are cast. What makes House district forecasting a particularly interesting problem is that it operates at a scale where national patterns and local demographic realities both matter, and where the tension between them is often where predictive value lives.

The demographic dimension draws from applied demography, which studies how population characteristics change over time and how those changes shape behavior. In electoral contexts, research has shown that shifts in educational attainment, age composition, and racial makeup within a geography tend to precede and predict shifts in partisan preference, sometimes by a full election cycle or more. The data science dimension involves constructing and analyzing a longitudinal, multivariate dataset assembled from several public sources, which raises methodological questions around missingness, redistricting-induced discontinuities, and the treatment of incumbency as a confounding variable. This project takes both dimensions seriously and sits comfortably at their intersection.

**Background Reading**

Link to folder: https://drive.google.com/drive/folders/1S7Gke4rbAwfpAnQpiY-gqzIhO4VVoOz9

| Title | Description | Link |
|-------|-------------|------|
| How 538's 2024 House Election Forecast Works | Explains how 538 builds its House model, including how it handles the near-total absence of district-level polling by relying on structural fundamentals like partisan lean and incumbency | https://drive.google.com/file/d/1_XpWw4vkTwosIME5htjxZggmaChgts-N/view?usp=sharing |
| Changing Partisan Coalitions in a Politically Divided Nation | Pew Research report documenting how the demographic composition of each party's coalition has shifted across race, education, age, and geography since 1996 | https://drive.google.com/file/d/1mUd1RupApQUo0GbbUSNzs4fcOBR5xdBe/view?usp=sharing |
| Forecasting Turnout | Academic paper evaluating the predictive power of leading turnout models at the congressional district level, comparing registration rates, demographics, and early vote data | 	https://drive.google.com/file/d/1kror4_0mEFwmRzF0uHpHHHNFebaZPstN/view?usp=sharing |
| Using Machine Learning to Predict US House of Representatives Elections | Predicts House outcomes using partisan lean, economic conditions, and past results, achieving under 2 points RMSE, a useful methodological baseline that does not use longitudinal demographic change as a primary signal | https://drive.google.com/file/d/1Uy3AJAHUk3AsD7tx2Uf67Vy5zEpSfvnX/view?usp=sharing |
| Politics, Demographic Shifts, and Legal Battles Shape a New Era of Redistricting | UVA Karsh Institute piece on how redistricting in an era of polarization makes even small boundary changes consequential, and how demographic shifts interact with map-drawing to alter district competitiveness | 	https://drive.google.com/file/d/17Yvs9GGpYACilhzEhBr63owwRUFK7gGV/view?usp=sharing |

---

## Data Creation

**Provenance**

This dataset was constructed by combining three publicly available secondary sources.

**Source 1: MIT Election Data and Science Lab (MIT MEDSL).** The House election returns dataset is hosted on Harvard Dataverse and contains candidate-level vote totals for every US House race from 1976 through 2024. The dataset was downloaded manually from the Dataverse repository as a CSV file and filtered programmatically to general elections from 2012 through 2022, excluding runoffs, special elections, write-in candidates, and third-party candidates. Each row in the raw file represents a single candidate, so the data was aggregated to the race level to compute two-party vote shares, margin of victory, and a competitiveness flag for districts won by less than 10 percentage points. The filtered file was stored locally as `election_returns_clean.csv` and uploaded to the project OneDrive data folder.

**Source 2: US Census Bureau American Community Survey (ACS) 5-Year Estimates.** District-level demographic variables were pulled programmatically for each election year from 2012 through 2022 at the congressional district level using the Census Bureau public API. Variables pulled in the initial merge include educational attainment, racial composition, median household income, and total population. The enrichment stage added citizen voting age population, poverty rate, unemployment rate, median age, median home value, owner occupancy rate, and health insurance coverage rate. Raw counts were converted to proportional measures to enable comparison across districts of different population sizes.

**Source 3: Manually compiled national context variables.** National generic congressional ballot margins from RealClearPolitics final averages, presidential popular vote margins from FEC certified results, presidential approval ratings from Gallup annual averages, national unemployment rates from the Bureau of Labor Statistics, and GDP growth rates from the Bureau of Economic Analysis were compiled by hand for each election year from 2012 through 2022 and embedded in the enrichment pipeline script.

The two primary sources were merged on a shared `district_id` key constructed from state postal abbreviation and zero-padded district number, producing a combined longitudinal dataset covering all House districts across six election cycles. The dataset was further enriched with lagged features, derived competitiveness metrics, and urban-rural classification before being decomposed into five relational tables.

### Code Table

| File | Description | Link |
|------|-------------|------|
| `01_clean_election_returns.py` | Loads raw MIT MEDSL House returns CSV, filters to general elections 2012-2022, aggregates from candidate level to race level, computes two-party vote shares and margin of victory, flags competitive districts, outputs `election_returns_clean.csv` | https://github.com/ameliavasiliu/Project-1-Relational-Model/blob/main/code/01_clean_election_returns.py |
| `02_pull_acs_data.py` | Pulls congressional district-level demographic estimates from the Census Bureau ACS 5-year API for 2012-2022, computes derived variables including pct_college, pct_nonwhite, and pct_hispanic, outputs `acs_demographics_clean.csv` | https://github.com/ameliavasiliu/Project-1-Relational-Model/blob/main/code/02_pull_acs_data.py |
| `03_merge_datasets.py` | Merges election returns and ACS demographics on shared `district_id` and `year` keys, drops unmatched rows and missing state identifiers, outputs `combined_dataset.csv` | https://github.com/ameliavasiliu/Project-1-Relational-Model/blob/main/code/03_merge_datasets.py |
| `04_CVAP_enrich.py` | Adds CVAP from Census API, derives incumbency flags and open seat indicators, appends generic ballot margins and redistricting indicators, outputs `combined_dataset_enriched.csv` | https://github.com/ameliavasiliu/Project-1-Relational-Model/blob/main/code/04_CVAP_enrich.py |
| `05_margins_&_party_lines.py` | Adds lagged margin, party flip indicator, Democratic vote swing, competitiveness score, presidential context, national economic indicators, urban-rural classification, and additional ACS variables including poverty rate, unemployment rate, median age, home value, and health insurance rate; outputs `combined_dataset_master.csv` and `combined_dataset_master.parquet` | https://github.com/ameliavasiliu/Project-1-Relational-Model/blob/main/code/05_margins_%26_party_lines.py |
| `06_pipeline.ipynb` | Loads master parquet into DuckDB, creates five relational tables, runs six SQL analytical queries, trains a gradient boosting classifier to predict party flips, writes predictions back to the database, and produces five publication-quality figures | https://github.com/ameliavasiliu/Project-1-Relational-Model/blob/main/code/06_pipeline.py |

**Bias Identification**

Several sources of bias are present in this dataset. The MIT election returns only include Democratic and Republican candidates, meaning districts where a third-party candidate received a significant share of the vote will have inflated two-party vote shares that do not accurately reflect the true margin. Uncontested races were dropped entirely since margin of victory cannot be computed when only one major party fielded a candidate, which means the dataset systematically underrepresents the most non-competitive districts and may skew summary statistics on competitiveness. The ACS 5-year estimates smooth out rapid demographic changes by averaging data across five years, potentially misrepresenting a district's actual composition in any single election year. Redistricting following the 2020 Census means that district boundaries changed significantly between 2022 and earlier cycles, making longitudinal comparisons across that boundary potentially misleading since the same district ID may refer to a meaningfully different geographic area. The national context variables were compiled by hand from published averages, introducing a small risk of transcription error.

**Bias Mitigation**

The most direct mitigation for the uncontested race problem is to retain those races with a flag rather than dropping them, allowing analysts to filter or account for them explicitly rather than silently excluding them. For two-party vote share inflation from third-party candidates, a total vote share check can flag races where the combined Democratic and Republican share falls below 90 percent. The ACS smoothing bias can be partially mitigated by using the estimate year most closely preceding the election and by flagging districts that underwent boundary changes using the `redistricting_year` indicator variable already present in the dataset. For longitudinal analyses spanning the 2020 redistricting, analysts should restrict the sample to a single map era or use a redistricting-adjusted baseline such as Cook PVI to anchor comparisons across cycles. The hand-compiled national context variables were cross-checked against multiple published sources to reduce transcription error.

**Rationale for Critical Decisions**

The decision to define competitive districts as those won by less than 10 percentage points balances analytical breadth with practical relevance. A tighter threshold of 5 points would produce a cleaner set of genuinely contested races but would exclude districts that were competitive in one cycle and safe in another, reducing the longitudinal variation needed to study demographic change over time. Ten points is consistent with definitions used in the forecasting literature and captures enough districts per cycle to support modeling.

The decision to filter to general elections only and exclude runoffs, special elections, and third-party candidates maximizes comparability across cycles. Runoffs and special elections occur under different turnout conditions and political contexts than regular general elections, and including them would introduce variation unrelated to the structural signals the dataset is designed to capture.

The choice to use ACS 5-year estimates rather than 1-year estimates was driven by data availability. One-year estimates are not available for smaller congressional districts below a population threshold, which would introduce systematic missingness for rural and low-population districts that are disproportionately Republican-leaning and would bias the sample. Five-year estimates are available for all districts in all years.

The gradient boosting model was chosen over logistic regression because it handles non-linear interactions between features, such as the interaction between incumbency status and national environment in wave elections, without requiring manual feature engineering. The 2022 election year was held out as a test set rather than using random train-test splitting to avoid data leakage across time, since lagged features create temporal dependencies between rows.

---

## Metadata

**Schema -- ER Diagram (Logical Level)**

<img width="738" height="723" alt="Screenshot 2026-03-30 at 9 15 19 PM" src="https://github.com/user-attachments/assets/c0033421-8a10-4835-ab37-2a4012963a69" />


**Data Table**

| Table | Description | Link |
|-------|-------------|------|
| `districts.parquet` | One row per congressional district (435 rows). Stores static geographic identity and urban-rural classification. Primary key: `district_id` | https://myuva-my.sharepoint.com/:u:/r/personal/ega9cw_virginia_edu/Documents/Project-1-Relational-Model/data/tables/districts.parquet?csf=1&web=1&e=zxOGo7 |
| `elections.parquet` | One row per district per election year (2,256 rows). Stores vote outcomes, incumbency flags, lagged features, and party flip indicator. Primary key: (`district_id`, `year`) | https://myuva-my.sharepoint.com/:u:/r/personal/ega9cw_virginia_edu/Documents/Project-1-Relational-Model/data/tables/elections.parquet?csf=1&web=1&e=CXfD3a |
| `demographics.parquet` | One row per district per election year (2,256 rows). Stores ACS socioeconomic features. Primary key: (`district_id`, `year`) | https://myuva-my.sharepoint.com/:u:/r/personal/ega9cw_virginia_edu/Documents/Project-1-Relational-Model/data/tables/demographics.parquet?csf=1&web=1&e=4ecaH2 |
| `national_context.parquet` | One row per election year (6 rows). Stores national macroeconomic and political environment variables. Primary key: `year` | https://myuva-my.sharepoint.com/:u:/r/personal/ega9cw_virginia_edu/Documents/Project-1-Relational-Model/data/tables/national_context.parquet?csf=1&web=1&e=dEsnOy |
| `results_model.parquet` | One row per district per election year with model-predicted flip probabilities. Primary key: (`district_id`, `year`) | https://myuva-my.sharepoint.com/:u:/r/personal/ega9cw_virginia_edu/Documents/Project-1-Relational-Model/data/tables/results_model.parquet?csf=1&web=1&e=vum4qy |

**Data Dictionary**

*districts*

| Name | Data Type | Description | Example |
|------|-----------|-------------|---------|
| district_id | string | Unique identifier combining state postal abbreviation and zero-padded district number | AL-01 |
| state | string | Full state name in uppercase | ALABAMA |
| state_po | string | Two-letter state postal abbreviation | AL |
| district | int | Congressional district number within the state | 1 |
| urban_rural_code | int | NCHS 6-level urban-rural classification: 1 = large central metro, 6 = noncore rural | 4 |
| urban_rural_label | string | Text label corresponding to urban_rural_code | Small metro |

*elections*

| Name | Data Type | Description | Example |
|------|-----------|-------------|---------|
| district_id | string | Foreign key linking to districts | AL-01 |
| year | int | Election year | 2018 |
| dem_share | float | Democratic two-party vote share as a proportion | 0.453 |
| rep_share | float | Republican two-party vote share as a proportion | 0.547 |
| margin | float | Republican minus Democratic vote share; positive values indicate Republican win | 0.094 |
| winner | string | Party of the winning candidate | R |
| margin_abs | float | Absolute value of margin used to assess competitiveness regardless of winning party | 0.094 |
| competitive | bool | True if the district was won by less than 10 percentage points | True |
| competitiveness_score | float | One minus margin_abs; higher values indicate more competitive races | 0.906 |
| incumbent_running | bool | True if the prior cycle winner's party held the seat going into this election | True |
| open_seat | bool | True if the seat changed party from the prior cycle | False |
| prev_margin | float | Margin of victory in this district from the prior election cycle | 0.121 |
| prev_winner | string | Winning party in this district from the prior election cycle | R |
| dem_share_lag | float | Democratic vote share in this district from the prior election cycle | 0.440 |
| party_flip | bool | True if the winning party changed from the prior cycle | False |
| dem_swing | float | Change in Democratic vote share from the prior cycle | 0.013 |
| redistricting_year | bool | True if this cycle used maps redrawn after a decennial Census | False |

*demographics*

| Name | Data Type | Description | Example |
|------|-----------|-------------|---------|
| district_id | string | Foreign key linking to districts | AL-01 |
| year | int | Election year | 2018 |
| total_population | int | Total district population from ACS 5-year estimates | 706503 |
| cvap_total | int | Citizen voting age population | 543485 |
| median_income | float | Median household income in nominal US dollars | 47952 |
| median_age | float | Median age of the district population | 39.6 |
| median_home_value | float | Median home value in US dollars | 142300 |
| pct_college | float | Proportion of adults 25 and older with a bachelor's degree or higher | 0.243 |
| pct_hs_only | float | Proportion of adults 25 and older whose highest credential is a high school diploma or GED | 0.330 |
| pct_white | float | Proportion of total population identifying as white alone | 0.671 |
| pct_black | float | Proportion of total population identifying as Black or African American alone | 0.276 |
| pct_asian | float | Proportion of total population identifying as Asian alone | 0.014 |
| pct_hispanic | float | Proportion of total population identifying as Hispanic or Latino | 0.032 |
| pct_nonwhite | float | Proportion not identifying as white alone, computed as 1 minus pct_white | 0.329 |
| poverty_rate | float | Proportion of population below the federal poverty level | 0.174 |
| district_unemployment_rate | float | Civilian unemployment rate from ACS | 0.064 |
| owner_occ_rate | float | Proportion of occupied housing units that are owner-occupied | 0.681 |
| health_ins_rate | float | Proportion of population with health insurance coverage | 0.477 |

*national_context*

| Name | Data Type | Description | Example |
|------|-----------|-------------|---------|
| year | int | Election year, primary key | 2018 |
| generic_ballot_margin | float | National Democratic minus Republican generic ballot final average in percentage points; positive values favor Democrats | 8.6 |
| pres_popular_margin | float | Most recent presidential popular vote margin in percentage points | 2.1 |
| pres_year | float | Year of the most recent presidential election | 2016 |
| natl_unemployment_rate | float | National annual average unemployment rate in percent | 3.9 |
| natl_gdp_growth | float | National annual real GDP growth rate in percent | 3.0 |
| pres_approval | float | Presidential annual average approval rating in percent | 42.0 |

*results_model*

| Name | Data Type | Description | Example |
|------|-----------|-------------|---------|
| district_id | string | Foreign key linking to districts | AZ-01 |
| year | int | Election year | 2018 |
| predicted_flip | bool | Whether the model predicts this seat will flip party | True |
| predicted_competitive | bool | True if flip probability exceeds 0.15 | True |
| flip_probability | float | Model estimated probability of a party flip | 0.72 |
| model_name | string | Name of the model used to generate predictions | GradientBoostingClassifier |

**Data Dictionary -- Uncertainty Quantification for Numerical Features**

All statistics are computed from the full dataset (n = 2,256 district-year observations) unless noted otherwise.

*elections table*

| Feature | Mean | Std Dev | Min | Max | Missing | Uncertainty Notes |
|---------|------|---------|-----|-----|---------|-------------------|
| dem_share | 0.492 | 0.165 | 0.128 | 0.940 | 0 | Two-party conversion overstates both shares when a third party receives meaningful votes. In races where combined D and R share falls below 0.90, the estimated inflation in dem_share is approximately 0.04, based on the historical average third-party share in contested races |
| rep_share | 0.483 | 0.168 | 0.020 | 0.843 | 0 | Mathematically dependent on dem_share; the standard deviation of the sum (dem_share + rep_share) across the dataset is 0.014, reflecting small but non-trivial third-party presence in some races |
| margin | -0.009 | 0.330 | -0.905 | 0.715 | 0 | Derived from dem_share and rep_share; inherits both sources of measurement error. In the approximately 30 races in this dataset decided by less than 0.005, recount outcomes have historically altered certified results, introducing outcome-level uncertainty |
| margin_abs | 0.281 | 0.174 | 0.000 | 0.905 | 0 | Standard deviation is identical to margin at 0.174; loses directionality which may matter for some analyses |
| competitiveness_score | 0.719 | 0.174 | 0.095 | 1.000 | 0 | Derived as 1 minus margin_abs; no additional measurement uncertainty beyond margin |
| prev_margin | 0.002 | 0.319 | -0.905 | 0.684 | 581 | 581 missing values correspond to districts appearing for the first time in the sample. Standard deviation of 0.319 is slightly narrower than current-cycle margin std dev of 0.330, consistent with mean reversion in competitive races |
| dem_share_lag | 0.486 | 0.160 | 0.154 | 0.940 | 581 | Same 581 missing values as prev_margin; inherits dem_share uncertainty from prior cycle |
| dem_swing | -0.001 | 0.065 | -0.440 | 0.324 | 581 | Standard deviation of 0.065 reflects genuine district-level variation but is inflated for cycles crossing the 2020 redistricting boundary, where the same district_id may refer to a different geographic area. The redistricting component of swing uncertainty cannot be quantified from this data alone |

*demographics table*

| Feature | Mean | Std Dev | Min | Max | Missing | Uncertainty Notes |
|---------|------|---------|-----|-----|---------|-------------------|
| total_population | 735,276 | 41,453 | 523,851 | 941,210 | 0 | ACS 5-year published margins of error at the congressional district level are typically plus or minus 1 to 3 percent of the point estimate. The std dev of 41,453 reflects genuine cross-district population variation rather than measurement error |
| cvap_total | 566,756 | 36,549 | 413,396 | 721,226 | 0 | ACS published margin of error is typically plus or minus 0.5 to 1.5 percent of the CVAP estimate. Additional uncertainty comes from citizenship self-reporting, which may undercount non-citizen residents who misidentify as citizens |
| median_income | 63,322 | 18,897 | 25,351 | 168,712 | 0 | ACS income margins of error at the district level are typically plus or minus 2 to 5 percent of the point estimate. Expressed in nominal dollars, cumulative CPI inflation from 2012 to 2022 was approximately 30 percent, so cross-year comparisons require adjustment |
| median_age | 38.26 | 3.61 | 27.4 | 55.7 | 0 | ACS published margin of error for median age at the congressional district level is typically plus or minus 0.3 to 0.8 years. The std dev of 3.61 reflects genuine demographic variation across districts |
| median_home_value | 257,880 | 175,384 | 58,900 | 1,458,600 | 0 | Very high std dev of 175,384 reflects substantial cross-district variation. ACS margin of error can exceed plus or minus 10,000 dollars in rapidly appreciating high-cost markets; top-coded values in high-cost districts may compress the true upper tail |
| pct_college | 0.310 | 0.106 | 0.080 | 0.793 | 0 | Denominator is adults 25 and older rather than total population; sensitive to age composition differences across districts. ACS margin of error is typically plus or minus 0.5 to 1.5 percentage points at this geography |
| pct_hs_only | 0.274 | 0.065 | 0.069 | 0.473 | 0 | Same denominator limitation as pct_college. Does not capture adults with some college but no degree, which nationally represents approximately 20 percent of the adult population |
| pct_white | 0.724 | 0.170 | 0.135 | 0.967 | 0 | Based on self-reported race; multiracial respondents are counted as white alone only if white was their sole race selection. ACS margin of error is typically plus or minus 0.5 to 1.0 percentage points |
| pct_black | 0.121 | 0.134 | 0.004 | 0.667 | 0 | Same self-reporting limitation as pct_white. High std dev of 0.134 reflects substantial geographic concentration of the Black population across districts |
| pct_asian | 0.053 | 0.066 | 0.003 | 0.567 | 0 | Same self-reporting limitation as pct_white. ACS estimates for smaller subgroups carry proportionally larger margins of error relative to the point estimate |
| pct_hispanic | 0.170 | 0.173 | 0.008 | 0.911 | 0 | Denominator is total population rather than adults 25 and older, making this not directly comparable to the education proportions in the same table. High std dev of 0.173 reflects geographic concentration |
| pct_nonwhite | 0.276 | 0.170 | 0.033 | 0.865 | 0 | Derived as 1 minus pct_white; inherits all self-reporting uncertainty from that variable and does not distinguish among non-white demographic groups |
| poverty_rate | 0.140 | 0.051 | 0.039 | 0.395 | 0 | ACS poverty estimates have higher margins of error in more homogeneous districts. The federal poverty threshold is a fixed income level not adjusted for the regional cost of living, introducing geographic comparability concerns |
| district_unemployment_rate | 0.070 | 0.026 | 0.024 | 0.215 | 0 | ACS-based annual average rather than a point-in-time rate at election day. ACS margin of error at this geography is typically plus or minus 0.5 to 1.0 percentage points. Std dev of 0.026 reflects genuine variation across districts |
| owner_occ_rate | 0.649 | 0.104 | 0.091 | 0.841 | 0 | ACS housing tenure estimates are generally reliable; margin of error is typically plus or minus 1 to 2 percentage points at the district level |
| health_ins_rate | 0.489 | 0.008 | 0.448 | 0.522 | 0 | Notably low std dev of 0.008 across all 2,256 observations indicates limited cross-district variation as measured by the ACS. The ACS health insurance question wording changed slightly across survey years, introducing minor comparability concerns for trend analysis |

*national_context table*

| Feature | Mean | Std Dev | Min | Max | Missing | Uncertainty Notes |
|---------|------|---------|-----|-----|---------|-------------------|
| generic_ballot_margin | 0.96 | 5.13 | -5.7 | 8.6 | 0 | Based on RealClearPolitics final poll averages; methodological differences across pollsters and systematic house effects introduce measurement uncertainty of approximately plus or minus 1 to 2 percentage points. The std dev of 5.13 across the six election years reflects genuine year-to-year national environment variation |
| pres_popular_margin | 3.52 | 1.02 | 2.1 | 4.5 | 0 | Certified FEC results; measurement uncertainty is negligible. The std dev of 1.02 reflects variation across the three presidential cycles represented in this dataset |
| natl_unemployment_rate | 5.78 | 1.87 | 3.6 | 8.1 | 0 | BLS annual average from monthly Current Population Survey estimates; national-level standard error is less than 0.1 percentage points. The std dev of 1.87 across years reflects genuine economic cycle variation, including the 8.1 percent rate during the 2020 COVID contraction |
| natl_gdp_growth | 1.31 | 2.22 | -3.4 | 3.0 | 0 | BEA advance estimates are typically revised by less than 0.5 percentage points in subsequent releases. The std dev of 2.22 reflects genuine year-to-year variation, including the -3.4 percent contraction in 2020 |
| pres_approval | 44.03 | 3.41 | 40.4 | 49.1 | 0 | Gallup annual average based on approximately 350 daily tracking polls per year; margin of error is approximately plus or minus 1 percentage point. The std dev of 3.41 across election years reflects genuine variation in presidential popularity over the 2012 to 2022 period |
