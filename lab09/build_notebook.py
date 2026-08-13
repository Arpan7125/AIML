"""Builds lab09_2547116.ipynb from scratch using nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ---------------------------------------------------------------------------
md(r"""# India Air Quality & Crop Yield — EDA Lab

**Student ID:** 2547116

**Aim:** Check whether states with worse air quality also produce less crop, using two real, messy datasets.

**Files:**
- `city_day.csv` — city-day air quality (PM2.5, PM10, NO2, CO, AQI), 2015-2020.
- `crop_production.csv` — district-crop-year production records, 1997-2015.

Both are raw Kaggle data, not cleaned up for this lab. Covers Lab 1 (Tasks 1-5) and Lab 2 (Tasks 6-9 + optional) in one notebook.""")

md("## 0. Import Libraries")

code(r"""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

print("Libraries loaded.")""")

md("Just the usual EDA stack, nothing fancy needed.")

md("## 0.1 Load the Datasets")

code(r"""air_raw = pd.read_csv("city_day.csv/city_day.csv")
crop_raw = pd.read_csv("crop_production.csv/crop_production.csv")

print("Air quality data:", air_raw.shape)
display(air_raw.head())

print("Crop production data:", crop_raw.shape)
display(crop_raw.head())""")

md("Air data is one row per city per day, crop data is one row per district per crop per year. Different shapes already — merging these later won't be simple.")

# ===========================================================================
md("# Lab 1 — Data Preprocessing & Visualisation")

md("## Task 1: First-Impression Data Profile\n\nBefore trusting this data: how big is it, how much is missing, and what looks broken?")

code(r"""print("AIR QUALITY DATA")
print("Shape:", air_raw.shape)
air_raw.info()""")

code(r"""print("CROP PRODUCTION DATA")
print("Shape:", crop_raw.shape)
crop_raw.info()""")

code(r"""air_missing = air_raw.isna().mean().mul(100).round(1).sort_values(ascending=False)
crop_missing = crop_raw.isna().mean().mul(100).round(1).sort_values(ascending=False)

print("Missing % - air data:")
print(air_missing)
print()
print("Missing % - crop data:")
print(crop_missing)""")

code(r"""print("Exact duplicate rows - air:", air_raw.duplicated().sum())
print("Exact duplicate rows - crop:", crop_raw.duplicated().sum())

print()
print("Cities in air data:", air_raw["City"].nunique())
print("States in crop data:", crop_raw["State_Name"].nunique())
print()
print("Air date range:", air_raw["Date"].min(), "to", air_raw["Date"].max())
print("Crop year range:", crop_raw["Crop_Year"].min(), "to", crop_raw["Crop_Year"].max())
print()
print("AQI summary:")
print(air_raw["AQI"].describe())""")

md(r"""**city_day.csv**

| Aspect | Detail |
|---|---|
| Size | 29,531 rows x 16 cols, one row per city per day |
| Coverage | 26 cities, Jan 2015 - Jul 2020 |
| Missing data | very uneven — some cols ~7-12%, Xylene **61%** |
| Duplicates | none |
| Odd values | AQI hits **2049** somewhere (real scale tops out ~500) |
| Other | no `State` column, only `City` |

**crop_production.csv**

| Aspect | Detail |
|---|---|
| Size | 246,091 rows x 7 cols, one row per district-crop-year |
| Coverage | 33 states/UTs, 1997-2015 |
| Missing data | only `Production`, and just 1.5% |
| Duplicates | none |
| Odd values | state names and every `Season` value have trailing spaces |

**Biggest concern:** the two files barely share a time window — air data starts right when crop data ends, so matching by year won't work well. A couple of pollutant columns are too empty to trust, that AQI of 2049 is clearly bad data, and I'll need to strip whitespace before grouping states or the same state will get split into two groups.""")

# ---------------------------------------------------------------------------
md(r"""## Task 2: Missing Value Treatment

Not treating every column the same way here — dropping everything or filling everything with the mean would be lazy.

**Air data:**
- `Xylene`, `NH3`, `Toluene`, `AQI_Bucket` → **drop the columns**. Not needed for this analysis, and mostly empty anyway.
- `PM2.5`, `PM10`, `NO2`, `CO` → **fill with each city's median**. Pollution varies a lot city to city, so one global number would be misleading. Median over mean because these values are skewed.
- `AQI` (16% missing) → **drop the rows**. It's the main variable everything else depends on — didn't want to guess it.

**Crop data:**
- `Production` (1.5% missing) → **drop the rows**. Barely any loss, not worth imputing.""")

code(r"""air = air_raw.copy()
crop = crop_raw.copy()

print("Missing values BEFORE treatment (air):")
print(air.isna().sum())
print()
print("Missing values BEFORE treatment (crop):")
print(crop["Production"].isna().sum())""")

code(r"""# Drop pollutant columns that are barely usable or not needed for this lab
air = air.drop(columns=["NH3", "NOx", "NO", "O3", "SO2", "Benzene", "Toluene", "Xylene", "AQI_Bucket"])

# AQI is the key variable - drop rows where it's missing rather than guess it
air = air.dropna(subset=["AQI"])

# Remaining pollutants: fill gaps with each city's own median.
# A couple of cities have NO PM10 readings at all, so their "city median" is
# also NaN - catch those leftovers with the overall column median.
for col in ["PM2.5", "PM10", "NO2", "CO"]:
    air[col] = air.groupby("City")[col].transform(lambda s: s.fillna(s.median()))
    air[col] = air[col].fillna(air[col].median())

# Crop: drop the small number of rows missing Production
crop = crop.dropna(subset=["Production"])

print("Missing values AFTER treatment (air):")
print(air.isna().sum())
print()
print("Missing values AFTER treatment (crop):", crop["Production"].isna().sum())
print()
print("Air rows remaining:", len(air))
print("Crop rows remaining:", len(crop))""")

md("Zero nulls left in both files. Air data lost about 16% of its rows — all the ones with no AQI — which is a fair trade for not faking the most important column.")

# ---------------------------------------------------------------------------
md("## Task 3: State Names & Duplicate Records\n\nAir data only has `City`, so I need a state for each one before it can ever join to the crop file. Crop's state names also have stray whitespace that would quietly create fake duplicate states.")

code(r"""city_to_state = {
    "Ahmedabad": "Gujarat", "Aizawl": "Mizoram", "Amaravati": "Andhra Pradesh",
    "Amritsar": "Punjab", "Bengaluru": "Karnataka", "Bhopal": "Madhya Pradesh",
    "Brajrajnagar": "Odisha", "Chandigarh": "Chandigarh", "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Delhi": "Delhi", "Ernakulam": "Kerala",
    "Gurugram": "Haryana", "Guwahati": "Assam", "Hyderabad": "Telangana",
    "Jaipur": "Rajasthan", "Jorapokhar": "Jharkhand", "Kochi": "Kerala",
    "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh", "Mumbai": "Maharashtra",
    "Patna": "Bihar", "Shillong": "Meghalaya", "Talcher": "Odisha",
    "Thiruvananthapuram": "Kerala", "Visakhapatnam": "Andhra Pradesh",
}

air["State"] = air["City"].map(city_to_state)
print("Cities with no state match:", air.loc[air["State"].isna(), "City"].unique())""")

code(r"""print("Crop state names with extra whitespace, before cleaning:")
print([s for s in crop["State_Name"].unique() if s != s.strip()])
print("Crop season values before cleaning:")
print(crop["Season"].unique())

crop["State_Name"] = crop["State_Name"].str.strip()
crop["Season"] = crop["Season"].str.strip()
crop["Crop"] = crop["Crop"].str.strip()

print()
print("Crop season values after cleaning:")
print(crop["Season"].unique())""")

code(r"""air_before, crop_before = len(air), len(crop)

air = air.drop_duplicates()
crop = crop.drop_duplicates(subset=["State_Name", "District_Name", "Crop_Year", "Season", "Crop"])

pd.DataFrame({
    "Dataset": ["air", "crop"],
    "Rows before": [air_before, crop_before],
    "Rows after": [len(air), len(crop)],
    "Duplicates removed": [air_before - len(air), crop_before - len(crop)],
})""")

md(r"""**Found and fixed:**
- No `State` column in air data — added one with a city-to-state lookup. Every city matched.
- Some crop state names, and *every* `Season` value, had trailing spaces — `.str.strip()` sorted both.
- Checked for exact duplicate rows and repeated state-district-crop-year-season combos in both files. Genuinely none — good, but worth checking rather than assuming.

Skipping the whitespace fix would've quietly split `"Telangana"` and `"Telangana "` into two different states later.""")

# ---------------------------------------------------------------------------
md("## Task 4: Where Do Cities Sit on the AQI Scale?\n\nTwo questions: where do most values cluster (histogram), and is the mean skewed by a few extreme readings (boxplot). Using both, side by side.")

code(r"""mean_aqi = air["AQI"].mean()
median_aqi = air["AQI"].median()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(air["AQI"], bins=40, kde=True, ax=axes[0], color="#4C78A8")
axes[0].axvline(mean_aqi, color="red", linestyle="--", label=f"Mean: {mean_aqi:.0f}")
axes[0].axvline(median_aqi, color="green", linestyle="-", label=f"Median: {median_aqi:.0f}")
axes[0].set_title("Distribution of AQI across all city-days")
axes[0].set_xlabel("AQI")
axes[0].set_ylabel("Number of records")
axes[0].legend()

sns.boxplot(y=air["AQI"], ax=axes[1], color="#F58518")
axes[1].set_title("AQI spread and outliers")
axes[1].set_ylabel("AQI")

plt.tight_layout()
plt.show()

print(f"Mean AQI: {mean_aqi:.1f}, Median AQI: {median_aqi:.1f}")""")

md("Most city-days sit in the moderate-to-poor range (roughly 80-200) — pollution's a widespread problem, not just a few bad cities. Mean is well above median, and the boxplot shows a long tail past 500, so a handful of very bad days are dragging the average up. Median's the fairer number to report publicly.")

# ---------------------------------------------------------------------------
md("## Task 5: Handling Extreme AQI Values\n\nThe real AQI scale tops out around 400-500 (\"Severe\"). Anything way past that is a sensor glitch, not a real reading. Using the IQR rule to catch these automatically, then capping (not deleting) so I keep the row.")

code(r"""q1 = air["AQI"].quantile(0.25)
q3 = air["AQI"].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr

extreme_count = (air["AQI"] > upper_bound).sum()
print(f"Upper bound from IQR rule: {upper_bound:.1f}")
print(f"Values above this bound: {extreme_count} out of {len(air)} rows ({extreme_count/len(air)*100:.1f}%)")

air_before_capping = air["AQI"].copy()
air["AQI"] = air["AQI"].clip(upper=upper_bound)""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=air_before_capping, ax=axes[0], color="#F58518")
axes[0].set_title("AQI before capping")
axes[0].set_ylabel("AQI")

sns.boxplot(y=air["AQI"], ax=axes[1], color="#54A24B")
axes[1].set_title("AQI after capping")
axes[1].set_ylabel("AQI")

plt.tight_layout()
plt.show()""")

md("IQR works well here since AQI is already skewed — no assumption of symmetry needed. Capping instead of deleting keeps the city/date info for that row, it just can't distort the average anymore. Boxplots show it clearly: same shape, tail gone.")

# ===========================================================================
md("# Lab 2 — Data Exploration & Inferences")

md("## Task 6: Is Air Quality Improving Over Time?\n\nA journalist wants a plain answer: better, worse, or the same since 2015? Pulling `Year` out of the date, then a line plot for the trend.")

code(r"""air["Date"] = pd.to_datetime(air["Date"])
air["Year"] = air["Date"].dt.year
air["Month"] = air["Date"].dt.month

yearly_aqi = air.groupby("Year", as_index=False)["AQI"].mean()
worst_year = yearly_aqi.loc[yearly_aqi["AQI"].idxmax()]
best_year = yearly_aqi.loc[yearly_aqi["AQI"].idxmin()]

plt.figure(figsize=(9, 5))
sns.lineplot(data=yearly_aqi, x="Year", y="AQI", marker="o", linewidth=2.5, color="#4C78A8")
plt.scatter(worst_year["Year"], worst_year["AQI"], color="red", s=120, zorder=5, label=f"Worst: {int(worst_year['Year'])}")
plt.scatter(best_year["Year"], best_year["AQI"], color="green", s=120, zorder=5, label=f"Best: {int(best_year['Year'])}")
plt.title("Average AQI by Year (2015-2020)")
plt.xlabel("Year")
plt.ylabel("Average AQI")
plt.legend()
plt.tight_layout()
plt.show()

print(yearly_aqi)""")

md("**For the journalist:** air quality's been getting steadily better each year, not worse — average AQI drops from about 213 in 2015 to 114 in 2020. But 2020 only runs to July and overlaps COVID lockdowns, which alone would push AQI down. So the improving trend looks real, just don't credit 2020 entirely to policy.")

# ---------------------------------------------------------------------------
md("## Task 7: Is Air Quality Really Worse at Harvest Time?\n\nAn NGO says Oct-Dec (crop-burning season) is consistently worse. Checking average AQI by month — a bar chart makes 12 categories easy to compare.")

code(r"""month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

monthly_aqi = air.groupby("Month", as_index=False)["AQI"].mean()
monthly_aqi["Month_name"] = monthly_aqi["Month"].map(month_names)

colors = ["#E45756" if m in (10, 11, 12) else "#4C78A8" for m in monthly_aqi["Month"]]

plt.figure(figsize=(11, 5))
sns.barplot(data=monthly_aqi, x="Month_name", y="AQI", order=[month_names[m] for m in range(1, 13)], palette=colors)
plt.title("Average AQI by Month (harvest months Oct-Dec highlighted in red)")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.tight_layout()
plt.show()

print(monthly_aqi)""")

md("**Response to the NGO:** partly right. Oct-Dec is clearly worse than the monsoon months (110-120 vs 190-240), so the harvest-season link holds. But January is just as bad, and that has nothing to do with crop burning — so winter weather is likely doing as much damage as residue burning, not less.")

# ---------------------------------------------------------------------------
md(r"""## Task 8: Making the Two Datasets Talk to Each Other

**Transformation needed:** air is city-day, crop is district-crop-year — totally different grains, so a row-by-row join is meaningless. Worse, the years barely overlap: crop data stops in 2015, right when air data starts, and only one state has crop records in that exact year. A state-and-year merge would leave almost nothing.

**Fix:** collapse both down to one row per state — average AQI/pollutants for air, total production and area for crop — and merge on `State` alone. Loses the year dimension, but it's the only way to compare them at all.""")

code(r"""air_state = air.groupby("State", as_index=False).agg(
    Avg_AQI=("AQI", "mean"),
    Avg_PM25=("PM2.5", "mean"),
    Avg_PM10=("PM10", "mean"),
    Avg_NO2=("NO2", "mean"),
    Avg_CO=("CO", "mean"),
)

# Coconut is counted in individual nuts, not tonnes - mixing it in with weight-based
# crops like rice or wheat wrecks the production totals for coconut-growing states.
crop_for_yield = crop[crop["Crop"] != "Coconut"]

crop_state = crop_for_yield.groupby("State_Name", as_index=False).agg(
    Total_Area=("Area", "sum"),
    Total_Production=("Production", "sum"),
).rename(columns={"State_Name": "State"})
crop_state["Yield"] = crop_state["Total_Production"] / crop_state["Total_Area"]

merged = pd.merge(air_state, crop_state, on="State", how="inner")
print("States present in both files:", merged.shape[0])
display(merged.sort_values("Avg_AQI"))""")

code(r"""numeric_cols = merged.select_dtypes(include=np.number).columns
corr_matrix = merged[numeric_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5)
plt.title("Correlation Matrix - Merged State-Level Data")
plt.tight_layout()
plt.show()""")

md(r"""**Two things worth flagging:**

1. **AQI vs Yield is basically flat.** Gujarat has by far the worst air in this data but an entirely ordinary yield — breaks the simple "more pollution, less crop" story on its own. Probably comes down to irrigation, soil, and crop choice mattering more than air quality.
2. **AQI is *positively* linked to total production and area.** Looks backwards, but it's likely just a size effect — bigger states have more industry (higher AQI) and more farmland, not because pollution helps crops grow.

A correlation matrix only shows things move together, never why.""")

# ---------------------------------------------------------------------------
md(r"""## Task 9: Briefing for the Minister

Minister, three things stood out.

**One:** air quality has improved almost every year since 2015 — a good sign, though the last year overlaps COVID lockdowns, so not all of that credit belongs to policy.

**Two:** pollution peaks October through January — harvest season plus winter together. Any clean-air push should target these months specifically, not run flat all year.

**Three:** once the data is cleaned up properly, there's no clear link between a state's air quality and its farm output. Our most polluted state has an entirely ordinary crop yield.

**Recommendation:** focus pollution controls on October-January, since that's where the real spike is.

**Honest limit:** our air and farm records don't come from the same years, so we can't yet say pollution and crop output are connected — only that we found no sign of it so far.""")

# ===========================================================================
md("# Optional Advanced Tasks")

md("## Task A: Do the Two Extremes Tell the Same Story?\n\nIf \"more pollution, less crop\" were true, the dirtiest and cleanest states should look clearly different. Let's check.")

code(r"""extremes = pd.concat([
    merged.nsmallest(3, "Avg_AQI").assign(Group="3 least polluted"),
    merged.nlargest(3, "Avg_AQI").assign(Group="3 most polluted"),
])

plt.figure(figsize=(10, 5))
sns.barplot(data=extremes, x="State", y="Yield", hue="Group", dodge=False)
plt.title("Crop Yield: Least Polluted vs Most Polluted States")
plt.xlabel("State")
plt.ylabel("Yield (production per unit area)")
plt.tight_layout()
plt.show()

display(extremes[["State", "Avg_AQI", "Yield", "Group"]])""")

md("Doesn't separate cleanly at all. Gujarat and Bihar (dirtiest air) sit mid-pack on yield, while Mizoram (cleanest air) has one of the *lowest* yields in the dataset. If anything, this argues against the hypothesis, not for it.")

md("## Task B: Putting a Number on the Relationship\n\nPearson's r measures the strength and direction of a straight-line relationship, from -1 to +1.")

code(r"""r_value = merged["Avg_AQI"].corr(merged["Yield"])
print(f"Correlation between Avg_AQI and Yield: {r_value:.3f}")

plt.figure(figsize=(8, 5))
sns.regplot(data=merged, x="Avg_AQI", y="Yield", scatter_kws={"s": 70}, line_kws={"color": "red"})
plt.title(f"AQI vs Crop Yield by State (r = {r_value:.2f})")
plt.xlabel("Average AQI")
plt.ylabel("Yield (production per unit area)")
plt.tight_layout()
plt.show()""")

md("A value that close to zero is, for practical purposes, no relationship — 0.8 would be strong, 0.2 already weak. Honest read for the research team: no evidence pollution and yield are linked here, but that's not proof they *aren't* — mismatched years and rough state averages just aren't strong enough to catch it either way.")

md(r"""## Task C: One Plot to Rule Them All

**Pick: the AQI vs Yield scatter from Task B.**

It's the one chart that actually tests the headline claim ("polluted states produce less") against real numbers — and shows it doesn't hold. Every other chart here looks at air or crop data alone; this is the only one that puts them side by side, honestly, flat line and all.

**Caveat across every optional task:** this doesn't prove pollution affects farming, or that it doesn't. Rainfall, irrigation, soil quality and farmer income aren't in this data at all, and any of them could be the real story behind (or hiding) a relationship.""")

# ===========================================================================
md("# Self Learning")

md("""Something the lab sheet didn't ask for, but I ran into anyway while doing Task 8.""")

code(r"""kerala_check = crop[crop["State_Name"] == "Kerala"].groupby("Crop")["Production"].sum().sort_values(ascending=False)
print(kerala_check.head(3))""")

md(r"""**Topic: why one crop nearly wrecked the whole analysis.**

My first attempt at the state-level merge gave Kerala a "yield" of over 3,000 — obviously wrong, since every other state was under 15. I didn't know why, so I dug into which crop was driving Kerala's production total, and it was almost entirely **Coconut**.

Turns out this is a known quirk of this exact Kaggle dataset (I looked it up after spotting the number): coconut production is recorded in **individual nuts**, not tonnes, while every other crop here is recorded by weight. So a state that grows a lot of coconuts racks up a "production" number in the tens of millions that has nothing to do with tonnes of food, and it silently wrecks any total that mixes it in with rice or wheat.

I hadn't thought about raw data being able to mix *units*, not just formats or spellings, before this. The fix was simple once I understood it — drop Coconut before summing production for the yield calculation — but finding it meant not trusting a weird number just because the code ran without errors.""")

# ---------------------------------------------------------------------------
md(r"""## Deliverables Check

- Tasks 1-9 + optional A/B/C, all numbered with a reasoning cell each.
- 7 labelled visualisations across the notebook.
- Correlation-vs-causation and confounders flagged wherever a relationship comes up.""")

nb["cells"] = cells
nbf.write(nb, "lab09_2547116.ipynb")
print("Notebook written.")
