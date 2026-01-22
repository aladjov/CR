import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

# Output directory
FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"

n_rows = 30801

print("Generating synthetic retail dataset with 30,801 rows and 15 columns...")

custid = range(1, n_rows + 1)

retained = np.random.choice([0, 1], size=n_rows, p=[0.25, 0.75])

start_date = datetime(2022, 1, 1)
created = [start_date + timedelta(days=int(np.random.exponential(180))) for _ in range(n_rows)]

firstorder = []
for c in created:
    days_after = int(np.random.exponential(10)) + 1
    firstorder.append(c + timedelta(days=days_after))

lastorder = []
for f in firstorder:
    days_after = int(np.random.exponential(60)) + 1
    lastorder.append(f + timedelta(days=days_after))

esent = np.random.poisson(lam=15, size=n_rows)

eopenrate = np.clip(np.random.beta(2, 2, size=n_rows), 0.1, 0.9)

eclickrate = np.clip(eopenrate * np.random.beta(2, 5, size=n_rows), 0, 1)

avgorder = np.random.lognormal(mean=4.0, sigma=0.5, size=n_rows)

ordfreq = np.random.gamma(shape=2, scale=1.5, size=n_rows)

paperless = np.random.choice([0, 1], size=n_rows, p=[0.4, 0.6])
refill = np.random.choice([0, 1], size=n_rows, p=[0.5, 0.5])
doorstep = np.random.choice([0, 1], size=n_rows, p=[0.3, 0.7])

favday = np.random.randint(0, 7, size=n_rows)

cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
          "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
city = np.random.choice(cities, size=n_rows, p=[0.15, 0.12, 0.11, 0.10, 0.09,
                                                 0.09, 0.08, 0.08, 0.09, 0.09])

df = pd.DataFrame({
    "custid": custid,
    "retained": retained,
    "created": created,
    "firstorder": firstorder,
    "lastorder": lastorder,
    "esent": esent,
    "eopenrate": np.round(eopenrate, 3),
    "eclickrate": np.round(eclickrate, 3),
    "avgorder": np.round(avgorder, 2),
    "ordfreq": np.round(ordfreq, 2),
    "paperless": paperless,
    "refill": refill,
    "doorstep": doorstep,
    "favday": favday,
    "city": city
})

output_path = FIXTURES_DIR / "customer_retention_retail.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Dataset generated successfully!")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Saved to: {output_path}")
print(f"\nColumn summary:")
print(df.info())
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nRetention rate: {df['retained'].mean():.2%}")
