# Medallion Architecture for Event- and Entity-Based Datasets

I am working on a detailed Medallion Architecture that handles both event-based and entity-based datasets and converts them to a single Gold dataset useful for modeling.

There are two important things to track:
1) the Delta table names in the respective layers  
2) the script names producing them  

The architecture also standardizes Point-in-Time (PIT) columns differently for event datasets, entity datasets, and aggregated datasets.

---

## Event-based dataset example: customer_emails

Layer | Dataset name | Script name
------|--------------|------------
landing | customer_emails | landing_customer_emails
bronze | customer_emails | bronze_event_customer_emails
bronze | customer_emails_aggregated | bronze_entity_customer_emails_aggregated
silver | silver_featureset_{{CN}} | silver_featureset_{{CN}}
gold | gold_features_{{CN}} | gold_features_{{CN}}

---

## Entity-based dataset example: customer_profiles

Layer | Dataset name | Script name
------|--------------|------------
landing | customer_profiles | landing_customer_profiles
bronze | customer_profiles | bronze_entity_customer_profiles
silver | silver_featureset_{{CN}} | silver_featureset_{{CN}}
gold | gold_features_{{CN}} | gold_features_{{CN}}

---

## Composite dataset name {{CN}}

{{CN}} is a composite name of all data sources included in the Silver and Gold datasets.

Requirements:
- human-readable
- deterministic
- unique per combination of data sources

Recommended structure:
- readable prefix derived from source names
- short hash suffix derived from canonical ordered list of sources

Example:
customer_emails_profiles__vxvt2xv

---

# Point-in-Time (PIT) Columns and Standardization

PIT columns are introduced differently depending on dataset type.  
They are not introduced once globally, but derived at the correct stage in the pipeline.

---

## Event datasets

Event datasets must introduce a standard column:

event_timestamp

This represents when the real-world event occurred.

Derived from:
- source datetime column
- ingestion time fallback if unavailable

Used for:
- PIT filtering at raw event level
- aggregations into entity features

Example Bronze event table:

customer_id | event_type | event_timestamp
------------|------------|----------------
1 | email_sent | 2000-01-01
1 | email_sent | 2000-01-10

---

## Aggregated datasets (event to entity)

Aggregation converts event-time data into feature-time data.

This introduces:

feature_timestamp

This represents when the feature value is valid for prediction.

Typical derivation:
- window end
- snapshot time
- aggregation time

Example:

customer_id | emails_last_30d | feature_timestamp
------------|------------------|------------------
1 | 2 | 2000-01-31

After this step:
- event_timestamp is no longer used for modeling PIT
- feature_timestamp becomes the primary time axis

---

## Entity datasets (one row per customer)

Entity datasets must introduce:

feature_timestamp

This represents when the entity state was valid.

Derived from:
- last update time
- snapshot time
- ingestion timestamp fallback

Example:

customer_id | age | plan | feature_timestamp
------------|-----|------|------------------
1 | 34 | premium | 2000-01-20

---

# Silver Layer

The Silver layer joins entity-grain datasets and aggregated event features into a single feature base:

silver_featureset_{{CN}}

At this stage:
- all rows are entity-level
- all rows contain feature_timestamp
- feature_timestamp becomes the primary PIT anchor

Used for:
- building modeling datasets
- training
- inference
- evaluation

---

# Gold Layer

The Gold layer produces the final modeling dataset:

gold_features_{{CN}}

This layer introduces outcome information:

- label_timestamp
- target
- optional label_available_flag

These columns are introduced only in Gold and must not exist in Bronze or Silver.

---

## Why label fields exist only in Gold

label_timestamp represents when the outcome became known.

target represents the outcome value used for supervision.

These are only needed for:
- training datasets
- evaluation datasets
- monitoring

They must not be used in:
- feature engineering
- aggregations
- Silver joins

This prevents:
- leakage
- target-derived features
- contamination of modeling inputs

---

## Example Gold dataset

customer_id | features... | feature_timestamp | label_timestamp | target
------------|-------------|------------------|-----------------|-------
1 | ... | 2000-01-31 | 2000-02-01 | true

---

# PIT Filtering Logic

## Raw event PIT (before aggregation)

Used during aggregation:

event_timestamp <= cutoff

This determines which events were visible at a given time.

---

## Feature PIT (after aggregation and for entity datasets)

Used for modeling and scoring:

feature_timestamp <= cutoff

This determines which feature states were valid at prediction time.

---

## Training dataset construction

feature_timestamp <= cutoff  
label_timestamp > cutoff  

Meaning:
- prediction happens at cutoff
- outcome occurs later

---

## Scoring or inference dataset

feature_timestamp <= now  

Labels ignored.

---

# Responsibilities of Standardization Scripts

Two types of standardization scripts are required.

---

## Script A — event dataset standardization

Landing or Bronze layer:

Introduces:
event_timestamp

Used for:
- PIT filtering of raw data
- aggregation

---

## Script B — entity and aggregated dataset standardization

Bronze layer:

Introduces:
feature_timestamp

Used for:
- modeling PIT
- training and scoring filters

---

# Summary

This Medallion Architecture:

- distinguishes event datasets from entity datasets
- introduces PIT timestamps at the correct layer
- converts event_timestamp into feature_timestamp during aggregation
- uses feature_timestamp as the primary PIT anchor for modeling
- introduces label_timestamp and target only in Gold
- prevents leakage by keeping labels out of feature engineering layers
- produces a single reusable feature base (silver_featureset_{{CN}})
- produces a single modeling dataset (gold_features_{{CN}})
