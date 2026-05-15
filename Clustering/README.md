# Car Recommendation System via Custom Clustering

This repository contains a Jupyter Notebook that implements a car recommendation engine from scratch. The project relies on custom implementations of density-based and hierarchical clustering algorithms to group similar vehicles and recommend them based on user preferences.

## Features

* **Custom Clustering Algorithms**: Implements DBSCAN, OPTICS, and Hierarchical clustering entirely from scratch (or utilizing basic distance matrices).
* **Automated Hyperparameter Tuning**: Includes grid-search-style analysis for clustering parameters (e.g., varying `eps` and `min_pts` for DBSCAN).
* **End-to-End Recommendation Pipeline**: Maps raw user input into a clustered feature space to find the closest matching vehicle profile.
* **Custom Preprocessing Pipeline**: Handles complex tabular data with a dedicated `Preprocessing` class.

## Clustering Algorithms Implemented

The project implements the following clustering techniques:

1. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
   * Implements custom cluster expansion (`expand_cluster`) operating on distance matrices.
   * Identifies core points, edge points, and noise based on `eps` and `min_pts`.
2. **OPTICS (Ordering Points To Identify the Clustering Structure)**
   * Implements reachability-distance based cluster ordering (`expand_cluster_order`).
   * Handles clusters of varying densities better than standard DBSCAN.
3. **Hierarchical Clustering**
   * Custom `Hierarchical` class supporting configurable cluster counts and linkage criteria (e.g., `linkage='single'`).
   * Integrates with `scipy.cluster.hierarchy` for dendrogram visualization.

## Data Preprocessing

The notebook features a robust, custom-built preprocessing workflow:
* **Custom `Preprocessing` Class**: Handles missing values and scales numeric/categorical features.
* **Encoding**: Implements custom One-Hot Encoding for categorical vehicle attributes.
* **Data Splits**: Properly manages train, validation, and test sets (`X_train`, `X_val`, `X_test`) to prevent data leakage during distance calculations.

## The Recommendation Engine

The core of the application is the `car_recommender` function:
1. **User Input**: Takes raw user preferences for a car.
2. **Transformation**: Passes the input through the custom `preprocessor` to match the model's scaled feature space.
3. **Matching**: Compares the processed user vector against the derived cluster `centroids`.
4. **Output**: Returns the best-matching car recommendations from the original dataset (`df_analysis`).
