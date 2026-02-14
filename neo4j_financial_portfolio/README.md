# neo4j_financial_portfolio


**UC Berkeley MIDS | DATASCI 205** • Michael Strommer, Leo Lazzarini, Shikha Sharma • August 2025

## Overview

This project uses graph databases to aid financial asset allocation by uncovering hidden relationships and risk clusters that traditional correlation analysis misses. Built for Hedge Fund, we developed two graph structures analyzing investment pod relationships and factor exposures to optimize portfolio diversification.

## Problem & Solution

**Challenge:** Traditional methods using Sharpe ratios and correlations fail to identify hidden common exposures and non-linear risk clusters.

**Solution:** Graph network analysis with clustering algorithms to reveal portfolio structure and diversification gaps through community detection.

## Technical Implementation

**Architecture:** Neo4j graph database with MongoDB (document storage) and Redis (real-time processing)

**Graph Design:**
- **Graph 1:** Temporal network tracking weekly returns across investment pods (2020-2025)
- **Graph 2:** Multi-dimensional factor exposure mapping

**Algorithms:** Pearson Similarity Correlation, Louvain Modularity Algorithm, Linear Regression for time series analysis

## Key Results

**Community Detection:** Identified 4 distinct investment communities:
- Long Term Equity (Global, US)
- High Yield Equity (Global, European)  
- Long Term Total Return (Europe, US, Sweden)
- Long Short Equity (Global)

**Business Impact:** Recommendation to diversify across communities rather than within clusters, significantly improving risk management through hidden correlation identification.

## Skills Demonstrated

Graph Database Design • Data Engineering • Machine Learning • Financial Analytics • Time Series Analysis • Portfolio Theory • Network Analysis

---
*Advanced data engineering and graph theory applied to solve real-world quantitative finance challenges.*
