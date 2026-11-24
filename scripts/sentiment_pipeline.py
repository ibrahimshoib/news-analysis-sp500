"""
Financial News Sentiment Pipeline
Extracts sentiment signals from nasdaq_clean.csv using domain-adapted financial VADER
Aggregates by trading date and produces daily time series features.
"""

import time
import os
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Domain-specific financial lexicon adaptations (Loughran-McDonald & Wall St terms)
FINANCIAL_LEXICON = {
    # Bullish / Positive
    "soars": 2.5, "soared": 2.5, "soaring": 2.5,
    "surges": 2.5, "surged": 2.5, "surging": 2.5,
    "rallies": 2.0, "rallied": 2.0, "rally": 2.0,
    "outperform": 2.0, "outperformed": 2.0, "outperforming": 2.0,
    "upgrade": 2.0, "upgraded": 2.0, "upgrades": 2.0,
    "bullish": 2.0, "bull": 1.5,
    "dividend hike": 2.5, "buyback": 2.0,
    "beats estimate": 2.5, "beat estimates": 2.5, "beats consensus": 2.5,
    "earnings beat": 2.5, "revenue beats": 2.5,
    "record high": 2.5, "all-time high": 2.5,
    "profit jumps": 2.5, "growth accelerates": 2.0,
    "breakout": 1.8, "upside": 1.8,
    
    # Bearish / Negative
    "plunges": -2.5, "plunged": -2.5, "plunging": -2.5,
    "tumbles": -2.5, "tumbled": -2.5, "tumbling": -2.5,
    "slumps": -2.0, "slumped": -2.0, "slump": -2.0,
    "downgrade": -2.0, "downgraded": -2.0, "downgrades": -2.0,
    "underperform": -2.0, "underperformed": -2.0,
    "bearish": -2.0, "bear": -1.5,
    "misses estimate": -2.5, "missed estimates": -2.5, "misses consensus": -2.5,
    "earnings miss": -2.5, "revenue misses": -2.5,
    "layoffs": -2.0, "job cuts": -2.0, "firing": -1.5,
    "bankruptcy": -3.0, "bankrupt": -3.0, "default": -2.5,
    "investigation": -2.0, "lawsuit": -1.8, "probe": -1.8,
    "fraud": -3.0, "scandal": -2.5, "sec probe": -2.5,
    "profit drops": -2.5, "loss widens": -2.5, "guidance cut": -2.5,
    "selloff": -2.2, "crash": -3.0, "recession": -2.2
}

def get_analyzer():
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(FINANCIAL_LEXICON)
    return sia

def process_news_sentiment(input_file="nasdaq_clean.csv",
                           daily_output="daily_sentiment_market.csv",
                           sample_output="sample_scored_articles.csv",
                           chunk_size=50000):
    sia = get_analyzer()
    print(f"Starting sentiment analysis on {input_file} in chunks of {chunk_size}...")
    t0 = time.time()
    
    daily_stats = {}
    sample_records = []
    total_processed = 0
    
    # Read only essential columns to preserve memory
    cols = ["published_date", "Date", "title", "market_return"]
    
    chunk_idx = 0
    for chunk in pd.read_csv(input_file, usecols=cols, chunksize=chunk_size):
        chunk_idx += 1
        chunk_t0 = time.time()
        
        # Clean titles
        titles = chunk["title"].fillna("").astype(str).tolist()
        
        # Score sentiments
        compounds = []
        pos_list = []
        neg_list = []
        neu_list = []
        
        for t in titles:
            scores = sia.polarity_scores(t)
            compounds.append(scores["compound"])
            pos_list.append(scores["pos"])
            neg_list.append(scores["neg"])
            neu_list.append(scores["neu"])
            
        chunk["compound"] = compounds
        chunk["pos"] = pos_list
        chunk["neg"] = neg_list
        chunk["neu"] = neu_list
        
        # Binary classifications for count aggregation
        chunk["is_pos"] = (chunk["compound"] >= 0.05).astype(int)
        chunk["is_neg"] = (chunk["compound"] <= -0.05).astype(int)
        chunk["is_neu"] = ((chunk["compound"] > -0.05) & (chunk["compound"] < 0.05)).astype(int)
        
        # Keep first 5000 records across chunks for inspection
        if len(sample_records) < 5000:
            sample_records.append(chunk[["published_date", "Date", "title", "compound", "pos", "neg", "market_return"]].head(1000))
        
        # Group by published_date and Date
        # Note: 'Date' is the next trading day of S&P500
        grouped = chunk.groupby(["published_date", "Date", "market_return"]).agg(
            n_articles=("compound", "count"),
            compound_sum=("compound", "sum"),
            compound_sq_sum=("compound", lambda x: np.sum(x**2)),
            pos_count=("is_pos", "sum"),
            neg_count=("is_neg", "sum"),
            neu_count=("is_neu", "sum"),
            pos_score_sum=("pos", "sum"),
            neg_score_sum=("neg", "sum")
        ).reset_index()
        
        for _, row in grouped.iterrows():
            key = (row["published_date"], row["Date"], row["market_return"])
            if key not in daily_stats:
                daily_stats[key] = {
                    "n_articles": 0,
                    "compound_sum": 0.0,
                    "compound_sq_sum": 0.0,
                    "pos_count": 0,
                    "neg_count": 0,
                    "neu_count": 0,
                    "pos_score_sum": 0.0,
                    "neg_score_sum": 0.0
                }
            d = daily_stats[key]
            d["n_articles"] += int(row["n_articles"])
            d["compound_sum"] += float(row["compound_sum"])
            d["compound_sq_sum"] += float(row["compound_sq_sum"])
            d["pos_count"] += int(row["pos_count"])
            d["neg_count"] += int(row["neg_count"])
            d["neu_count"] += int(row["neu_count"])
            d["pos_score_sum"] += float(row["pos_score_sum"])
            d["neg_score_sum"] += float(row["neg_score_sum"])
            
        total_processed += len(chunk)
        elapsed = time.time() - t0
        rate = total_processed / elapsed
        print(f"Chunk {chunk_idx:2d} done: {len(chunk):,} articles in {time.time()-chunk_t0:.1f}s | Total: {total_processed:,} ({rate:.0f} arts/s)")
    
    print("\nAssembling daily summary table...")
    rows = []
    for (pub_date, mkt_date, mkt_ret), d in daily_stats.items():
        n = d["n_articles"]
        if n == 0:
            continue
        mean_compound = d["compound_sum"] / n
        var_compound = max(0.0, (d["compound_sq_sum"] / n) - (mean_compound ** 2))
        std_compound = np.sqrt(var_compound)
        
        pos_ratio = d["pos_count"] / n
        neg_ratio = d["neg_count"] / n
        neu_ratio = d["neu_count"] / n
        net_sentiment = pos_ratio - neg_ratio
        bull_bear_ratio = (d["pos_count"] + 1) / (d["neg_count"] + 1)
        
        rows.append({
            "published_date": pub_date,
            "market_date": mkt_date,
            "article_count": n,
            "sentiment_mean": mean_compound,
            "sentiment_std": std_compound,
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
            "neu_ratio": neu_ratio,
            "net_sentiment": net_sentiment,
            "bull_bear_ratio": bull_bear_ratio,
            "pos_score_mean": d["pos_score_sum"] / n,
            "neg_score_mean": d["neg_score_sum"] / n,
            "market_return": mkt_ret,
            "market_direction": 1 if mkt_ret > 0 else 0,
            "market_return_abs": abs(mkt_ret)
        })
        
    df_daily = pd.DataFrame(rows)
    df_daily["published_date"] = pd.to_datetime(df_daily["published_date"])
    df_daily["market_date"] = pd.to_datetime(df_daily["market_date"])
    df_daily = df_daily.sort_values("published_date").reset_index(drop=True)
    
    # Add rolling momentum indicators
    df_daily["sentiment_ma3"] = df_daily["sentiment_mean"].rolling(3, min_periods=1).mean()
    df_daily["sentiment_ma7"] = df_daily["sentiment_mean"].rolling(7, min_periods=1).mean()
    df_daily["sentiment_momentum_7d"] = df_daily["sentiment_mean"] - df_daily["sentiment_ma7"]
    df_daily["volume_ma7"] = df_daily["article_count"].rolling(7, min_periods=1).mean()
    df_daily["volume_surprise"] = df_daily["article_count"] / (df_daily["volume_ma7"] + 1e-6)
    
    df_daily.to_csv(daily_output, index=False)
    print(f"Saved daily dataset ({len(df_daily):,} days) to {daily_output}")
    
    if sample_records:
        df_sample = pd.concat(sample_records, ignore_index=True)
        df_sample.to_csv(sample_output, index=False)
        print(f"Saved sample scored articles ({len(df_sample):,} rows) to {sample_output}")
        
    print(f"Total time: {time.time()-t0:.1f}s")
    return df_daily

if __name__ == "__main__":
    process_news_sentiment()
