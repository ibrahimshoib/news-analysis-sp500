import pandas as pd

NEWS_FILE = "nasdaq_exteral_data.csv"  # Adjust to your actual filename
INDEX_FILE = "sp500_index.csv"
OUTPUT_FILE = "nasdaq_clean.csv"

# 1. Prepare the index and calculate daily returns
index = pd.read_csv(INDEX_FILE)

index["Date"] = pd.to_datetime(index["Date"], errors="coerce")
index["S&P500"] = pd.to_numeric(index["S&P500"], errors="coerce")

index = (
    index.dropna(subset=["Date", "S&P500"])
         .drop_duplicates("Date")
         .sort_values("Date")
        # earliest to latest
)

index = index.loc[index["S&P500"] > 0].copy()
index["market_return"] = index["S&P500"].pct_change(fill_method=None)
# add a row for percentage changes.

# clean news in chunks
columns = ["Date", "Article_title", "Url", "Article"]
seen_urls = set()
first_chunk = True

chunknum=1

for news in pd.read_csv(
    NEWS_FILE,
    usecols=columns,
    dtype="string",
    chunksize=10_000
    # have to read in chunks because of massive file size
):

    print("\nProcessing chunk #",chunknum)
    chunknum+=1

    news = news.rename(columns={
        "Date": "published_at",
        "Article_title": "title",
        "Url": "url",
        "Article": "text"
    })

    # # Keep Nasdaq articles; other sources have different coverage/formats
    # news["url"] = news["url"].str.strip()
    # news = news.loc[
    #     news["url"].str.contains(
    #         r"^https?://(?:www\.)?nasdaq\.com/",
    #         case=False,
    #         regex=True,
    #         na=False
    #     )
    # ].copy()

    # Preserve the recorded calendar date.
    # Do not interpret placeholder midnight times as actual release times.
    news["published_date"] = pd.to_datetime(
        news["published_at"].str[:10],
        format="%Y-%m-%d",
        errors="coerce"
    )

    news = news.dropna(subset=["published_date", "title", "text", "url"])

    # Restrict to the main study period covered by the index
    news = news.loc[
        news["published_date"].between(
            index["Date"].min(), pd.Timestamp("2023-12-31")
        )
    ].copy()

    # Light text cleaning: preserve numbers, punctuation and negation
    for column in ["title", "text"]:
        news[column] = (
            news[column]
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
        )

    # Remove the repeated Nasdaq disclaimer
    disclaimer = (
        "The views and opinions expressed herein are the views and "
        "opinions of the author and do not necessarily reflect those "
        "of Nasdaq, Inc."
    )
    news["text"] = (
        news["text"]
            .str.replace(disclaimer, "", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )

# same as drop na ig?
    news = news.loc[
        news["title"].ne("") & news["text"].ne("") & news["url"].ne("")
    ].copy()

    # # normalize URLs and remove duplicates across chunks (remove any queries at the end of urls or trailing slashes wghera)
    # # urls not supposed to have ?
    # news["url"] = (
    #     news["url"]
    #         .str.split(r"[?#]", n=1, regex=True)
    #         .str[0]
    #         .str.rstrip("/")
    # )

# within chunk:
    news = news.drop_duplicates("url")

# inter chunk de duplication:
    news = news.loc[~news["url"].isin(seen_urls)].copy()
    seen_urls.update(news["url"])

    if news.empty:
        continue


# merge index values with the chunk. based on dates.
    news = pd.merge_asof(
        news.sort_values("published_date"),
        index[["Date", "market_return"]],
        left_on="published_date",
        right_on="Date",
        direction="forward", #that means index is next or same day as the news
        allow_exact_matches=False #same day excluded, next day only now.
        # last two terms ensure that index date > news date
    )

# if no return for a date, remove it.
# what if i want that news for 5 day predictions?
    news = news.dropna(subset=["Date", "market_return"])

    if news.empty:
        continue

    news.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )
    first_chunk = False