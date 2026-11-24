import unittest

import numpy as np
import pandas as pd

from scripts.news_research import START_TEST, END_TEST, load_daily, make_features, pool_sessions


class SessionAlignmentTests(unittest.TestCase):
    def test_pooling_reconstructs_the_article_level_mean_and_dispersion(self):
        daily = pd.DataFrame({
            "published_date": pd.to_datetime(["2023-01-06", "2023-01-07"]),
            "market_date": pd.to_datetime(["2023-01-09", "2023-01-09"]),
            "article_count": [2, 1],
            "sentiment_mean": [0.0, 1.0],
            "sentiment_std": [0.2, 0.0],
            "market_return": [0.01, 0.01],
        })
        session = pool_sessions(daily).iloc[0]
        actual_scores = [-0.2, 0.2, 1.0]
        self.assertEqual(session["articles"], len(actual_scores))
        self.assertAlmostEqual(session["sentiment"], np.mean(actual_scores))
        self.assertAlmostEqual(session["dispersion"], np.std(actual_scores))

    def test_target_session_news_cannot_change_its_forecast_features(self):
        sessions = pool_sessions(load_daily())
        features = make_features(sessions)
        target = pd.Timestamp("2023-01-09")
        cutoff = pd.Timestamp("2023-01-06")
        self.assertEqual(features.loc[target, "cutoff_session"], cutoff)
        self.assertEqual(features.loc[target, "available_last_news_date"], pd.Timestamp("2023-01-05"))
        self.assertEqual(features.loc[target, "previous_return"], sessions.loc[cutoff, "market_return"])
        changed = sessions.copy()
        changed.loc[target, "sentiment"] += 1
        changed.loc[target, "market_return"] += 1
        updated = make_features(changed)
        self.assertEqual(updated.loc[target, "available_sentiment"], features.loc[target, "available_sentiment"])
        self.assertEqual(updated.loc[target, "previous_return"], features.loc[target, "previous_return"])

    def test_every_test_row_has_unique_target_and_eligible_news(self):
        features = make_features(pool_sessions(load_daily())).loc[START_TEST:END_TEST]
        self.assertTrue(features.index.is_unique)
        self.assertTrue(features["available_last_news_date"].lt(features["cutoff_session"]).all())
        self.assertTrue(features["cutoff_session"].lt(features.index).all())
        self.assertEqual(features.index.max(), pd.Timestamp("2023-12-29"))


if __name__ == "__main__":
    unittest.main()
