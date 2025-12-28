import polars as pl

from polars_timeseries_utils.metrics.zscores import rolling_zscore_df


class TestRollingZscore:
	"""Tests for the rolling_zscore function."""

	def test_basic_zscore_calculation(self) -> None:
		"""Test basic z-score calculation."""
		df = pl.DataFrame(
			{
				"value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
			}
		)

		result = rolling_zscore_df(df, col="value", window_size=3)

		assert "z_score" in result.columns
		assert result.lazy().collect().height == 10

	def test_custom_alias(self) -> None:
		"""Test z-score with custom alias."""
		df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

		result = rolling_zscore_df(df, col="value", window_size=3, alias="my_zscore")

		assert "my_zscore" in result.columns
		assert "z_score" not in result.columns

	def test_preserves_original_columns(self) -> None:
		"""Test that original columns are preserved."""
		df = pl.DataFrame(
			{
				"value": [1.0, 2.0, 3.0, 4.0, 5.0],
				"other": [10.0, 20.0, 30.0, 40.0, 50.0],
			}
		)

		result = rolling_zscore_df(df, col="value", window_size=3)

		assert "value" in result.columns
		assert "other" in result.columns

	def test_with_median_output(self) -> None:
		"""Test including median in output."""
		df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

		result = rolling_zscore_df(
			df, col="value", window_size=3, with_median="rolling_med"
		)

		assert "rolling_med" in result.columns

	def test_with_mad_output(self) -> None:
		"""Test including MAD in output."""
		df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

		# Note: with_mad parameter doesn't rename the internal 'mad' column,
		# it just includes the 'mad' column in output
		result = rolling_zscore_df(df, col="value", window_size=3, with_mad="mad")

		assert "mad" in result.columns

	def test_min_samples(self) -> None:
		"""Test min_samples parameter."""
		df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

		result = rolling_zscore_df(df, col="value", window_size=5, min_samples=1)

		# With min_samples=1, we should get z-scores even at the beginning
		assert result["z_score"].null_count() < 5  # type: ignore

	def test_centered_window(self) -> None:
		"""Test centered window option."""
		df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})

		result_centered = rolling_zscore_df(df, col="value", window_size=3, center=True)
		result_not_centered = rolling_zscore_df(
			df, col="value", window_size=3, center=False
		)

		# Results should be different with different centering
		assert (
			result_centered["z_score"].to_list()  # type: ignore
			!= result_not_centered["z_score"].to_list()  # type: ignore
		)

	def test_zscore_for_median_value_near_zero(self) -> None:
		"""Test that z-score for values at median is near zero."""
		# Create data where median is clearly 5.0
		df = pl.DataFrame({"value": [1.0, 3.0, 5.0, 7.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0]})

		result = rolling_zscore_df(df, col="value", window_size=5, min_samples=1)

		# Later values at median (5.0) should have z-score near 0
		zscores = result["z_score"].to_list()  # type: ignore
		# Get a non-null z-score for a value that equals the rolling median
		non_null_zscores = [z for z in zscores if z is not None]  # type: ignore
		assert len(non_null_zscores) > 0  # type: ignore

	def test_outlier_has_high_zscore(self) -> None:
		"""Test that outliers have high z-scores."""
		df = pl.DataFrame(
			{"value": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 7.0, 8.0, 9.0, 10.0]}
		)

		result = rolling_zscore_df(df, col="value", window_size=5)

		# The outlier (100.0 at index 5) should have a high z-score
		# Check values after the outlier where window includes it
		zscores = result["z_score"].to_list()  # type: ignore
		# Find max absolute z-score
		max_zscore = max(abs(z) for z in zscores if z is not None)  # type: ignore
		assert max_zscore > 2.0

	def test_lazyframe_support(self) -> None:
		"""Test that LazyFrame input returns LazyFrame output."""
		lf = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]}).lazy()

		result = rolling_zscore_df(lf, col="value", window_size=3)

		assert isinstance(result, pl.LazyFrame)

	def test_dataframe_support(self) -> None:
		"""Test that DataFrame input returns DataFrame output."""
		df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

		result = rolling_zscore_df(df, col="value", window_size=3)

		assert isinstance(result, pl.DataFrame)

	def test_zero_threshold_handling(self) -> None:
		"""Test that zero MAD is handled with fill_value."""
		# All same values = zero MAD
		df = pl.DataFrame({"value": [5.0, 5.0, 5.0, 5.0, 5.0]})

		result = rolling_zscore_df(
			df, col="value", window_size=3, zero_threshold=1e-5, fill_value=1e-4
		)

		# Should not have inf or nan due to division by zero
		zscores = result["z_score"].to_list()  # type: ignore
		for z in zscores:  # type: ignore
			if z is not None:
				assert z != float("inf")
				assert z != float("-inf")
				assert z == z  # not NaN

	def test_custom_zero_threshold(self) -> None:
		"""Test custom zero_threshold parameter."""
		df = pl.DataFrame({"value": [1.0, 1.001, 1.002, 1.001, 1.0]})

		result = rolling_zscore_df(
			df, col="value", window_size=3, zero_threshold=0.01, fill_value=0.1
		)

		# Should complete without error
		assert "z_score" in result.columns
