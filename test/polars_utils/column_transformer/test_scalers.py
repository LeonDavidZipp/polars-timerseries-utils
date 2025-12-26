import polars as pl
import pytest

from polars_timeseries_utils.column_transformer import MinMaxScaler, StandardScaler


class TestMinMaxScaler:
	"""Tests for the MinMaxScaler class."""

	def test_init(self) -> None:
		"""Test initialization."""
		scaler = MinMaxScaler()
		assert scaler.min is None
		assert scaler.max is None

	def test_fit(self, series_for_scaling: pl.Series) -> None:
		"""Test fitting the scaler."""
		scaler = MinMaxScaler()
		scaler.fit(series_for_scaling)

		assert scaler.min == 0.0
		assert scaler.max == 100.0

	def test_transform_scales_to_0_1(self, series_for_scaling: pl.Series) -> None:
		"""Test that values are scaled to [0, 1] range."""
		scaler = MinMaxScaler()
		result = scaler.fit_transform(series_for_scaling)

		# [0, 25, 50, 75, 100] -> [0.0, 0.25, 0.5, 0.75, 1.0] (approximately)
		assert result[0] == pytest.approx(0.0, abs=0.01)
		assert result[2] == pytest.approx(0.5, abs=0.01)
		assert result[4] == pytest.approx(1.0, abs=0.01)

	def test_transform_negative_values(
		self, series_with_negative_values: pl.Series
	) -> None:
		"""Test scaling with negative values."""
		scaler = MinMaxScaler()
		result = scaler.fit_transform(series_with_negative_values)

		# [-50, -25, 0, 25, 50] -> should scale to approximately [0, 0.25, 0.5, 0.75, 1]
		assert result[0] == pytest.approx(0.0, abs=0.01)
		assert result[2] == pytest.approx(0.5, abs=0.01)
		assert result[4] == pytest.approx(1.0, abs=0.01)

	def test_transform_without_fit_raises(self, series_for_scaling: pl.Series) -> None:
		"""Test that transform without fit raises."""
		scaler = MinMaxScaler()
		with pytest.raises(ValueError, match="has not been fitted"):
			scaler.transform(series_for_scaling)

	def test_preserves_dtype(self, series_for_scaling: pl.Series) -> None:
		"""Test that the original dtype is preserved."""
		scaler = MinMaxScaler()
		result = scaler.fit_transform(series_for_scaling)

		assert result.dtype == series_for_scaling.dtype

	def test_separate_fit_transform(self, series_for_scaling: pl.Series) -> None:
		"""Test separate fit and transform calls."""
		scaler = MinMaxScaler()
		scaler.fit(series_for_scaling)
		result = scaler.transform(series_for_scaling)

		assert result[0] == pytest.approx(0.0, abs=0.01)
		assert result[4] == pytest.approx(1.0, abs=0.01)

	def test_transform_new_data(self, series_for_scaling: pl.Series) -> None:
		"""Test transforming new data with fitted scaler."""
		scaler = MinMaxScaler()
		scaler.fit(series_for_scaling)

		# New data with same scale
		new_series = pl.Series("value", [50.0, 75.0])
		result = scaler.transform(new_series)

		assert result[0] == pytest.approx(0.5, abs=0.01)
		assert result[1] == pytest.approx(0.75, abs=0.01)


class TestStandardScaler:
	"""Tests for the StandardScaler class."""

	def test_init(self) -> None:
		"""Test initialization."""
		scaler = StandardScaler()
		assert scaler.mean is None
		assert scaler.std is None

	def test_fit(self, series_for_scaling: pl.Series) -> None:
		"""Test fitting the scaler."""
		scaler = StandardScaler()
		scaler.fit(series_for_scaling)

		# [0, 25, 50, 75, 100] -> mean = 50
		assert scaler.mean == 50.0
		assert scaler.std is not None
		assert scaler.std > 0

	def test_transform_centers_mean(self, series_for_scaling: pl.Series) -> None:
		"""Test that mean is approximately 0 after transform."""
		scaler = StandardScaler()
		result = scaler.fit_transform(series_for_scaling)

		# Mean should be approximately 0
		assert result.mean() == pytest.approx(0.0, abs=0.01)

	def test_transform_without_fit_raises(self, series_for_scaling: pl.Series) -> None:
		"""Test that transform without fit raises."""
		scaler = StandardScaler()
		with pytest.raises(ValueError, match="has not been fitted"):
			scaler.transform(series_for_scaling)

	def test_preserves_dtype(self, series_for_scaling: pl.Series) -> None:
		"""Test that the original dtype is preserved."""
		scaler = StandardScaler()
		result = scaler.fit_transform(series_for_scaling)

		assert result.dtype == series_for_scaling.dtype

	def test_separate_fit_transform(self, series_for_scaling: pl.Series) -> None:
		"""Test separate fit and transform calls."""
		scaler = StandardScaler()
		scaler.fit(series_for_scaling)
		result = scaler.transform(series_for_scaling)

		assert result.mean() == pytest.approx(0.0, abs=0.01)

	def test_transform_new_data(self, series_for_scaling: pl.Series) -> None:
		"""Test transforming new data with fitted scaler."""
		scaler = StandardScaler()
		scaler.fit(series_for_scaling)

		# Transform the mean value should give approximately 0
		new_series = pl.Series("value", [50.0])
		result = scaler.transform(new_series)

		assert result[0] == pytest.approx(0.0, abs=0.01)
