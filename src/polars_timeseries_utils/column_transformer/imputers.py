from typing import Any, Self, override

import polars as pl
from polars._typing import PythonLiteral

from .base import BaseColumnTransformer
from .types import RollingStrategy, Strategy


class Imputer(BaseColumnTransformer):
	"""
	Simple imputer for handling missing values in a Polars Series.

	Attributes:
		value: The value to replace missing values with.
		strategy: The imputation strategy to use.
		fitted_value: The computed value during fit (for statistical strategies).
	"""

	def __init__(
		self,
		value: Any | None = None,
		strategy: Strategy | None = None,
	):
		"""
		Initialize the imputer with a value or strategy.

		Args:
			value: The value to replace missing values with. If provided, 'strategy' must be None.
			strategy: The imputation strategy to use. If provided, 'value' must be None.
		"""

		if value is not None and strategy is not None:
			raise ValueError("Exactly one parameter must be not None.")
		elif value is None and strategy is None:
			raise ValueError("Exactly one parameter must be not None.")
		self.strategy = strategy
		self.value = value
		self.fitted_value: PythonLiteral | None = None

	@override
	def fit(self, s: pl.Series) -> Self:
		match self.strategy:
			case None | Strategy.FORWARD | Strategy.BACKWARD:
				return self
			case Strategy.MEDIAN:
				self.fitted_value = s.median()
			case Strategy.MEAN:
				self.fitted_value = s.mean()
			case Strategy.MIN:
				self.fitted_value = s.min()
			case Strategy.MAX:
				self.fitted_value = s.max()
			case Strategy.ZERO:
				if s.dtype == pl.String or s.dtype == pl.Categorical:
					self.fitted_value = "0"
				else:
					self.fitted_value = 0
			case Strategy.ONE:
				if s.dtype == pl.String or s.dtype == pl.Categorical:
					self.fitted_value = "1"
				else:
					self.fitted_value = 1
		return self

	@override
	def transform(self, s: pl.Series) -> pl.Series:
		if self.value is not None:
			fill_val = self.value
			temp_df = pl.DataFrame({s.name: s}).with_columns(
				pl.col(s.name).fill_null(value=fill_val)
			)
		elif self.strategy in [Strategy.FORWARD, Strategy.BACKWARD]:
			temp_df = pl.DataFrame({s.name: s}).with_columns(
				pl.col(s.name).fill_null(strategy=self.strategy)  # type: ignore
			)
		else:
			if self.fitted_value is None:
				raise ValueError(f"{self.__class__.__name__} has not been fit yet.")
			temp_df = pl.DataFrame({s.name: s}).with_columns(
				pl.col(s.name).fill_null(value=self.fitted_value)
			)

		return temp_df.select(pl.col(s.name).cast(s.dtype)).to_series()


class RollingImputer(BaseColumnTransformer):
	"""
	Impute missing values using rolling statistics.

	Attributes:
		strategy (Literal): The rolling statistic to use ('rolling_min', 'rolling_max',
			'rolling_mean', 'rolling_median').
		window_size (int): The size of the rolling window.
		weights (list[float] | None): The weights for the rolling mean. Only used if strategy is 'rolling_mean'.
		min_samples (int | None): The minimum number of samples required in the window to compute the statistic.
			If None, it defaults to window_size.
		center (bool): Whether to set the labels at the center of the window.
	"""

	def __init__(
		self,
		window_size: int,
		strategy: RollingStrategy = RollingStrategy.MEAN,
		min_samples: int = 1,
		weights: list[float] | None = None,
		center: bool = False,
	):
		self.strategy = strategy
		self.window_size = window_size
		self.weights = weights
		self.min_samples = min_samples or window_size
		self.center = center

	@override
	def fit(self, s: pl.Series) -> Self:
		"""
		Fit method for compatibility. No fitting is required for rolling imputation.

		Args:
			s (pl.Series): The input series to fit.

		Returns:
			Self: The fitted imputer.
		"""

		return self

	@override
	def transform(self, s: pl.Series) -> pl.Series:
		stat = "stat"
		match self.strategy:
			case RollingStrategy.MIN:
				temp_df = pl.LazyFrame({s.name: s}).with_columns(
					pl.col(s.name)
					.rolling_min(
						self.window_size,
						min_samples=self.min_samples,
						center=self.center,
					)
					.alias(stat)
				)
			case RollingStrategy.MAX:
				temp_df = pl.LazyFrame({s.name: s}).with_columns(
					pl.col(s.name)
					.rolling_max(
						self.window_size,
						min_samples=self.min_samples,
						center=self.center,
					)
					.alias(stat)
				)
			case RollingStrategy.MEAN:
				temp_df = pl.LazyFrame({s.name: s}).with_columns(
					pl.col(s.name)
					.rolling_mean(
						self.window_size,
						min_samples=self.min_samples,
						center=self.center,
					)
					.alias(stat)
				)
			case RollingStrategy.MEDIAN:
				temp_df = pl.LazyFrame({s.name: s}).with_columns(
					pl.col(s.name)
					.rolling_median(
						self.window_size,
						min_samples=self.min_samples,
						center=self.center,
					)
					.alias(stat)
				)

		temp_df = temp_df.with_columns(
			pl.coalesce(pl.col(s.name), pl.col(stat))
			.interpolate("linear")
			.fill_null(strategy="forward")
			.fill_null(0)
			.cast(s.dtype)
			.alias(s.name)
		)

		return temp_df.select(s.name).collect().to_series()


class MinMaxScaler(BaseColumnTransformer):
	"""
	Scales a Polars Series to a given range [0, 1] using Min-Max scaling.

	Attributes:
		min (float | None): The minimum value of the series.
		max (float | None): The maximum value of the series.
	"""

	def __init__(self):
		"""
		Initializes the MinMaxScaler.
		"""

		self.min: PythonLiteral | None = None
		self.max: PythonLiteral | None = None

	def fit(self, s: pl.Series) -> Self:
		self.min = s.min()
		self.max = s.max()
		print(f"Fitted MinMaxScaler with min: {self.min}, max: {self.max}")
		return self

	def transform(self, s: pl.Series) -> pl.Series:
		if not all([self.min is not None, self.max is not None]):
			raise ValueError("MinMaxScaler has not been fitted yet.")

		temp_df = pl.DataFrame({s.name: s}).with_columns(
			pl.when(pl.col(s.name).is_null())
			.then(None)
			.otherwise((pl.col(s.name) - self.min) / (self.max - self.min + 1e-8))  # type: ignore
			.cast(s.dtype)
			.alias(s.name)
		)
		print(f"Transformed series with MinMaxScaler: {temp_df.head(5)}")
		return temp_df.select(s.name).to_series()


class StandardScaler(BaseColumnTransformer):
	"""
	Scales a Polars Series to have zero mean and unit variance using Standard scaling.
	"""

	def __init__(self):
		"""
		Initializes the StandardScaler.
		"""

		self.mean: PythonLiteral | None = None
		self.std: PythonLiteral | None = None

	def fit(self, s: pl.Series) -> Self:
		self.mean = s.mean()
		self.std = s.std()
		return self

	def transform(self, s: pl.Series) -> pl.Series:
		if not all([self.mean is not None, self.std is not None]):
			raise ValueError("StandardScaler has not been fitted yet.")

		temp_df = pl.DataFrame({s.name: s}).with_columns(
			((pl.col(s.name) - self.mean) / self.std).cast(s.dtype).alias(s.name)
		)
		return temp_df.select(s.name).to_series()
