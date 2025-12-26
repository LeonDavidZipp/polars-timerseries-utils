from typing import Self, override

import polars as pl
from polars._typing import PythonLiteral

from .base import BaseColumnTransformer
from .types import RollingStrategy


class Smoother(BaseColumnTransformer):
	"""
	A class to smooth outliers in a Polars Series using the median and MAD approach.

	Attributes:
		max_zscore (float): The maximum z-score threshold to identify outliers.
	"""

	def __init__(
		self,
		max_zscore: float = 3.0,
		zero_threshold: float = 1e-5,
		fill_value: float = 1e-4,
	):
		"""
		Args:
			max_zscore (float): The maximum z-score threshold to identify outliers. Defaults to 3.0.
			zero_threshold (float): The threshold below which MAD is considered zero. Defaults to 1e-5.
			fill_value (float): The value to replace zero MADs with. Defaults to 1e-4.
		"""

		self.max_zscore = max_zscore
		self.zero_threshold = zero_threshold
		self.fill_value = fill_value
		self.median: PythonLiteral | None = None
		self.mad: PythonLiteral | None = None

	@override
	def fit(self, s: pl.Series) -> Self:
		self.median = s.median()
		mad = (s - self.median).abs().median() or 0
		if mad < self.zero_threshold:  # type: ignore
			mad = self.fill_value
		self.mad = mad
		return self

	@override
	def transform(self, s: pl.Series) -> pl.Series:
		if not all([self.median is not None, self.mad is not None]):
			raise ValueError(f"{self.__class__.__name__} has not been fit yet.")

		z_score = "z_score"
		temp_df = (
			pl.DataFrame({s.name: s})
			.with_columns(
				((pl.col(s.name) - self.median) / (self.mad * 1.4826 + 1e-8)).alias(  # type: ignore
					z_score
				)
			)
			.with_columns(
				pl.when(pl.col(z_score) >= self.max_zscore)
				.then(self.median + self.max_zscore * self.mad)  # type: ignore
				.when(pl.col(z_score) <= -self.max_zscore)
				.then(self.median - self.max_zscore * self.mad)  # type: ignore
				.otherwise(pl.col(s.name))
				.cast(s.dtype)
				.alias(s.name)
			)
		)
		return temp_df.select(s.name).to_series()


def rolling_zscore(
	df: pl.DataFrame | pl.LazyFrame,
	col: str,
	window_size: int,
	min_samples: int = 1,
	center: bool = False,
	zero_threshold: float = 1e-5,
	fill_value: float = 1e-4,
	alias: str = "z_score",
	with_median: RollingStrategy | None = None,
	with_mad: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
	"""
	Calculate the rolling z-score of a Polars Series.

	Args:
		df (pl.DataFrame | pl.LazyFrame): The input series.
		col (str): The name of the column to calculate the z-score for.
		window_size (int): The size of the rolling window.
		min_samples (int): The minimum number of samples required in the window to compute the statistics.
		center (bool): Whether to set the labels at the center of the window.
		zero_threshold (float): The threshold below which MAD is considered zero.
		fill_value (float): The value to replace zero MADs with.
		alias (str): The name of the output z-score column.
		with_median (str | None): If provided, include the rolling median column with this name in the output.
		with_mad (str | None): If provided, include the rolling MAD column with this name in the output.

	Returns:
		pl.DataFrame | pl.LazyFrame: The DataFrame with the rolling z-score column added.
	"""

	mad = "mad"
	cols = (
		df.collect_schema().names()
		+ [alias]
		+ ([with_median] if with_median else [])
		+ ([with_mad] if with_mad else [])
	)
	with_median = with_median if with_median else RollingStrategy.MEDIAN
	return (
		df.with_columns(
			pl.col(col)
			.rolling_median(window_size, min_samples=min_samples, center=center)
			.alias(with_median)
		)
		.with_columns(
			(pl.col(col) - pl.col(with_median))
			.abs()
			.rolling_median(window_size, min_samples=min_samples, center=center)
			.alias(mad)
		)
		.with_columns(
			pl.when(pl.col(mad).is_between(-zero_threshold, zero_threshold))
			.then(fill_value)
			.otherwise(pl.col(mad))
		)
		.with_columns(
			((pl.col(col) - pl.col(with_median)) / (pl.col(mad) * 1.4826 + 1e-8)).alias(
				alias
			)
		)
		.select(cols)
	)


class RollingSmoother(BaseColumnTransformer):
	"""
	A class to smooth outliers in a Polars Series using a rolling median and MAD approach.

	Attributes:
		window_size (int): The size of the rolling window.
		min_samples (int): The minimum number of samples required in the window to compute the statistics.
		max_zscore (float): The maximum z-score threshold to identify outliers.
		center (bool): Whether to set the labels at the center of the window.
	"""

	def __init__(
		self,
		window_size: int,
		min_samples: int = 1,
		max_zscore: float = 3.0,
		zero_threshold: float = 1e-5,
		fill_value: float = 1e-4,
		center: bool = False,
	):
		"""
		Args:
			window_size (int): The size of the rolling window.
			min_samples (int | None): The minimum number of samples required in the window to compute the statistics. Defaults to 1.
			max_zscore (float): The maximum z-score threshold to identify outliers. Defaults to 3.0.
			zero_threshold (float): The threshold below which MAD is considered zero. Defaults to 1e-5.
			fill_value (float): The value to replace zero MADs with. Defaults to 1e-4.
			center (bool): Whether to set the labels at the center of the window. Default is False, because it is safe for time series data.
		"""

		self.window_size = window_size
		self.min_samples = min_samples or window_size
		self.max_zscore = max_zscore
		self.zero_threshold = zero_threshold
		self.fill_value = fill_value
		self.center = center

	@override
	def fit(self, s: pl.Series) -> Self:
		"""
		Fit method for compatibility. No fitting is required for rolling smoothing.

		Args:
			s (pl.Series): The input series to fit.

		Returns:
			Self: The fitted smoother.
		"""

		return self

	@override
	def transform(self, s: pl.Series) -> pl.Series:
		mad = "mad"
		z_score = "z_score"
		temp_df = rolling_zscore(
			df=pl.LazyFrame({s.name: s}),
			col=s.name,
			window_size=self.window_size,
			min_samples=self.min_samples,
			center=self.center,
			zero_threshold=self.zero_threshold,
			fill_value=self.fill_value,
			alias=z_score,
			with_median=RollingStrategy.MEDIAN,
			with_mad=mad,
		).with_columns(
			pl.when(pl.col(z_score) >= self.max_zscore)
			.then(pl.col(RollingStrategy.MEDIAN) + self.max_zscore * pl.col(mad))
			.when(pl.col(z_score) <= -self.max_zscore)
			.then(pl.col(RollingStrategy.MEDIAN) - self.max_zscore * pl.col(mad))
			.otherwise(pl.col(s.name))
			.cast(s.dtype)
			.alias(s.name)
		)
		return temp_df.select(s.name).lazy().collect().to_series()
