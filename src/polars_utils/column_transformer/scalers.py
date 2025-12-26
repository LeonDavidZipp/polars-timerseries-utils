from typing import Self

import polars as pl
from polars._typing import PythonLiteral

from .base import BaseColumnTransformer


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
