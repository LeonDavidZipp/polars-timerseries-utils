from typing import Self, override

import polars as pl

from .base import BaseColumnTransformer


class LagTransformer(BaseColumnTransformer):
	"""
	Creates lag features from a time series.

	This transformer shifts the series by specified lag periods, useful for
	creating features for time series forecasting models.

	Attributes:
		lags (list[int]): The lag periods to create (e.g., [1, 2, 3] for t-1, t-2, t-3).
		fill_value: The value to fill for positions where lag is not available.
	"""

	def __init__(
		self,
		lags: list[int] | int = 1,
		fill_value: float | None = None,
	):
		"""
		Initializes the LagTransformer.

		Args:
			lags: Single lag value or list of lag values. Positive values shift
				backward (past values), negative values shift forward (future values).
			fill_value: Value to fill for positions where lag creates nulls.
				If None, nulls are preserved.
		"""

		self.lags = [lags] if isinstance(lags, int) else lags
		self.fill_value = fill_value
		super().__init__()
		self.is_fitted = True

	@override
	def fit(self, s: pl.Series) -> Self:
		"""
		Fit method for compatibility. No fitting is required for lag transformation.

		Args:
			s (pl.Series): The input series to fit.

		Returns:
			Self: The fitted transformer.
		"""

		self.is_fitted = True
		return self

	@override
	def transform(self, s: pl.Series) -> pl.Series:
		"""
		Transform the series by applying the first lag only.

		For multiple lags, use transform_multi() which returns a DataFrame.

		Args:
			s (pl.Series): The input series to transform.

		Returns:
			pl.Series: The lagged series (using first lag value).
		"""

		if not self.is_fitted:
			raise ValueError(f"{self.__class__.__name__} has not been fitted yet.")

		lag = self.lags[0]
		result = s.shift(lag)

		if self.fill_value is not None:
			result = result.fill_null(self.fill_value)

		return result


class DiffTransformer(BaseColumnTransformer):
	"""
	Applies differencing to a time series for stationarity.

	Differencing computes the change between consecutive observations,
	which is useful for removing trends and achieving stationarity.

	Attributes:
		order (int): The order of differencing (1 for first-order, 2 for second-order).
		periods (int): The number of periods to shift for differencing.
	"""

	def __init__(
		self,
		order: int = 1,
		periods: int = 1,
	):
		"""
		Initializes the DiffTransformer.

		Args:
			order: The order of differencing. 1 means x[t] - x[t-1],
				2 means diff(diff(x)), etc.
			periods: The number of periods to shift. Default is 1.
				Use 12 for seasonal differencing on monthly data, etc.
		"""

		if order < 1:
			raise ValueError("Order must be at least 1.")
		if periods < 1:
			raise ValueError("Periods must be at least 1.")

		self.order = order
		self.periods = periods
		self._initial_values: list[pl.Series] = []
		super().__init__()

	@override
	def fit(self, s: pl.Series) -> Self:
		"""
		Fit the transformer by storing initial values for inverse transform.

		Args:
			s (pl.Series): The input series to fit.

		Returns:
			Self: The fitted transformer.
		"""

		self._initial_values = []
		current = s

		for _ in range(self.order):
			self._initial_values.append(current.head(self.periods))
			current = current.diff(n=self.periods)

		self.is_fitted = True
		return self

	@override
	def transform(self, s: pl.Series) -> pl.Series:
		"""
		Apply differencing to the series.

		Args:
			s (pl.Series): The input series to transform.

		Returns:
			pl.Series: The differenced series.
		"""

		if not self.is_fitted:
			raise ValueError(f"{self.__class__.__name__} has not been fitted yet.")

		result = s
		for _ in range(self.order):
			result = result.diff(n=self.periods)

		return result
