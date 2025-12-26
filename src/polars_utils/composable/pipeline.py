from typing import Self, override

import polars as pl

from .base import BaseMultiColumnTransformer


class Pipeline(BaseMultiColumnTransformer):
	def __init__(self, steps: list[BaseMultiColumnTransformer]) -> None:
		self.steps = steps

	@override
	def fit(self, df: pl.DataFrame | pl.LazyFrame) -> Self:
		for step in self.steps:
			df = step.fit_transform(df)
		return self

	@override
	def transform(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
		for step in self.steps:
			df = step.transform(df)
		return df
