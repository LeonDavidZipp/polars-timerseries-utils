import polars as pl

from ..column_transformer.types import RollingStrategy


def rolling_zscore(
	df: pl.DataFrame | pl.LazyFrame,
	col: str,
	window_size: int,
	min_samples: int = 1,
	center: bool = False,
	zero_threshold: float = 1e-5,
	fill_value: float = 1e-4,
	alias: str = "z_score",
	with_median: str | None = None,
	with_mad: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
	"""
	Calculate the rolling z-score of a Polars Series.

	Args:
		df (pl.DataFrame | pl.LazyFrame): The input DataFrame.
		col (str): The name of the column to calculate the z-score for.
		window_size (int): The size of the rolling window.
		min_samples (int): The minimum number of samples required in the window to
			compute the statistics.
		center (bool): Whether to set the labels at the center of the window.
		zero_threshold (float): The threshold below which MAD is considered zero.
		fill_value (float): The value to replace zero MADs with.
		alias (str): The name of the output z-score column.
		with_median (str | None): If provided, include the rolling median column with
			this name in the output.
		with_mad (str | None): If provided, include the rolling MAD column with this
			name in the output.

	Returns:
		pl.DataFrame | pl.LazyFrame: The DataFrame with the rolling z-score column
			added.
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
