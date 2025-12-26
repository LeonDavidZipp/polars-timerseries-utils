import polars as pl
import pytest

from polars_timeseries_utils.column_transformer import Imputer, MinMaxScaler
from polars_timeseries_utils.column_transformer.types import Strategy
from polars_timeseries_utils.composable import (
	ColumnTransformerMetadata,
	MultiColumnTransformer,
	MultiColumnTransformerMetadata,
	Pipeline,
)


class TestMultiColumnTransformer:
	"""Tests for the MultiColumnTransformer class."""

	def test_init(self) -> None:
		"""Test initialization."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a", "col_b"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)

		assert len(mct.transformers) == 1
		assert mct.is_fitted is False

	def test_init_no_transformers(self) -> None:
		"""Test initialization with no transformers."""
		with pytest.raises(
			ValueError,
			match=f"{MultiColumnTransformer.__name__} must have at least one transformer.",
		):
			MultiColumnTransformer([])

	def test_fit(self, df_multi_column: pl.DataFrame) -> None:
		"""Test fitting the transformer."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a", "col_b"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)
		mct.fit(df_multi_column)

		assert mct.is_fitted is True
		assert len(mct.col_to_transformer) == 2
		assert "col_a" in mct.col_to_transformer
		assert "col_b" in mct.col_to_transformer

	def test_transform_fills_nulls(self, df_multi_column: pl.DataFrame) -> None:
		"""Test that transform fills nulls in specified columns."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a", "col_b"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)
		result = mct.fit_transform(df_multi_column)

		# Check nulls are filled
		assert result["col_a"].null_count() == 0
		assert result["col_b"].null_count() == 0

	def test_transform_without_fit_raises(self, df_multi_column: pl.DataFrame) -> None:
		"""Test that transform without fit raises."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)

		with pytest.raises(RuntimeError, match="must be fitted"):
			mct.transform(df_multi_column)

	def test_preserves_column_order(self, df_multi_column: pl.DataFrame) -> None:
		"""Test that column order is preserved."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a", "col_b"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)
		result = mct.fit_transform(df_multi_column)

		assert result.columns == df_multi_column.columns

	def test_preserves_shape(self, df_multi_column: pl.DataFrame) -> None:
		"""Test that shape is preserved."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a", "col_b"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)
		result = mct.fit_transform(df_multi_column)

		assert result.shape == df_multi_column.shape

	def test_untransformed_columns_unchanged(
		self, df_multi_column: pl.DataFrame
	) -> None:
		"""Test that columns not in transformers are unchanged."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)
		result = mct.fit_transform(df_multi_column)

		# col_c should be unchanged
		assert result["col_c"].to_list() == df_multi_column["col_c"].to_list()

	def test_by_dtype_selection(self, df_multi_column: pl.DataFrame) -> None:
		"""Test selecting columns by dtype."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=[pl.Float64],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)
		mct.fit(df_multi_column)

		# Should have transformers for all float columns
		float_cols = [
			c for c, dtype in df_multi_column.schema.items() if dtype == pl.Float64
		]
		for col in float_cols:
			assert col in mct.col_to_transformer

	def test_get_transformer(self, df_multi_column: pl.DataFrame) -> None:
		"""Test getting transformer for a column."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)
		mct.fit(df_multi_column)

		tf = mct.get_transformer("col_a")
		assert tf is not None
		assert isinstance(tf, Imputer)

		tf_none = mct.get_transformer("nonexistent")
		assert tf_none is None

	def test_lazyframe_support(self, lf_multi_column: pl.LazyFrame) -> None:
		"""Test that LazyFrame is supported."""
		transformers = [
			ColumnTransformerMetadata(
				name="imputer",
				columns=["col_a", "col_b"],
				transformer=Imputer(strategy=Strategy.MEAN),
			)
		]
		mct = MultiColumnTransformer(transformers)
		result = mct.fit_transform(lf_multi_column)

		assert isinstance(result, pl.LazyFrame)

		# Collect and check nulls are filled
		collected = result.collect()
		assert collected["col_a"].null_count() == 0
		assert collected["col_b"].null_count() == 0


class TestPipeline:
	"""Tests for the Pipeline class."""

	def test_init(self) -> None:
		"""Test initialization."""
		step1 = MultiColumnTransformer(
			[
				ColumnTransformerMetadata(
					name="imputer",
					columns=["col_a"],
					transformer=Imputer(strategy=Strategy.MEAN),
				)
			]
		)
		pipeline = Pipeline([step1])

		assert len(pipeline.steps) == 1

	def test_single_step(self, df_multi_column: pl.DataFrame) -> None:
		"""Test pipeline with single step."""
		step1 = MultiColumnTransformer(
			[
				ColumnTransformerMetadata(
					name="imputer",
					columns=["col_a", "col_b"],
					transformer=Imputer(strategy=Strategy.MEAN),
				)
			]
		)
		steps = [MultiColumnTransformerMetadata(name="step1", transformer=step1)]
		pipeline = Pipeline(steps)
		result = pipeline.fit_transform(df_multi_column)

		assert result["col_a"].null_count() == 0
		assert result["col_b"].null_count() == 0

	def test_multi_step(self, df_multi_column: pl.DataFrame) -> None:
		"""Test pipeline with multiple steps."""
		step1 = MultiColumnTransformer(
			[
				ColumnTransformerMetadata(
					name="imputer",
					columns=["col_a", "col_b"],
					transformer=Imputer(strategy=Strategy.MEAN),
				)
			]
		)
		step2 = MultiColumnTransformer(
			[
				ColumnTransformerMetadata(
					name="scaler",
					columns=["col_c"],
					transformer=MinMaxScaler(),
				)
			]
		)
		steps = [
			MultiColumnTransformerMetadata(name="step1", transformer=step1),
			MultiColumnTransformerMetadata(name="step2", transformer=step2),
		]
		pipeline = Pipeline(steps)
		result = pipeline.fit_transform(df_multi_column)

		# Check imputation happened
		assert result["col_a"].null_count() == 0
		assert result["col_b"].null_count() == 0

		# Check scaling happened (col_c should be in [0, 1] range approximately)
		assert result["col_c"].min() >= -0.1
		assert result["col_c"].max() <= 1.1

	def test_preserves_shape(self, df_multi_column: pl.DataFrame) -> None:
		"""Test that shape is preserved through pipeline."""
		step1 = MultiColumnTransformer(
			[
				ColumnTransformerMetadata(
					name="imputer",
					columns=["col_a"],
					transformer=Imputer(strategy=Strategy.MEAN),
				)
			]
		)
		steps = [MultiColumnTransformerMetadata(name="step1", transformer=step1)]
		pipeline = Pipeline(steps)
		result = pipeline.fit_transform(df_multi_column)

		assert result.shape == df_multi_column.shape

	def test_preserves_columns(self, df_multi_column: pl.DataFrame) -> None:
		"""Test that columns are preserved through pipeline."""
		step1 = MultiColumnTransformer(
			[
				ColumnTransformerMetadata(
					name="imputer",
					columns=["col_a"],
					transformer=Imputer(strategy=Strategy.MEAN),
				)
			]
		)
		steps = [MultiColumnTransformerMetadata(name="step1", transformer=step1)]
		pipeline = Pipeline(steps)
		result = pipeline.fit_transform(df_multi_column)

		assert result.columns == df_multi_column.columns

	def test_lazyframe_support(self, lf_multi_column: pl.LazyFrame) -> None:
		"""Test that LazyFrame is supported."""
		step1 = MultiColumnTransformer(
			[
				ColumnTransformerMetadata(
					name="imputer",
					columns=["col_a", "col_b"],
					transformer=Imputer(strategy=Strategy.MEAN),
				)
			]
		)
		steps = [MultiColumnTransformerMetadata(name="step1", transformer=step1)]
		pipeline = Pipeline(steps)
		result = pipeline.fit_transform(lf_multi_column)

		assert isinstance(result, pl.LazyFrame)

		collected = result.collect()
		assert collected["col_a"].null_count() == 0
