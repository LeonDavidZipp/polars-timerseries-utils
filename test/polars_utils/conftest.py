from datetime import datetime

import polars as pl
import pytest

# =============================================================================
# CLEAN DATAFRAMES (no missing values, proper types)
# =============================================================================


@pytest.fixture
def df_clean() -> pl.DataFrame:
	"""Clean DataFrame with datetime timestamp column named 'timestamp' and numeric columns."""
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_clean_date_col() -> pl.DataFrame:
	"""Clean DataFrame with datetime column named 'date'."""
	return pl.DataFrame(
		{
			"date": [datetime(2023, 1, i) for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"y": [float(i * 5 + 3) for i in range(1, 21)],
			"extra": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_clean_datetime_col() -> pl.DataFrame:
	"""Clean DataFrame with datetime column named 'datetime'."""
	return pl.DataFrame(
		{
			"datetime": [datetime(2023, 1, i, 12, 30, 0) for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_clean_ds_col() -> pl.DataFrame:
	"""Clean DataFrame with datetime column named 'ds' (Prophet format)."""
	return pl.DataFrame(
		{
			"ds": [datetime(2023, 1, i) for i in range(1, 21)],
			"y": [float(i * 10) for i in range(1, 21)],
			"value": [float(i * 5 + 3) for i in range(1, 21)],
			"extra": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_clean_time_col() -> pl.DataFrame:
	"""Clean DataFrame with datetime column named 'time'."""
	return pl.DataFrame(
		{
			"time": [datetime(2023, 1, i) for i in range(1, 21)],
			"values": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_clean_unordered() -> pl.DataFrame:
	"""Clean DataFrame with unordered timestamps."""
	import random

	indices = list(range(1, 21))
	random.seed(42)
	random.shuffle(indices)
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in indices],
			"value": [float(i * 10) for i in indices],
			"measurement": [float(i * 5 + 3) for i in indices],
			"target": [float(i * 2 - 1) for i in indices],
		}
	)


@pytest.fixture
def lf_clean() -> pl.LazyFrame:
	"""Clean LazyFrame with datetime timestamp column."""
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	).lazy()


# =============================================================================
# UNCLEAN DATAFRAMES - String timestamps that need casting
# =============================================================================


@pytest.fixture
def df_unclean_timestamp_str_ymd_dash() -> pl.DataFrame:
	"""DataFrame with string timestamp in YYYY-MM-DD format."""
	return pl.DataFrame(
		{
			"timestamp": [f"2023-01-{i:02d}" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_timestamp_str_dmy_dash() -> pl.DataFrame:
	"""DataFrame with string timestamp in DD-MM-YYYY format."""
	return pl.DataFrame(
		{
			"timestamp": [f"{i:02d}-01-2023" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_timestamp_str_ymd_slash() -> pl.DataFrame:
	"""DataFrame with string timestamp in YYYY/MM/DD format."""
	return pl.DataFrame(
		{
			"timestamp": [f"2023/01/{i:02d}" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_timestamp_str_dmy_slash() -> pl.DataFrame:
	"""DataFrame with string timestamp in DD/MM/YYYY format."""
	return pl.DataFrame(
		{
			"timestamp": [f"{i:02d}/01/2023" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_timestamp_str_datetime_ymd_dash() -> pl.DataFrame:
	"""DataFrame with string timestamp in YYYY-MM-DD HH:MM:SS format."""
	return pl.DataFrame(
		{
			"timestamp": [f"2023-01-{i:02d} 12:30:00" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_timestamp_str_datetime_ymd_slash() -> pl.DataFrame:
	"""DataFrame with string timestamp in YYYY/MM/DD HH:MM:SS format."""
	return pl.DataFrame(
		{
			"timestamp": [f"2023/01/{i:02d} 12:30:00" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_timestamp_str_datetime_dmy_dash() -> pl.DataFrame:
	"""DataFrame with string timestamp in DD-MM-YYYY HH:MM:SS format."""
	return pl.DataFrame(
		{
			"timestamp": [f"{i:02d}-01-2023 12:30:00" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_timestamp_str_datetime_dmy_slash() -> pl.DataFrame:
	"""DataFrame with string timestamp in DD/MM/YYYY HH:MM:SS format."""
	return pl.DataFrame(
		{
			"timestamp": [f"{i:02d}/01/2023 12:30:00" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


# =============================================================================
# UNCLEAN DATAFRAMES - Value columns that need casting
# =============================================================================


@pytest.fixture
def df_unclean_value_str() -> pl.DataFrame:
	"""DataFrame with string value column that can be cast to numeric."""
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in range(1, 21)],
			"value": [str(float(i * 10)) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


# =============================================================================
# UNCLEAN DATAFRAMES - Non-standard column names
# =============================================================================


@pytest.fixture
def df_unclean_no_standard_timestamp_name() -> pl.DataFrame:
	"""DataFrame with datetime column but non-standard name."""
	return pl.DataFrame(
		{
			"my_custom_date": [datetime(2023, 1, i) for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_no_standard_value_name() -> pl.DataFrame:
	"""DataFrame with numeric column but non-standard name."""
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in range(1, 21)],
			"my_custom_value": [float(i * 10) for i in range(1, 21)],
			"another_number": [float(i * 5 + 3) for i in range(1, 21)],
			"third_number": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_no_standard_names() -> pl.DataFrame:
	"""DataFrame with no standard column names for timestamp or value."""
	return pl.DataFrame(
		{
			"my_custom_date": [datetime(2023, 1, i) for i in range(1, 21)],
			"my_custom_value": [float(i * 10) for i in range(1, 21)],
			"another_number": [float(i * 5 + 3) for i in range(1, 21)],
			"third_number": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


# =============================================================================
# UNCLEAN DATAFRAMES - Missing values
# =============================================================================


@pytest.fixture
def df_unclean_with_nulls() -> pl.DataFrame:
	"""DataFrame with null values in numeric columns."""
	values = [float(i * 10) if i % 5 != 0 else None for i in range(1, 21)]
	measurements = [float(i * 5 + 3) if i % 7 != 0 else None for i in range(1, 21)]
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in range(1, 21)],
			"value": values,
			"measurement": measurements,
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_unclean_with_timestamp_nulls() -> pl.DataFrame:
	"""DataFrame with null values in timestamp column."""
	timestamps = [datetime(2023, 1, i) if i % 5 != 0 else None for i in range(1, 21)]
	return pl.DataFrame(
		{
			"timestamp": timestamps,
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


# =============================================================================
# INVALID DATAFRAMES - Should raise errors
# =============================================================================


@pytest.fixture
def df_invalid_no_datetime_col() -> pl.DataFrame:
	"""DataFrame with no datetime column at all (all strings, non-date)."""
	return pl.DataFrame(
		{
			"name": [f"item_{i}" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"category": [f"cat_{i % 3}" for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_invalid_no_numeric_col() -> pl.DataFrame:
	"""DataFrame with no numeric columns at all."""
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in range(1, 21)],
			"name": [f"item_{i}" for i in range(1, 21)],
			"category": [f"cat_{i % 3}" for i in range(1, 21)],
			"description": [f"desc_{i}" for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_invalid_non_castable_timestamp() -> pl.DataFrame:
	"""DataFrame with string timestamp column that cannot be cast to datetime."""
	return pl.DataFrame(
		{
			"timestamp": [f"not_a_date_{i}" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_invalid_non_castable_value() -> pl.DataFrame:
	"""DataFrame with string value column that cannot be cast to numeric."""
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in range(1, 21)],
			"value": [f"not_a_number_{i}" for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


@pytest.fixture
def df_invalid_missing_specified_col() -> pl.DataFrame:
	"""DataFrame missing a specified column name."""
	return pl.DataFrame(
		{
			"date": [datetime(2023, 1, i) for i in range(1, 21)],
			"amount": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	)


# =============================================================================
# LAZYFRAME VARIANTS
# =============================================================================


@pytest.fixture
def lf_unclean_timestamp_str() -> pl.LazyFrame:
	"""LazyFrame with string timestamp that needs casting."""
	return pl.DataFrame(
		{
			"timestamp": [f"2023-01-{i:02d}" for i in range(1, 21)],
			"value": [float(i * 10) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	).lazy()


@pytest.fixture
def lf_unclean_value_str() -> pl.LazyFrame:
	"""LazyFrame with string value that needs casting."""
	return pl.DataFrame(
		{
			"timestamp": [datetime(2023, 1, i) for i in range(1, 21)],
			"value": [str(float(i * 10)) for i in range(1, 21)],
			"measurement": [float(i * 5 + 3) for i in range(1, 21)],
			"target": [float(i * 2 - 1) for i in range(1, 21)],
		}
	).lazy()
