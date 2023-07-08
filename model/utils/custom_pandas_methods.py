import pandas as pd

def reorder_columns(
    self,
    reference_column: str,
    column_to_move: str | list[str],
) -> pd.DataFrame:
    """Reorder columns in a DataFrame.

    Moves the specified column(s) to a new position in the DataFrame, just
    before the specified reference column.

    Parameters
    ----------
    self : pandas.DataFrame
        The input DataFrame.
    reference_column : str
        The name of the reference column.

    Returns
    -------
    pandas.DataFrame
        The reordered DataFrame.

    Raises
    ------
    ValueError
        If an invalid column name or index is provided.

    """
    insert_position = self.columns.get_loc(reference_column)

    if isinstance(column_to_move, str):
        column_to_move = [column_to_move]

    if set(column_to_move).issubset(self.columns):
        remaining_cols = list(
            self.drop(column_to_move, axis=1)
            .columns
        )
    else:
        raise ValueError("Invalid column name or index.")

    columns_adjusted = (
        remaining_cols[:insert_position]
        + list(column_to_move)
        + remaining_cols[insert_position:]
    )

    return self[columns_adjusted]

pd.DataFrame.reorder_columns = reorder_columns
