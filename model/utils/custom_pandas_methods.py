import pandas as pd

def reorder_columns(
    self,
    last_col: str,
    column_to_move: str | list[str],
) -> pd.DataFrame:
    """Reorder columns in a DataFrame.

    Moves the specified column(s) to a new position in the DataFrame,
    just after the specified last column.

    Parameters
    ----------
    self : pandas.DataFrame
        The input DataFrame.
    last_col : str
        The name of the column after which the specified column(s) will
        be moved.
    column_to_move : str or list of str
        The name(s) of the column(s) to be moved.

    Returns
    -------
    pandas.DataFrame
        The reordered DataFrame.

    Raises
    ------
    ValueError
        If an invalid column name or index is provided.

    """
    insert_position2 = self.columns.get_loc(last_col)

    if isinstance(column_to_move, str):
        column_to_move = [column_to_move]

    if set(column_to_move).issubset(self.columns):
        remaining_cols = list(
            self.drop(column_to_move, axis=1)
            .columns
        )
    else:
        raise ValueError("Invalid column name or index.")

    insert_position = self.columns.get_loc(last_col)

    columns_adjusted = (
        remaining_cols[:insert_position]
        + list(column_to_move)
        + remaining_cols[insert_position:]
    )

    columns_adjusted2 = (
        remaining_cols[:insert_position2]
        + list(column_to_move)
        + remaining_cols[insert_position2:]
    )

    return self[columns_adjusted]

pd.DataFrame.reorder_columns = reorder_columns
