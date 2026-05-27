import pandas as pd

def melt_pod_historical_prices(df):
    """
    Melt wide-format timeseries data to long format with columns: date, pod_id, price
    
    Parameters:
    df (pd.DataFrame): Wide-format DataFrame with 'Dates' column and asset columns
    
    Returns:
    pd.DataFrame: Long-format DataFrame with columns [date, pod_id, price]
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # make sure Dates column is properly formatted
    if 'Dates' in df_copy.columns:
        df_copy['Dates'] = pd.to_datetime(df_copy['Dates'])
    else:
        raise ValueError("'Dates' column not found in DataFrame")
    
    # Get all columns except 'Dates' - asset/pod columns
    asset_columns = [col for col in df_copy.columns if col != 'Dates']
    
    # Melt the DataFrame
    melted_df = pd.melt(
        df_copy,
        id_vars=['Dates'],
        value_vars=asset_columns,
        var_name='pod_id',
        value_name='price'
    )
    
    # Rename the date column for consistency
    melted_df = melted_df.rename(columns={'Dates': 'date'})
    
    # Remove rows with NaN prices (optional - uncomment if desired)
    # melted_df = melted_df.dropna(subset=['price'])
    
    # Sort by date and pod_id for better organization
    melted_df = melted_df.sort_values(['pod_id', 'date']).reset_index(drop=True).dropna()
    
    return melted_df