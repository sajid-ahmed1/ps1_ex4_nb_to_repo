import pandas as pd
import numpy as np

def preliminary_analysis(df: pd.DataFrame):
    """
    Performs preliminary analysis on the pandas DataFrame by printing the first 5 rows, the information and the description.
    
    Args:
        df: pd.DataFrame - The DataFrame to analyze.
    Returns:
        None
    """
    print("Dataframe first 5 rows:")
    print(df.head())
    print("\nDataframe info:")
    print(df.info())
    print("\nDataframe description:")
    print(df.describe())
    return None


def missing_values_output(train_df: pd.DataFrame): 
    """
    Looks at the missing values in the training DataFrame and returns a DataFrame with total and percentage of missing values along with data types.

    Args:
        train_df: pd.DataFrame - The training DataFrame to analyze.
    Returns:
        pd.DataFrame: A DataFrame containing total and percentage of missing values and data types for each column.
    """
    total = train_df.isnull().sum()
    percent = (train_df.isnull().sum()/train_df.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in train_df.columns:
        dtype = str(train_df[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    df_missing_train = np.transpose(tt)

    return df_missing_train

def most_frequent_data(df: pd.DataFrame):
    """
    Outputs the mode for each column of a dataframe.

    Args:
        df: pd.DataFrame - The dataframe to analyse.
    Returns:
        pd.DataFrame: a dataframe containing the mode of each column, its count, its proportion of the total data.
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return np.transpose(tt)

def unique_values(df: pd.DataFrame):
    """
    Outputs the number of unique values for each column of a dataframe.

    Args:
        df: pd.DataFrame - The dataframe to analyse.
    Returns:
        pd.DataFrame: a dataframe containing the number of unique values in each column, and the number of total entries for each column.
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return np.transpose(tt)