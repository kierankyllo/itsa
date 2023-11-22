
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from statsmodels.discrete.discrete_model import NegativeBinomialP
import sys
import os

def create_dir_if_not_exists(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_synthetic_data():
    """Generate synthetic Data data for testing."""
    np.random.seed(69)  # Seed for reproducibility
    dates = pd.date_range(start='2021-01-01', end='2021-12-31')
    trend = np.linspace(50, 200, len(dates))  # Linear trend from 50 to 200
    intervention_point = len(dates) // 2  # Introduce an 'intervention' at the midpoint
    trend[intervention_point:] += 50
    noise = np.random.normal(0, 20, len(dates))  # Add some noise
    Data = np.maximum(0, trend + noise).astype(int)  # Ensure non-negative Data
    return pd.DataFrame({'Date': dates, 'Data': Data})

def estimate_alpha(df, formula):
    """Estimate the dispersion parameter 'alpha' using the method of moments."""
    # Calculate the sample mean and sample variance of the 'Data' column
    sample_mean = df['Data'].mean()
    sample_variance = df['Data'].var()
    
    # Calculate alpha using the method of moments formula
    alpha = sample_mean ** 2 / (sample_variance - sample_mean)
  
    # Check if alpha is negative and print a warning or take corrective action
    if alpha <= 0:
        print(f"Warning: Estimated alpha is non-positive ({alpha}). This may indicate a problem with the data or model.")
        # Here, you might want to set alpha to a small positive value as a placeholder
        # Or re-estimate alpha using a different method, or inspect the data/model further
        alpha = 1  # This is arbitrary and should be replaced with appropriate domain-specific value

    return alpha


def interrupted_time_series_analysis(df, start_date, end_date):
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Set the intervention flag
        df['Intervention'] = 0  # Initialize the column with 0
        intervention_start = pd.to_datetime(start_date)
        intervention_end = pd.to_datetime(end_date)
        df.loc[intervention_start:intervention_end, 'Intervention'] = 1
        if 'Intercept' not in df.columns:
            df = sm.add_constant(df)
        # print("Checking the Intervention variable:")
        # print(df['Intervention'].value_counts())
        
        # Ensure 'Data' are non-negative and replace NaNs
        df['Data'] = df['Data'].apply(lambda x: max(x, 0))
        df.dropna(subset=['Data'], inplace=True)

        # Apply log-plus-one transformation
        df['Data'] = np.log(df['Data'] + 1)
        
        # Drop any rows that have NaN or infinite values after the above transformation
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Estimate alpha
        alpha = estimate_alpha(df, 'Data ~ Intervention')
        if alpha == 1:
            print("Fitting model with alpha = 1 , Poisson Distribution")
        else:
            print("Fitting model with alpha:", alpha)
        model = sm.GLM(df['Data'], df[['const', 'Intervention']], family=NegativeBinomial(alpha=alpha)).fit()
        return model
    except Exception as e:
        print(f"An error occurred during model fitting: {e}")
        return None



def normal_mode(csv_file_path, start_date, end_date):
    """Run the analysis in normal mode with actual data provided by the user."""
    print("Running in normal mode...")
    data = pd.read_csv(csv_file_path)
    check_data(data)
    
    # Select only numeric columns for checking NaNs and infinities.
    numeric_data = data.select_dtypes(include=[np.number])

    # Check for NaNs and infinities in the numeric columns
    if numeric_data.isnull().any().any() or np.isinf(numeric_data.values).any():
        print("Data contains NaNs or infinities in numeric columns. Cleaning data...")
        # Replace infinities with NaN, then drop rows with NaN values in numeric columns
        numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=numeric_data.columns, inplace=True)
    
    results = interrupted_time_series_analysis(data, start_date, end_date)
    if results is not None:
        print(results.summary())
    else:
        print("Model fitting was unsuccessful. Please check the data.")

def check_data(df):
    for col in df.select_dtypes(include=[np.number]).columns:  # Only check numeric columns
        try:
            # Check for infinities
            if np.isinf(df[col].values).any():
                print(f"Infinity found in column: {col}")
            
            # Check for NaNs
            if df[col].isnull().any():
                print(f"NaN found in column: {col}")
                
        except TypeError as e:
            print(f"TypeError encountered in column: {col} - {e}")



def test_mode():
    """Run the analysis in test mode with synthetic data."""
    print("Running in test mode...")
    synthetic_data = generate_synthetic_data()
    # Assuming the intervention happens exactly at the midpoint of the data
    midpoint = len(synthetic_data) // 2
    start_date = str(synthetic_data['Date'][midpoint].date())
    end_date = str(synthetic_data['Date'].iloc[-1].date())
    results = interrupted_time_series_analysis(synthetic_data, start_date, end_date)
    print(results.summary())

def display_help():
    """Display help instructions for using the script."""
    help_text = """
    Usage:
        python script.py [csv_file_path] [start_date] [end_date] [--help]
    
    Arguments:
    csv_file_path: The path to the CSV file containing the Data data.
    start_date: The start date of the intervention in the format YYYY-MM-DD.
    end_date: The end date of the intervention in the format YYYY-MM-DD.
    --help: Display this help message.
    
    If only the CSV file path is provided, the script runs in normal mode with the full data range.
    If no arguments are provided, the script runs in test mode.
    """
    print(help_text)

def main():
    # Check for help flag
    if '--help' in sys.argv:
        display_help()
    elif len(sys.argv) == 1:
        # No arguments provided, run in test mode
        test_mode()
    elif len(sys.argv) == 4:
        # Assume the first argument is the CSV file path, followed by start and end dates
        csv_file_path = sys.argv[1]
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        normal_mode(csv_file_path, start_date, end_date)
    else:
        print("Incorrect number of arguments. Use --help to see usage instructions.")
        sys.exit(1)

if __name__ == '__main__':
    main()