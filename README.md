# itsa
Carries out ITSA analysis on negative binomial distributed univariate time series.  Provide the time series and the known intervention interval and recieve glm analysis output.  If no csv is provided the script operates in test mode and generates synthetic test data. Will perform a test to determine type of distribution by testing alpha.

Requires:

pandas, numpy, statsmodels

csv series with column headers: ['Date','Data']

    Usage:
        python script.py [csv_file_path] [start_date] [end_date] [--help]
    
    Arguments:
    csv_file_path: The path to the CSV file containing the Data data.
    start_date: The start date of the intervention in the format YYYY-MM-DD.
    end_date: The end date of the intervention in the format YYYY-MM-DD.
    --help: Display this help message.
    
    If only the CSV file path is provided, the script runs in normal mode with the full data range.
    If no arguments are provided, the script runs in test mode.
