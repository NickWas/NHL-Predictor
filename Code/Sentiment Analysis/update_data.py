import argparse
import logging
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import time
from pathlib import Path

from reddit_scraper import RedditScraper
from sentiment_analyzer import SentimentAnalyzer
from player_detector import PlayerDetector
import utils
import config

os.makedirs('log', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/updater.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

REDDIT_DATA_FILE = "reddit_raw_data_latest.csv"
SENTIMENT_DATA_FILE = "sentiment_data_20250504_033314.csv"
PLAYER_ABNORMALITIES_FILE = "player_abnormalities_20250504_033455.csv"
# ABNORMALITY_REPORT_FILE = "player_abnormalities.json"
COMBINED_REPORT_FILE = "nhl_sentiment_report.json"

def get_latest_file(directory, prefix):
    """
    Get the most recent file with a given prefix in a directory.
    
    Args:
        directory (str): Directory to search in
        prefix (str): File prefix to match
        
    Returns:
        str: Path to the most recent file, or None if not found
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        return None
    
    matching_files = list(directory_path.glob(f"{prefix}_*.csv"))
    if not matching_files:
        return None
    
    # sort by time last modified
    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

def save_data_without_timestamp(df, filename, directory='results'):
    """
    Save DataFrame to a CSV file without adding a timestamp.
    
    Args:
        df (DataFrame): DataFrame to save
        filename (str): Filename to save as
        directory (str): Directory to save in
        
    Returns:
        str: Path to the saved file
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")
    return filepath

def save_json_without_timestamp(data, filename, directory='results'):
    """
    Save data to a JSON file without adding a timestamp.
    
    Args:
        data (dict): Data to save
        filename (str): Filename to save as
        directory (str): Directory to save in
        
    Returns:
        str: Path to the saved file
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    # pure laziness
    def json_serialize(obj):
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)
    
    def process_data(item):
        try:
            if isinstance(item, dict):
                return {k: process_data(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple, set)):
                return [process_data(i) for i in item]
            elif item is None:
                return None
            # Safely check if it's a datetime without using pandas functions
            elif hasattr(item, 'isoformat'):
                return item.isoformat()
            # Stick to basic types to avoid issues
            elif isinstance(item, (str, int, float, bool)):
                return item
            else:
                # Convert anything else to string
                return str(item)
        except Exception as e:
            # In case of any error, convert to string
            logger.warning(f"Error processing data for JSON: {str(e)}")
            return str(item)
    
    try:
        # same thing as before. too lazy to make seperate function
        processed_data = process_data(data)
        
        with open(filepath, 'w') as f:
            json.dump(processed_data, f, indent=2, default=json_serialize)
        
        logger.info(f"Data saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving JSON file: {str(e)}")
        return None

def load_existing_data():
    """
    Load existing Reddit data from CSV files.
    
    Returns:
        DataFrame: Existing Reddit data
    """
    os.makedirs('results', exist_ok=True)
    
    reddit_file = get_latest_file('results', 'reddit_raw_data')
    
    reddit_data = pd.DataFrame()
    if reddit_file:
        try:
            logger.info(f"Loading existing Reddit data from {reddit_file}")
            reddit_data = pd.read_csv(reddit_file)
            # convert 'created_utc' to datetime for easier filtering
            if 'created_utc' in reddit_data.columns:
                reddit_data['created_utc'] = pd.to_datetime(reddit_data['created_utc'])
        except Exception as e:
            logger.error(f"Error loading Reddit data: {str(e)}")
    
    return reddit_data

# may be useful
def scrape_recent_data(days=7, subreddits=None):
    """
    Scrape recent data from Reddit.
    
    Args:
        days (int): Number of days to look back
        subreddits (list): List of subreddits to scrape
        
    Returns:
        DataFrame: DataFrame containing scraped data
    """
    logger.info(f"Scraping data from the past {days} days")
    
    scraper = RedditScraper()
    
    if days <= 1:
        time_filter = 'day'
    elif days <= 7:
        time_filter = 'week'
    elif days <= 31:
        time_filter = 'month'
    else:
        time_filter = 'year'
    
    # Scrape each subreddit
    all_data = []
    
    # Use provided subreddits or default from config
    subreddits = subreddits or config.NHL_SUBREDDITS
    
    for subreddit in subreddits:
        try:
            # Scrape with time filter
            df = scraper.scrape_subreddit(subreddit, time_filter=time_filter)
            
            # Convert 'created_utc' to datetime for filtering
            if 'created_utc' in df.columns:
                df['created_utc'] = pd.to_datetime(df['created_utc'])
            
            # Filter to only include data from past N days
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['created_utc'] >= cutoff_date]
            
            all_data.append(df)
            
            # Sleep to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error scraping subreddit r/{subreddit}: {str(e)}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Scraped {len(combined_df)} items from the past {days} days")
        return combined_df
    else:
        logger.warning("No recent data was scraped")
        return pd.DataFrame()

# absolutely critical function. ensures no duplicates
def combine_data(existing_df, new_df, id_columns=None):
    """
    Combine existing and new data, removing duplicates.
    
    Args:
        existing_df (DataFrame): Existing data
        new_df (DataFrame): New data to add
        id_columns (list): Columns to use for identifying duplicates
        
    Returns:
        DataFrame: Combined data without duplicates
    """
    if existing_df.empty:
        return new_df.copy()
    
    if new_df.empty:
        return existing_df.copy()
    
    # default id col if not specified
    if id_columns is None:
        if 'id' in existing_df.columns:
            id_columns = ['id']
        else:
            id_columns = existing_df.columns.tolist()
    
    # make sure all required cols exist in both dataframes
    for col in id_columns:
        if col not in existing_df.columns or col not in new_df.columns:
            logger.warning(f"Column '{col}' not found in both dataframes, using all columns for deduplication")
            id_columns = [col for col in existing_df.columns if col in new_df.columns]
            break
    
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    if id_columns:
        before_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=id_columns, keep='first')
        after_count = len(combined_df)
        logger.info(f"Removed {before_count - after_count} duplicate rows")
    
    return combined_df

def update_data(days=7, skip_scraping=False):
    """
    Run the NHL sentiment analysis update.
    
    Args:
        days: Number of days to look back
        skip_scraping: Whether to skip scraping and use existing data only
    """
    os.makedirs('results', exist_ok=True)
    
    logger.info("Initializing components...")
    scraper = RedditScraper()
    analyzer = SentimentAnalyzer()
    detector = PlayerDetector()
    
    utils.download_nltk_data()
    
    logger.info("Loading existing data...")
    # generate sentiment from existing data + newly scraped data. fairly lightweight so should be fine
    existing_reddit_data = load_existing_data()
    logger.info(f"Loaded {len(existing_reddit_data)} existing records")
    
    if skip_scraping:
        logger.info("Skipping scraping, looking for existing data...")
        cache_files = [f for f in os.listdir('cache') if f.startswith('scraped_data_') and f.endswith('.csv')]
        
        if not cache_files:
            logger.error("No existing data found but scraping was skipped. Cannot proceed.")
            raise FileNotFoundError("No existing scraped data found")
            
        cache_files.sort(key=lambda f: os.path.getmtime(os.path.join('cache', f)), reverse=True)
        latest_file = os.path.join('cache', cache_files[0])
        logger.info(f"Loading data from {latest_file}")
        new_data = pd.read_csv(latest_file)
    else:
        logger.info(f"Scraping data from Reddit (looking back {days} days)...")
        
        # determine how many days we want to go back.
        # we can only scrape in day, week, month, year, or all, but we can filter data by utc stamp later
        time_filter = 'week'
        if days <= 1:
            time_filter = 'day'
        elif days <= 7:
            time_filter = 'week'
        elif days <= 30:
            time_filter = 'month'
        elif days <= 365:
            time_filter = 'year'
        else:
            time_filter = 'all'
            
        logger.info(f"Using time filter: {time_filter}")
        
        new_data = scraper.scrape_nhl_subreddits(time_filter=time_filter, save_to_csv=False)
        
        # extra-fine grained filtering
        if time_filter != 'day' and days not in [7, 30, 365]:
            cutoff_date = (datetime.now() - timedelta(days=days)).timestamp()
            new_data = new_data[new_data['created_utc'] >= cutoff_date]
            logger.info(f"Filtered to {len(new_data)} posts from the last {days} days")
    
    # combine dataframes so we have continuity
    logger.info("Combining new data with existing data...")
    combined_data = combine_data(existing_reddit_data, new_data, id_columns=['id'])
    logger.info(f"Combined data contains {len(combined_data)} records")
    
    reddit_data_file = save_data_without_timestamp(combined_data, "reddit_raw_data_latest.csv")
    logger.info(f"Combined Reddit data saved to {reddit_data_file}")
    
    logger.info(f"Filtering data to the specified timeframe of {days} days...")
    cutoff_date = datetime.now() - timedelta(days=days)
    logger.info(f"Cutoff date: {cutoff_date}")
    
    # Inspect the data format first
    if not combined_data.empty and 'created_utc' in combined_data.columns:
        logger.info(f"Data contains {len(combined_data)} records total")
        logger.info(f"Column types: {combined_data.dtypes}")
        
        # Print a few sample values to understand the data format
        sample_rows = min(5, len(combined_data))
        for i in range(sample_rows):
            if i < len(combined_data):
                val = combined_data['created_utc'].iloc[i]
                logger.info(f"Sample {i}: {val}, type: {type(val)}")
        
        # Simple approach: ensure created_utc is in datetime format and compare directly
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(combined_data['created_utc']):
                logger.info("Converting created_utc to datetime format...")
                combined_data['created_utc'] = pd.to_datetime(combined_data['created_utc'], errors='coerce')
                
            # Drop rows with invalid dates
            valid_data = combined_data.dropna(subset=['created_utc'])
            logger.info(f"Records with valid dates: {len(valid_data)}")
            
            if not valid_data.empty:
                # Log date range stats
                min_date = valid_data['created_utc'].min()
                max_date = valid_data['created_utc'].max()
                logger.info(f"Date range in data: {min_date} to {max_date}")
                
                # Direct datetime comparison
                cutoff_date_pd = pd.to_datetime(cutoff_date)
                logger.info(f"Pandas cutoff date: {cutoff_date_pd}")
                
                # Count records that should match
                matching_count = (valid_data['created_utc'] >= cutoff_date_pd).sum()
                logger.info(f"Records >= cutoff date: {matching_count}")
                
                # Filter the data
                timeframe_data = valid_data[valid_data['created_utc'] >= cutoff_date_pd]
                logger.info(f"After date filtering: {len(timeframe_data)} records")
            else:
                logger.warning("No valid dates found in the data")
                timeframe_data = combined_data
        except Exception as e:
            logger.error(f"Error during date filtering: {str(e)}")
            timeframe_data = combined_data
    else:
        logger.warning("No created_utc column found or dataframe is empty, using all data for analysis")
        timeframe_data = combined_data
    
    logger.info(f"Analyzing {len(timeframe_data)} posts from the last {days} days")
    
    if timeframe_data.empty:
        logger.warning(f"No data available in the {days}-day timeframe, using all available data")
        timeframe_data = combined_data
        logger.info(f"Using all available data: {len(timeframe_data)} posts")
    
    logger.info("Analyzing sentiment...")
    sentiment_data = analyzer.analyze_dataframe(timeframe_data)
    
    sentiment_file = save_data_without_timestamp(sentiment_data, "sentiment_data_latest.csv")
    logger.info(f"Sentiment data saved to {sentiment_file}")
    
    logger.info("Analyzing team sentiment...")
    team_sentiment = analyzer.analyze_team_sentiment(sentiment_data, save_results=False)
    
    team_summary = analyzer.get_team_sentiment_summary(team_sentiment)
    
    # no timestamps to preserve memory and sanity
    team_report_path = os.path.join('results', f"team_sentiment_{days}days.json")
    with open(team_report_path, 'w') as f:
        json.dump(team_summary, f, indent=2)
    logger.info(f"Team sentiment report saved to {team_report_path}")
    
    # create plot
    logger.info("Creating team sentiment visualization...")
    visualization_path = os.path.join('results', f"team_sentiment_{days}days.png")
    utils.plot_team_sentiment(team_sentiment, output_file=visualization_path)
    logger.info(f"Visualization saved to {visualization_path}")
    
    logger.info("Detecting player abnormalities...")
    abnormalities = detector.detect_abnormalities(sentiment_data, save_to_file=False)
    
    abnormality_summary = detector.get_abnormality_summary(abnormalities, save_to_file=False)
    
    player_abnormalities = detector.group_by_player(abnormalities, save_to_file=False)
    
    abnormality_file = save_data_without_timestamp(pd.DataFrame(abnormalities), "player_abnormalities_latest.csv")
    logger.info(f"Player abnormalities saved to {abnormality_file}")
    
    latest_abnorm_file = save_data_without_timestamp(pd.DataFrame(abnormalities), f"player_abnormalities_{days}days.csv")
    logger.info(f"Day-specific player abnormalities saved to {latest_abnorm_file}")
    
    abnormality_report_path = os.path.join('results', f"player_abnormalities_{days}days.json")
    with open(abnormality_report_path, 'w') as f:
        json.dump({
            'summary': abnormality_summary,
            'details': player_abnormalities
        }, f, indent=2, default=str)
    logger.info(f"Abnormality report saved to {abnormality_report_path}")
    
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'timeframe_days': days,
        'total_records_analyzed': len(sentiment_data),
        'team_sentiment': team_summary,
        'player_abnormalities': abnormality_summary
    }
    
    output_path = os.path.join('results', f"nhl_sentiment_report_{days}days.json")
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Combined report for {days}-day timeframe generated and saved to {output_path}")
    
    logger.info("Analysis completed successfully!")
    
    # immediate debugging stuff. if this looks fine, ur fine
    print(f"\n======= NHL SENTIMENT ANALYSIS SUMMARY ({days} DAYS) =======")
    print(f"Total items analyzed: {len(sentiment_data)}")
    print(f"Teams analyzed: {team_summary['total_teams']}")
    if team_summary.get('most_loved'):
        print(f"Most loved team: {team_summary['most_loved']} (score: {team_summary['most_loved_score']:.3f})")
    if team_summary.get('most_hated'):
        print(f"Most hated team: {team_summary['most_hated']} (score: {team_summary['most_hated_score']:.3f})")
    print(f"Player abnormalities detected: {abnormality_summary['total']}")
    print(f"Injuries: {abnormality_summary['injuries']}")
    print(f"Returns: {abnormality_summary['returns']}")
    print("==========================================================")

def generate_combined_report(team_summary, abnormality_summary):
    """
    Generate a combined report with all analysis results.
    
    Args:
        team_summary: Team sentiment summary
        abnormality_summary: Player abnormality summary
    """
    # create report data
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'team_sentiment': team_summary,
        'player_abnormalities': abnormality_summary
    }
    
    output_path = os.path.join('results', COMBINED_REPORT_FILE)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Combined report generated and saved to {output_path}")
    
    return output_path

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='NHL Sentiment Analysis Update Tool')
    
    parser.add_argument('--days', type=int, default=7, 
                        help='Number of days to look back for new data')
    parser.add_argument('--skip-scraping', action='store_true',
                        help='Skip scraping and just use existing data')
    parser.add_argument('--team', type=str, help='Analyze a specific team')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    update_data(days=args.days, skip_scraping=args.skip_scraping) 