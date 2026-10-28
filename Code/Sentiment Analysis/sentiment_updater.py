import pandas as pd
import logging
import os
import json
import argparse
from datetime import datetime, timedelta
import time
import schedule

from reddit_scraper import RedditScraper
from sentiment_analyzer import SentimentAnalyzer
import utils
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_updater.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentUpdater:
    """
    A class to update NHL team sentiment over time with decay.
    """
    
    def __init__(self, decay_factor=0.95, sentiment_file=None):
        """
        Initialize the sentiment updater.
        
        Args:
            decay_factor (float): Factor to decay historical sentiment (0-1)
            sentiment_file (str): Path to historical sentiment file to load
        """
        self.decay_factor = decay_factor
        self.scraper = RedditScraper()
        self.analyzer = SentimentAnalyzer()
        self.historical_sentiment = {}
        self.last_update = None
        
        # Ensure NLTK data is downloaded
        utils.download_nltk_data()
        
        # Load historical sentiment if available
        if sentiment_file and os.path.exists(sentiment_file):
            self.load_historical_sentiment(sentiment_file)
            
        logger.info(f"Sentiment updater initialized (decay factor: {decay_factor})")
    
    def load_historical_sentiment(self, file_path):
        """
        Load historical sentiment from file.
        
        Args:
            file_path (str): Path to historical sentiment file
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if 'team_sentiment' in data:
                self.historical_sentiment = data['team_sentiment']
            elif isinstance(data, dict):
                self.historical_sentiment = data
                
            if 'last_update' in data:
                self.last_update = datetime.strptime(data['last_update'], '%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Loaded historical sentiment for {len(self.historical_sentiment)} teams")
            
        except Exception as e:
            logger.error(f"Error loading historical sentiment: {e}")
    
    def fetch_new_data(self, hours=24):
        """
        Fetch new data from the last X hours.
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            DataFrame: DataFrame with new Reddit data
        """
        time_filter = "day"
        
        logger.info(f"Fetching new data from the last {hours} hours")
        
        data = self.scraper.scrape_nhl_subreddits()
        
        if data.empty:
            logger.warning("No new data fetched")
            return pd.DataFrame()
        
        data['created_utc'] = pd.to_datetime(data['created_utc'])
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        new_data = data[data['created_utc'] >= cutoff_time]
        
        logger.info(f"Fetched {len(new_data)} items from the last {hours} hours")
        
        return new_data
    
    def apply_sentiment_decay(self):
        """
        Apply time decay to historical sentiment scores.
        """
        if not self.historical_sentiment:
            logger.info("No historical sentiment to decay")
            return
        
        logger.info(f"Applying sentiment decay factor: {self.decay_factor}")
        
        for team in self.historical_sentiment:
            if isinstance(self.historical_sentiment[team], (int, float)):
                self.historical_sentiment[team] *= self.decay_factor
        
        logger.info("Sentiment decay applied")
    
    def update_sentiment(self, hours=24, append_mode=True):
        """
        Update sentiment with new data.
        
        Args:
            hours (int): Number of hours to look back for new data
            append_mode (bool): Whether to append to historical sentiment or reset
            
        Returns:
            dict: Updated team sentiment scores
        """
        new_data = self.fetch_new_data(hours=hours)
        
        if new_data.empty:
            logger.warning("No new data to update sentiment")
            return self.historical_sentiment
        
        logger.info("Analyzing sentiment in new data")
        sentiment_data = self.analyzer.analyze_dataframe(new_data)
        
        new_team_sentiment = self.analyzer.analyze_team_sentiment(sentiment_data)
        
        if append_mode and self.historical_sentiment:
            # Apply decay to historical sentiment
            self.apply_sentiment_decay()
            
            logger.info("Combining historical and new sentiment")
            
            combined_sentiment = dict(self.historical_sentiment)
            
            # Add new sentiment to historical
            for team, score in new_team_sentiment.items():
                if team in combined_sentiment:
                    # Weighted average: 70% historical (with decay), 30% new
                    combined_sentiment[team] = (combined_sentiment[team] * 0.7) + (score * 0.3)
                else:
                    combined_sentiment[team] = score
            
            updated_sentiment = combined_sentiment
        else:
            # Use only new sentiment
            logger.info("Using only new sentiment (resetting historical)")
            updated_sentiment = new_team_sentiment
        
        self.historical_sentiment = updated_sentiment
        self.last_update = datetime.now()
        
        self.save_updated_sentiment()
        
        # Generate visualization
        utils.plot_team_sentiment(updated_sentiment)
        
        return updated_sentiment
    
    def save_updated_sentiment(self):
        """
        Save updated sentiment to file.
        """
        # Create data structure with metadata
        data = {
            'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S'),
            'decay_factor': self.decay_factor,
            'team_sentiment': self.historical_sentiment
        }
        
        utils.save_to_json(data, "updated_team_sentiment")
        
        logger.info("Updated sentiment saved")
    
    def start_scheduled_updates(self, interval_hours=24):
        """
        Start scheduled sentiment updates.
        
        Args:
            interval_hours (int): Interval between updates in hours
        """
        logger.info(f"Starting scheduled updates every {interval_hours} hours")
        
        def update_job():
            logger.info(f"Running scheduled sentiment update")
            self.update_sentiment(hours=interval_hours)
        
        schedule.every(interval_hours).hours.do(update_job)
        
        update_job()
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NHL Sentiment Updater")
    parser.add_argument("--decay", type=float, default=0.95, help="Decay factor for historical sentiment (0-1)")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back for new data")
    parser.add_argument("--historical", type=str, help="Path to historical sentiment file")
    parser.add_argument("--reset", action="store_true", help="Reset historical data instead of appending")
    parser.add_argument("--schedule", action="store_true", help="Run as a scheduled service")
    parser.add_argument("--interval", type=int, default=24, help="Interval in hours between updates when scheduled")
    return parser.parse_args()

def main():
    """Main function to run the sentiment updater."""
    args = parse_arguments()
    
    updater = SentimentUpdater(
        decay_factor=args.decay,
        sentiment_file=args.historical
    )
    
    if args.schedule:
        updater.start_scheduled_updates(interval_hours=args.interval)
    else:
        updated_sentiment = updater.update_sentiment(
            hours=args.hours,
            append_mode=not args.reset
        )
        
        print("\n======= NHL TEAM SENTIMENT UPDATE =======")
        print(f"Update time: {updater.last_update}")
        print(f"Teams analyzed: {len(updated_sentiment)}")
        
        sorted_teams = sorted(updated_sentiment.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 Teams:")
        for team, score in sorted_teams[:5]:
            print(f"  {team}: {score:.3f}")
        
        print("\nBottom 5 Teams:")
        for team, score in sorted_teams[-5:]:
            print(f"  {team}: {score:.3f}")
        
        print("=========================================")

if __name__ == "__main__":
    main() 