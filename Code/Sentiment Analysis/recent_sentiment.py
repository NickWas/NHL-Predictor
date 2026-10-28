import pandas as pd
import logging
import os
import json
import argparse
from datetime import datetime, timedelta

from reddit_scraper import RedditScraper
from sentiment_analyzer import SentimentAnalyzer
import utils
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recent_sentiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RecentSentimentAnalyzer:
    """
    A class to analyze recent NHL team sentiment from Reddit.
    """
    
    def __init__(self, days_to_analyze=7):
        """
        Initialize the recent sentiment analyzer.
        
        Args:
            days_to_analyze (int): Number of recent days to analyze
        """
        self.days_to_analyze = days_to_analyze
        self.scraper = RedditScraper()
        self.analyzer = SentimentAnalyzer()
        
        utils.download_nltk_data()
        
        logger.info(f"Recent sentiment analyzer initialized (analyzing past {days_to_analyze} days)")
    
    def fetch_recent_data(self, subreddits=None, posts_limit=100):
        """
        Fetch recent data from NHL subreddits.
        
        Args:
            subreddits (list): List of subreddits to scrape
            posts_limit (int): Maximum posts per subreddit
            
        Returns:
            DataFrame: DataFrame with recent Reddit data
        """
        subreddits = subreddits or config.NHL_SUBREDDITS
        
        time_filter = "day" if self.days_to_analyze <= 1 else "week"
        
        logger.info(f"Fetching recent data from {len(subreddits)} subreddits (time filter: {time_filter})")
        
        data = self.scraper.scrape_nhl_subreddits(subreddits=subreddits)
        
        if data.empty:
            logger.warning("No data fetched")
            return pd.DataFrame()
        
        data['created_utc'] = pd.to_datetime(data['created_utc'])
        
        cutoff_date = datetime.now() - timedelta(days=self.days_to_analyze)
        recent_data = data[data['created_utc'] >= cutoff_date]
        
        logger.info(f"Fetched {len(recent_data)} items from the past {self.days_to_analyze} days")
        
        if not recent_data.empty:
            utils.save_to_csv(recent_data, f"recent_data_{self.days_to_analyze}days")
        
        return recent_data
    
    def analyze_recent_sentiment(self, data=None):
        """
        Analyze sentiment from recent data.
        
        Args:
            data (DataFrame, optional): Recent data to analyze. If None, fetches new data.
            
        Returns:
            dict: Dictionary with team sentiment scores
        """
        if data is None or data.empty:
            data = self.fetch_recent_data()
            
        if data.empty:
            logger.warning("No data available for sentiment analysis")
            return {}
        
        logger.info("Analyzing sentiment in recent data")
        
        sentiment_data = self.analyzer.analyze_dataframe(data)
        
        utils.save_to_csv(sentiment_data, f"recent_sentiment_{self.days_to_analyze}days")
        
        team_sentiment = self.analyzer.analyze_team_sentiment(sentiment_data)
        
        team_summary = self.analyzer.get_team_sentiment_summary(team_sentiment)
        
        utils.save_to_json(team_summary, f"recent_team_summary_{self.days_to_analyze}days")
        
        utils.plot_team_sentiment(team_sentiment)
        
        return team_sentiment
    
    def generate_opinion_poll(self, team_sentiment=None):
        """
        Generate a contemporary opinion poll of NHL teams.
        
        Args:
            team_sentiment (dict, optional): Team sentiment scores. If None, analyzes new data.
            
        Returns:
            dict: Opinion poll results
        """
        if team_sentiment is None:
            team_sentiment = self.analyze_recent_sentiment()
            
        if not team_sentiment:
            logger.warning("No sentiment data available for opinion poll")
            return {}
        
        logger.info("Generating contemporary opinion poll")
        
        sorted_teams = sorted(team_sentiment.items(), key=lambda x: x[1], reverse=True)
        
        poll_results = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_period': f"Past {self.days_to_analyze} days",
            'most_popular_teams': sorted_teams[:5],
            'least_popular_teams': sorted_teams[-5:],
            'neutral_zone': sorted_teams[len(sorted_teams)//2-2:len(sorted_teams)//2+3],
            'all_teams': dict(sorted_teams)
        }
        
        utils.save_to_json(poll_results, f"team_opinion_poll_{self.days_to_analyze}days")
        
        return poll_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NHL Recent Sentiment Analysis")
    parser.add_argument("--days", type=int, default=7, help="Number of recent days to analyze")
    parser.add_argument("--load-data", type=str, help="Path to load data from instead of scraping")
    return parser.parse_args()

def main():
    """Main function to run the recent sentiment analysis."""
    args = parse_arguments()
    
    analyzer = RecentSentimentAnalyzer(days_to_analyze=args.days)
    
    if args.load_data and os.path.exists(args.load_data):
        logger.info(f"Loading data from {args.load_data}")
        data = pd.read_csv(args.load_data)
        data['created_utc'] = pd.to_datetime(data['created_utc'])
    else:
        data = analyzer.fetch_recent_data()
    
    poll_results = analyzer.generate_opinion_poll()
    
    print("\n======= CONTEMPORARY NHL TEAM OPINION POLL =======")
    print(f"Analysis period: Past {args.days} days")
    print(f"Total items analyzed: {len(data) if not data.empty else 0}")
    
    if poll_results and 'most_popular_teams' in poll_results:
        print("\nMost Popular Teams:")
        for team, score in poll_results['most_popular_teams']:
            print(f"  {team}: {score:.3f}")
        
        print("\nLeast Popular Teams:")
        for team, score in poll_results['least_popular_teams']:
            print(f"  {team}: {score:.3f}")
    
    print("===================================================")

if __name__ == "__main__":
    main() 