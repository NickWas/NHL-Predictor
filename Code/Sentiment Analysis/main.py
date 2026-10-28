import argparse
import logging
import pandas as pd
import os
import json
from datetime import datetime

from reddit_scraper import RedditScraper
from sentiment_analyzer import SentimentAnalyzer
from player_detector import PlayerDetector
import utils
import config

os.makedirs('log', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/nhl_sentiment_tool.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_analysis(args):
    """
    Run the NHL sentiment analysis tool.
    
    Args:
        args: Command line arguments
    """
    os.makedirs('results', exist_ok=True)
    
    logger.info("Initializing components...")
    scraper = RedditScraper()
    analyzer = SentimentAnalyzer()
    detector = PlayerDetector()
    
    utils.download_nltk_data()
    
    if args.load_data and os.path.exists(args.load_data):
        logger.info(f"Loading data from {args.load_data}")
        data = pd.read_csv(args.load_data)
    else:
        logger.info("Scraping data from Reddit...")
        
        skip_existing = not args.scrape_all
        if skip_existing:
            logger.info("Will skip posts that have already been scraped")
        else:
            logger.info("Will scrape all posts, including ones scraped before")
            
        data = scraper.scrape_nhl_subreddits(skip_existing=skip_existing)
    
    if data.empty:
        logger.error("No data scraped or loaded. Exiting.")
        return
    if args.load_sentiment: 
        logger.info(f"Loading sentiment data from {args.load_sentiment}")
        sentiment_data = pd.read_csv(args.load_sentiment)
    else:
        # analyze sentiment
        logger.info("Analyzing sentiment...")
        sentiment_data = analyzer.analyze_dataframe(data)
    
    # save sentiment data + link to file incase of further use
    sentiment_file = utils.save_to_csv(sentiment_data, "sentiment_data")
    
    # analyze team sentiment; prunes sentiment data to only include teams mentioned in the data
    logger.info("Analyzing team sentiment...")
    team_sentiment = analyzer.analyze_team_sentiment(sentiment_data)
    
    # get team sentiment summary
    team_summary = analyzer.get_team_sentiment_summary(team_sentiment)
    
    # detect player abnormalities
    logger.info("Detecting player abnormalities...")
    abnormalities = detector.detect_abnormalities(sentiment_data)
    
    # get abnormality summary
    abnormality_summary = detector.get_abnormality_summary(abnormalities)
    
    # group abnormalities by player
    player_abnormalities = detector.group_by_player(abnormalities)
    
    # generate final report
    generate_report(team_summary, abnormality_summary, args.output)
    
    logger.info("Analysis completed successfully!")
    
    print("\n======= NHL SENTIMENT ANALYSIS SUMMARY =======")
    print(f"Total items analyzed: {len(sentiment_data)}")
    print(f"Teams analyzed: {team_summary['total_teams']}")
    if team_summary['most_loved']:
        print(f"Most loved team: {team_summary['most_loved']} (score: {team_summary['most_loved_score']:.3f})")
    if team_summary['most_hated']:
        print(f"Most hated team: {team_summary['most_hated']} (score: {team_summary['most_hated_score']:.3f})")
    print(f"Player abnormalities detected: {abnormality_summary['total']}")
    print(f"Injuries: {abnormality_summary['injuries']}")
    print(f"Returns: {abnormality_summary['returns']}")
    print("==============================================")

def generate_report(team_summary, abnormality_summary, output_path=None):
    """
    Generate a final report with the analysis results.
    
    Args:
        team_summary: Team sentiment summary
        abnormality_summary: Player abnormality summary
        output_path: Path to save the report
    """
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'team_sentiment': team_summary,
        'player_abnormalities': abnormality_summary
    }
    
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join('results', f"nhl_sentiment_report_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report generated and saved to {output_path}")
    
    return output_path

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='NHL Sentiment Analysis Tool')
    
    parser.add_argument('--load-data', type=str, help='Path to load scraped data from')
    parser.add_argument('--load-sentiment', type=str, help='Path to load sentiment data from')
    parser.add_argument('--output', type=str, help='Path to save the final report')
    parser.add_argument('--team', type=str, help='Analyze a specific team')
    parser.add_argument('--limit', type=int, default=config.POSTS_LIMIT, 
                        help='Maximum number of posts to scrape per subreddit')
    parser.add_argument('--time-filter', type=str, default=config.TIME_FILTER,
                        choices=['day', 'week', 'month', 'year', 'all'],
                        help='Time filter for posts')
    parser.add_argument('--scrape-all', action='store_true', 
                        help='Scrape all posts, even if already scraped before')
    parser.add_argument('--abnormalities', action='store_true', 
                        help='Detect player abnormalities')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    run_analysis(args) 