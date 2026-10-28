import pandas as pd
import numpy as np
from textblob import TextBlob
import logging
import utils
import config
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/sentiment_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A class to analyze sentiment in NHL-related Reddit content.
    """
    
    def __init__(self):
        """
        Initialize the sentiment analyzer.
        """
        utils.download_nltk_data()
        logger.info("Sentiment analyzer initialized")
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of a given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment analysis results
        """
        if not text or not isinstance(text, str):
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': 'neutral'
            }
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > config.SENTIMENT_THRESHOLD_POSITIVE:
            sentiment = 'positive'
        elif polarity < config.SENTIMENT_THRESHOLD_NEGATIVE:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def analyze_dataframe(self, df):
        """
        Analyze sentiment for all text in a DataFrame.
        
        Args:
            df (DataFrame): DataFrame containing Reddit data
            
        Returns:
            DataFrame: DataFrame with sentiment analysis added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to analyze")
            return df
        
        logger.info("Analyzing sentiment for DataFrame with {} rows".format(len(df)))
        
        # copy df to avoid modifying original
        df_analysis = df.copy()
        
        # sentiment columns
        df_analysis['polarity'] = 0.0
        df_analysis['subjectivity'] = 0.0
        df_analysis['sentiment'] = 'neutral'
        
        posts_mask = df_analysis['is_post'] == True
        for idx in df_analysis[posts_mask].index:
            # combine title and selftext for posts (doesn't matter since sentiment is calculated holistically anyway)
            # also we do not want to give weight to title or selftext unevenely. basically a shortcut
            title = str(df_analysis.at[idx, 'title'] or '') if pd.notna(df_analysis.at[idx, 'title']) else ''
            selftext = str(df_analysis.at[idx, 'selftext'] or '') if pd.notna(df_analysis.at[idx, 'selftext']) else ''
            combined_text = title + ' ' + selftext
            
            # analyze text wholesale
            sentiment_result = self.analyze_text(combined_text)
            
            # store arbitrary results for later profiling and scalability
            df_analysis.at[idx, 'polarity'] = sentiment_result['polarity']
            df_analysis.at[idx, 'subjectivity'] = sentiment_result['subjectivity']
            df_analysis.at[idx, 'sentiment'] = sentiment_result['sentiment']
        
        # same thing with comments
        # TODO: give comments original post context, increasing accuracy. make less hacky
        comments_mask = df_analysis['is_post'] == False
        for idx in df_analysis[comments_mask].index:
            comment_text = str(df_analysis.at[idx, 'comment_body'] or '') if pd.notna(df_analysis.at[idx, 'comment_body']) else ''
            
            sentiment_result = self.analyze_text(comment_text)
            
            df_analysis.at[idx, 'polarity'] = sentiment_result['polarity']
            df_analysis.at[idx, 'subjectivity'] = sentiment_result['subjectivity']
            df_analysis.at[idx, 'sentiment'] = sentiment_result['sentiment']
        
        logger.info("Sentiment analysis completed")
        return df_analysis
    
    # TODO: make more accurate. right now a wholesale method which should work well for large data sets
    def analyze_team_sentiment(self, df, save_results=True):
        """
        Analyze sentiment specifically for NHL teams.
        
        Args:
            df (DataFrame): DataFrame containing Reddit data with sentiment analysis
            save_results: (bool): Whether to save and plot the results
            
        Returns:
            dict: Dictionary mapping team names to sentiment scores
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, no team sentiment to analyze")
            # initialize with all teams and neutral sentiment
            return {team: 0.0 for team in config.TEAM_NAMES.keys()}
        
        logger.info("Analyzing team sentiment")
        
        # where we will store team sentiment
        team_mentions = defaultdict(list)
        
        posts_mask = df['is_post'] == True
        for idx, row in df[posts_mask].iterrows():
            title = str(row['title'] or '') if pd.notna(row['title']) else ''
            selftext = str(row['selftext'] or '') if pd.notna(row['selftext']) else ''
            combined_text = title + ' ' + selftext
            
            # ok kind of redundant but whatever
            # this is where we weed out teams from the text
            mentioned_teams = utils.detect_teams_in_text(combined_text)
            
            for team in mentioned_teams:
                team_mentions[team].append(row['polarity'])
        
        comments_mask = df['is_post'] == False
        for idx, row in df[comments_mask].iterrows():
            comment_text = str(row['comment_body'] or '') if pd.notna(row['comment_body']) else ''
            
            mentioned_teams = utils.detect_teams_in_text(comment_text)
            
            for team in mentioned_teams:
                team_mentions[team].append(row['polarity'])
        
        # calculate average sentiment for each team
        team_sentiment = {}
        for team, polarities in team_mentions.items():
            if polarities:
                team_sentiment[team] = np.mean(polarities)
        
        # Add all teams from config.TEAM_NAMES that weren't mentioned with a neutral sentiment
        for team in config.TEAM_NAMES.keys():
            if team not in team_sentiment:
                team_sentiment[team] = 0.0
        
        team_sentiment = dict(sorted(team_sentiment.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Team sentiment analysis completed for {len(team_sentiment)} teams")
        
        if save_results:
            utils.save_to_json(team_sentiment, "team_sentiment")
            utils.plot_team_sentiment(team_sentiment)
        
        return team_sentiment
    
    def get_team_sentiment_summary(self, team_sentiment):
        """
        Generate a summary of team sentiment analysis.
        
        Args:
            team_sentiment (dict): Dictionary mapping team names to sentiment scores
            
        Returns:
            dict: Dictionary containing summary information
        """
        if not team_sentiment:
            return {
                'most_loved': None,
                'most_hated': None,
                'neutral': [],
                'total_teams': 0
            }
        
        # sort teams by sentiment
        sorted_teams = sorted(team_sentiment.items(), key=lambda x: x[1], reverse=True)
        
        most_loved = sorted_teams[0][0] if sorted_teams else None
        most_hated = sorted_teams[-1][0] if sorted_teams else None
        
        # neutrals
        neutral_teams = [team for team, score in team_sentiment.items() 
                       if abs(score) < (config.SENTIMENT_THRESHOLD_POSITIVE + config.SENTIMENT_THRESHOLD_NEGATIVE) / 2]
        
        return {
            'most_loved': most_loved,
            'most_loved_score': team_sentiment.get(most_loved, 0) if most_loved else 0,
            'most_hated': most_hated,
            'most_hated_score': team_sentiment.get(most_hated, 0) if most_hated else 0,
            'neutral': neutral_teams,
            'total_teams': len(team_sentiment)
        }

# test
if __name__ == "__main__":
    from reddit_scraper import RedditScraper
    
    scraper = RedditScraper()
    analyzer = SentimentAnalyzer()
    
    data = scraper.scrape_nhl_subreddits()
    sentiment_data = analyzer.analyze_dataframe(data)
    team_sentiment = analyzer.analyze_team_sentiment(sentiment_data)
    summary = analyzer.get_team_sentiment_summary(team_sentiment)
    
    print("Most loved team:", summary['most_loved'])
    print("Most hated team:", summary['most_hated']) 