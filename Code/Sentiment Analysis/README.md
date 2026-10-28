# NHL Sentiment Analysis Tool

A tool for analyzing sentiment and public opinion towards NHL teams and players from Reddit discussions.

## Features

- **Reddit Data Scraping**: Collect posts and comments from NHL-related subreddits
- **Sentiment Analysis**: Analyze the sentiment of discussions towards teams and players
- **Player Abnormality Detection**: WIP

## Requirements

- Reddit API creds. Will include .env file
- If anything is going wrong text me or check the .env
- All dependencies are outlined in requirements.txt

```bash
pip install -r requirements.txt
```

### Files

There are two main files:
- main.py [used to populate csvs with a lot of data. no need to use it]
- update_data.py [should be your main workhorse. grabs data based on some arguments and ouputs it]

Example usage:
```bash
python update_data.py --days 7
```

Command line options:
- `--days`: Number of days to look back for data (default: 7)
- `--skip-scraping`: Use existing data without scraping new content
- `--team`: Analyze a specific team (not tested)

### Output Files

The tool generates several files in the `results` directory:
- `reddit_raw_data_latest.csv`: Combined raw data from Reddit
- `sentiment_data_latest.csv`: Sentiment analysis data (raw, not usable)
- `team_sentiment_{days}days.json`: Team sentiment report (formated, usable)
- `team_sentiment_{days}days.png`: Visualization of team sentiment (based directly on report)
- `player_abnormalities_{days}days.csv`: Player abnormality data (not up to standard)
- `player_abnormalities_{days}days.json`: Player abnormality report (not up to standard)
- `nhl_sentiment_report_{days}days.json`: Combined report with all information

### Getting Comprehensive Results

Some things you may notice when running update for smaller time values:
- Not all teams are ranked according to sentiment. You will see an obvious plateu of completely neutral teams. This is due to a lack of relational data.
- No player abnormality data
- Higher variation in sentiment. Generally speaking sentiment is positive across the board when it comes to NHL teams over spans of a year or two, but when you narrow the search to a day, you will see some teams dip negative. Remember, this tool is highly reliant upon an abundance of data, some of the results may look a little weird.

To avoid this:

1. Ensure you're analyzing a sufficient timeframe (14+ days recommended)
2. Run with default subreddits or add more NHL-related subreddits to `config.py`

Running periodic updates will give you day-to-day opinion variation for the 32 teams, this will be extremely variable as people on reddit are kind of wild. Another note: anything above 7 days is going to take a pretty long time to process.

Lastly, I would choose which files to display, run an update, and have a refresh happen every so often. The file gets replaced with each update, giving you new data.

Depending on the time-frame (for greater control) the name of the report will include the number of days in it {days}. If this is something you want to avoid it can be changed completely. Let me know.

```bash
# For comprehensive analysis:
python update_data.py --days 30
```

^ will seriously take a long time b/c of rate limiting :( 

### Searching For Files

Use prefix (i.e. 'team_sentiment_*.json') and search for last modified for easy data access.

Let me know if you need any help.
