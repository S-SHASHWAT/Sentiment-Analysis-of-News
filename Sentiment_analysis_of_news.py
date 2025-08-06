import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("news.csv", parse_dates=['date'])

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Analyze each headline
def analyze_sentiment(text):

    score = analyzer.polarity_scores(text)
    return score['compound']

# Apply sentiment analysis
df['sentiment_score'] = df['headline'].apply(analyze_sentiment)

# Optional: classify into sentiment categories
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda score: 'Positive' if score > 0.05 else ('Negative' if score < -0.05 else 'Neutral')
)

# Save to CSV
df.to_csv("news_with_sentiment.csv", index=False)

# Group by date and average sentiment
daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()

# 📈 Plotting
plt.figure(figsize=(12, 6))
plt.plot(daily_sentiment['date'], daily_sentiment['sentiment_score'], marker='o', linestyle='-', color='blue')
plt.title('Average Daily News Sentiment')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ----- FILTERING SECTION -----
# Ask user to filter by date or sentiment
print("\nFILTER OPTIONS:")
date_input = input("Enter a specific date to filter (YYYY-MM-DD), or press Enter to skip: ").strip()
sentiment_input = input("Enter sentiment to filter (Positive / Negative / Neutral), or press Enter to skip: ").strip().capitalize()

# Filter DataFrame based on user input
filtered_df = df.copy()

# Filter by date
if date_input:
    try:
        filter_date = pd.to_datetime(date_input).date()
        filtered_df = filtered_df[filtered_df['date'].dt.date == filter_date]
    except:
        print("Invalid date format. Skipping date filter.")

# Filter by sentiment
if sentiment_input in ['Positive', 'Negative', 'Neutral']:
    filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_input]

# Display result
if filtered_df.empty:
    print("\n⚠️ No news articles found for your filter criteria.")
else:
    print(f"\n✅ Filtered Results ({len(filtered_df)} articles):\n")
    print(filtered_df[['date', 'headline', 'sentiment_label', 'sentiment_score']])

# Save filtered results
filtered_df.to_csv("filtered_news.csv", index=False)

# ----- PLOTTING (Original dataset or filtered depending on user choice) -----
# You can either plot full data or filtered data based on your needs
plot_choice = input("\nDo you want to plot sentiment trend of full data or filtered data? (full/filtered): ").strip().lower()

if plot_choice == "filtered":
    plot_df = filtered_df
else:
    plot_df = df

# Group and plot
daily_sentiment = plot_df.groupby(plot_df['date'].dt.date)['sentiment_score'].mean().reset_index()

# 📈 Plotting
plt.figure(figsize=(12, 6))
plt.plot(daily_sentiment['date'], daily_sentiment['sentiment_score'], marker='o', linestyle='-', color='blue')
plt.title('Average Daily News Sentiment')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
