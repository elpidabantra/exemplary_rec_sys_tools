import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample web activity data
web_data = {
    'UserID': [1, 2, 3, 4, 5],
    'Page': ['Home', 'Products', 'Contact', 'Home', 'Products'],
    'Duration': [60, 120, 30, 50, 200],  # Duration in seconds
    'Timestamp': pd.date_range(start='2023-01-01', periods=5, freq='T')
}
web_df = pd.DataFrame(web_data)

# Sample social media data
social_data = {
    'UserID': [1, 2, 3, 4, 5],
    'Post': ['Loving this product!', 'Having issues with the service.', 'Great customer support!', 'Fantastic quality!', 'Not satisfied with the purchase.'],
    'Platform': ['Twitter', 'Facebook', 'Twitter', 'Instagram', 'Facebook'],
    'Timestamp': pd.date_range(start='2023-01-01', periods=5, freq='T')
}
social_df = pd.DataFrame(social_data)

# Combine web activity and social media data
combined_df = pd.merge(web_df, social_df, on='UserID', suffixes=('_web', '_social'))

# Function to calculate sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Calculate sentiment for social media posts
combined_df['Sentiment'] = combined_df['Post'].apply(get_sentiment)

# Aggregate web activity data
web_activity = web_df.groupby('UserID').agg({'Duration': 'sum', 'Page': 'count'}).rename(columns={'Duration': 'TotalDuration', 'Page': 'PageViews'})

# Aggregate social media data
social_activity = social_df.groupby('UserID').agg({'Post': 'count'}).rename(columns={'Post': 'NumPosts'})
social_sentiment = social_df.groupby('UserID').agg({'Post': lambda x: TextBlob(' '.join(x)).sentiment.polarity}).rename(columns={'Post': 'AvgSentiment'})

# Combine aggregated data
final_df = pd.merge(web_activity, social_activity, on='UserID')
final_df = pd.merge(final_df, social_sentiment, on='UserID')

# Sample target variable (e.g., conversion)
final_df['Converted'] = [1, 0, 1, 1, 0]  # Mock conversion data

# Features and target variable
X = final_df.drop('Converted', axis=1)
y = final_df['Converted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Function to simulate integration with a CRM system
def crm_integration(final_df):
    for index, row in final_df.iterrows():
        print(f"UserID: {row.name}")
        print(f"Total Duration: {row['TotalDuration']} seconds, Page Views: {row['PageViews']}")
        print(f"Number of Posts: {row['NumPosts']}, Average Sentiment: {row['AvgSentiment']:.2f}")
        print(f"Conversion Prediction: {model.predict_proba([row.drop('Converted')])[0][1]:.2f}")
        print("-" * 50)

# Integrate insights with CRM
crm_integration(final_df)
