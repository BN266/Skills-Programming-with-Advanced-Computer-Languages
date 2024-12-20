#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Personal Mood Tracker

A command-line application for tracking daily moods, activities, and sentiments.
Features include:
- Daily mood and activity logging
- Sentiment analysis of notes
- Mood trends visualization
- Activity correlation analysis
- Statistics and streaks tracking
- Data export functionality

Dependencies:
- pandas: Data manipulation and analysis
- matplotlib/seaborn: Data visualization
- textblob: Sentiment analysis
- openai: AI-powered mood suggestions

Usage:
1. Ensure all required packages are installed
2. Run the script
3. Use the menu to interact with the tracker

Note: This version includes an OpenAI API key for demonstration.
In production, API keys should be stored securely in environment variables.
"""

#!pip install textblob     # For sentiment analysis
#!pip install openai       # For OpenAI API
#!pip install pandas matplotlib seaborn  # For data analysis and visualization

# NOTE: This API key is included for demonstration/testing purposes only
# In production, API keys should be stored securely in environment variables
OPENAI_API_KEY = ""

# Standard library imports
import os
import random
from datetime import datetime, timedelta

# Third-party imports 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from openai import OpenAI

# Set plotting style for cleaner visuals
sns.set(style="whitegrid")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define the path for the CSV file where mood data is stored
DATA_FILE = 'data/mood_data.csv'

# Define common moods for validation and suggestion
COMMON_MOODS = ['happy', 'sad', 'anxious', 'excited', 'tired', 'energetic', 
                'calm', 'stressed', 'content', 'frustrated']

def validate_mood_input(mood):
    """
    Validates mood input against common moods.
    
    Args:
        mood (str): The mood input from the user
        
    Returns:
        tuple: (bool, str) - (is_valid, message/validated_mood)
    """
    # Normalize input
    mood = mood.lower().strip()
    
    if not mood:
        return False, "Mood cannot be empty."
    
    # Check if mood is in the predefined list
    if mood not in COMMON_MOODS:
        # If not found, provide a suggestion by comparing character sets
        suggestion = min(COMMON_MOODS, key=lambda x: len(set(x) - set(mood)))
        return False, f"Did you mean '{suggestion}'? Please use one of: {', '.join(COMMON_MOODS)}"
        
    return True, mood

def analyze_sentiment(notes):
    """
    Analyzes the sentiment of provided text using TextBlob.
    
    Args:
        notes (str): Text to analyze
        
    Returns:
        str: Sentiment category ('Positive', 'Negative', or 'Neutral')
    """
    blob = TextBlob(notes)
    polarity = blob.sentiment.polarity
    
    # Polarity > 0.1 is positive, < -0.1 is negative, else neutral
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def provide_mood_tips(sentiment):
    """
    Provides mood improvement tips based on the sentiment.
    
    Args:
        sentiment (str): The sentiment category
    """
    tips = {
        'Positive': "Keep up the great work! Consider maintaining your routine to sustain your positive mood.",
        'Neutral': "It's a balanced day. To enhance your mood, try engaging in activities you enjoy.",
        'Negative': "I'm sorry you're feeling this way. Consider reaching out to a friend or trying a relaxation technique."
    }
    print("\n**Mood Improvement Tips:**")
    # Get the corresponding tip or a default message
    print(tips.get(sentiment, "Stay positive!"))

def provide_suggestions(mood, activities):
    """
    Uses OpenAI API to provide personalized suggestions based on mood and activities.
    
    Args:
        mood (str): Current mood
        activities (str): Activities performed
    """
    prompt = f"""
    I am feeling {mood} today. I have done the following activities: {activities}. 
    Can you provide some suggestions or activities to help me maintain or improve my mood?
    """
    
    try:
        # Using the OpenAI client to get suggestions
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        suggestion = response.choices[0].message.content.strip()
        print("\n**Suggestions to Improve or Maintain Your Mood:**")
        print(suggestion)
        
    except Exception as e:
        # In case of errors (e.g., network issues, API errors), we handle them gracefully
        print(f"Error fetching suggestions: {e}")

def get_all_entries():
    """
    Retrieves all mood entries from the CSV file.
    Creates the file if it doesn't exist.
    
    Returns:
        pandas.DataFrame: DataFrame containing all mood entries
    """
    try:
        # If the data file doesn't exist, create it with appropriate columns
        if not os.path.exists(DATA_FILE):
            os.makedirs('data', exist_ok=True)
            df = pd.DataFrame(columns=['Date', 'Mood', 'Activities', 'Notes', 'Sentiment'])
            df.to_csv(DATA_FILE, index=False)
        return pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return pd.DataFrame()

def add_entry(mood, activities, notes):
    """
    Adds a new mood entry to the CSV file.
    
    Args:
        mood (str): The mood to record (already validated)
        activities (str): Comma-separated activities performed
        notes (str): Additional notes/comments
    """
    # Use today's date for the entry
    date = datetime.now().strftime('%Y-%m-%d')
    mood = mood.title()  # Store moods with a capitalized format for consistency
    activities = ', '.join([activity.strip().lower() for activity in activities.split(',')])
    notes = notes.strip()
    
    # Analyze sentiment of the notes
    sentiment = analyze_sentiment(notes)
    new_entry = {'Date': date, 'Mood': mood, 'Activities': activities, 'Notes': notes, 'Sentiment': sentiment}
    new_entry_df = pd.DataFrame([new_entry])
    
    try:
        # If file exists, we might need to handle overwriting today's entry
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            if date in df['Date'].values:
                # Ask user if they want to overwrite existing entry for today
                overwrite = input("An entry for today already exists. Do you want to overwrite it? (y/n): ").strip().lower()
                if overwrite != 'y':
                    print("Entry not saved.")
                    return
                # Remove old entry for today before adding the new one
                df = df[df['Date'] != date]
            df = pd.concat([df, new_entry_df], ignore_index=True)
        else:
            # If file does not exist, create it and add the entry
            os.makedirs('data', exist_ok=True)
            df = new_entry_df
            
        # Save updated data to CSV
        df.to_csv(DATA_FILE, index=False)
        print("Entry saved successfully.")
        
        # Provide suggestions based on mood and activities using OpenAI
        provide_suggestions(mood, activities)
        # Provide local mood improvement tips based on sentiment
        provide_mood_tips(sentiment)
        
    except Exception as e:
        print(f"Error adding new entry: {e}")

def view_mood_trends():
    """
    Analyzes and plots mood trends using improved visualizations.
    Shows mood distribution in a bar chart.
    """
    print("\n--- Mood Trends ---")
    df = get_all_entries()
    if df.empty:
        print("No data available to analyze.")
        return
        
    try:
        # Create figure for mood distribution
        plt.figure(figsize=(12, 6))
        
        # Plot mood distribution
        mood_counts = df['Mood'].value_counts()
        sns.barplot(y=mood_counts.index, x=mood_counts.values, palette='viridis', hue=mood_counts.index, legend=False)
        plt.title('Overall Mood Distribution')
        plt.xlabel('Count')
        plt.ylabel('Mood')
        
        plt.tight_layout()
        plt.show()
        plt.close()  # Close the plot window after showing
        
    except Exception as e:
        print(f"Error generating mood trends visualization: {e}")
        plt.close()  # Ensure plot window is closed even if error occurs

def view_activity_correlation():
    """
    Analyzes and plots correlation between activities and moods using a heatmap.
    Helps identify which activities are associated with certain moods.
    """
    print("\n--- Activity Correlations ---")
    df = get_all_entries()
    if df.empty:
        print("No data available to analyze.")
        return
        
    try:
        # Separate activities into individual rows for counting
        activity_mood = df[['Mood', 'Activities']].copy()
        activity_mood['Activities'] = activity_mood['Activities'].str.split(',')
        activity_mood = activity_mood.explode('Activities')
        activity_mood['Activities'] = activity_mood['Activities'].str.strip().str.lower()
        
        # Create a pivot table counting how many times each activity is associated with each mood
        pivot = pd.pivot_table(activity_mood, index='Activities', columns='Mood', aggfunc='size', fill_value=0)
        print("Activity-Mood Correlation Matrix:")
        print(pivot)
        
        # Plotting the pivot as a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, annot=True, fmt='d', cmap='coolwarm')
        plt.title('Correlation Between Activities and Moods')
        plt.xlabel('Mood')
        plt.ylabel('Activities')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error generating activity correlation: {e}")

def show_mood_stats():
    """
    Shows interesting statistics about mood patterns including:
    - Most common moods
    - Most productive moods (based on activity count)
    - Best days of the week (in terms of positive sentiment)
    - Sentiment distribution
    """
    df = get_all_entries()
    if df.empty:
        print("No data available for analysis.")
        return
    
    try:
        print("\n=== Mood Statistics ===")
        
        # Most common mood
        most_common = df['Mood'].mode()[0]
        mood_counts = df['Mood'].value_counts()
        print(f"Most frequent mood: {most_common} ({mood_counts[most_common]} times)")
        
        # Most productive mood (most activities on average)
        df['activity_count'] = df['Activities'].str.count(',') + 1
        mood_productivity = df.groupby('Mood')['activity_count'].mean().sort_values(ascending=False)
        productive_mood = mood_productivity.index[0]
        productive_avg = mood_productivity.iloc[0]
        print(f"Most productive mood: {productive_mood} (avg {productive_avg:.1f} activities)")
        
        # Convert date column to datetime and derive day of week
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.day_name()
        
        # Best day: The day of week with the most positive sentiment entries
        best_day = df[df['Sentiment'] == 'Positive']['DayOfWeek'].mode()[0]
        print(f"Best day of the week: {best_day}")
        
        # Sentiment distribution
        sentiment_dist = df['Sentiment'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"{sentiment}: {count} entries")
            
    except Exception as e:
        print(f"Error calculating mood statistics: {e}")

def track_mood_streaks():
    """
    Tracks and displays positive mood streaks and other streak-based statistics.
    This shows how many days in a row the user had a positive mood,
    and the longest positive mood streak recorded.
    Also shows how many days have passed since the last negative mood.
    """
    df = get_all_entries()
    if df.empty:
        print("No data available for streak analysis.")
        return
        
    try:
        # Sort entries by date to properly calculate streaks
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Initialize counters for streak calculations
        current_positive_streak = 0
        max_positive_streak = 0
        
        print("\n=== Mood Streaks ===")
        
        # Iterate through sentiments and count positive streaks
        for sentiment in df['Sentiment']:
            if sentiment == 'Positive':
                current_positive_streak += 1
                max_positive_streak = max(max_positive_streak, current_positive_streak)
            else:
                # Reset the positive streak counter if a non-positive day is encountered
                current_positive_streak = 0
                
        # Calculate how long since the last negative mood entry
        last_negative = df[df['Sentiment'] == 'Negative']['Date'].max()
        if pd.notna(last_negative):
            days_since_negative = (pd.Timestamp.now() - last_negative).days
            print(f"Days since last negative mood: {days_since_negative}")
        
        print(f"Longest positive mood streak: {max_positive_streak} days")
        # current_positive_streak here will show the streak at the end of iteration
        print(f"Current positive mood streak: {current_positive_streak} days")
        
    except Exception as e:
        print(f"Error tracking mood streaks: {e}")

def log_mood():
    """
    Enhanced mood logging function with input validation and error handling.
    Guides the user through logging their daily mood entry.
    """
    print("\n--- Log Today's Mood ---")
    
    # Get mood with validation, repeat prompt if invalid
    while True:
        mood = input("How are you feeling today? (e.g., Happy, Sad, Anxious): ").strip()
        is_valid, message = validate_mood_input(mood)
        
        if is_valid:
            # Mood is valid, break from the loop
            break
        print(f"Invalid input: {message}")
    
    # Get activities and ensure at least one activity is entered
    while True:
        activities = input("What activities did you do today? (comma-separated): ").strip()
        if activities:
            # Activities provided, break
            break
        print("Please enter at least one activity.")
    
    # Get optional notes (can be empty)
    notes = input("Any additional notes or comments: ").strip()
    
    try:
        # Add the new entry to the dataset
        add_entry(mood, activities, notes)
    except Exception as e:
        print(f"Error saving entry: {e}")
        print("Please try again.")

def export_csv():
    """
    Exports the mood data to a new CSV file with error handling.
    Useful for backup or sharing data externally.
    """
    export_path = 'data/mood_data_export.csv'
    try:
        df = pd.read_csv(DATA_FILE)
        df.to_csv(export_path, index=False)
        print(f"Data exported successfully to {export_path}.")
    except Exception as e:
        print(f"Error exporting data: {e}")

def populate_sample_data(num_entries=20):
    """
    Populates the CSV file with sample data for testing and demonstration.
    
    Args:
        num_entries (int): Number of sample entries to generate
    """
    moods = list(map(str.title, COMMON_MOODS))  # Use validated moods list
    activities = ['reading', 'jogging', 'coding', 'cooking', 'meditation', 
                 'gaming', 'studying', 'yoga', 'painting', 'cycling']
    sentiments = ['Positive', 'Negative', 'Neutral']
    
    sample_data = []
    for i in range(num_entries):
        # Create a sequence of dates going backwards from today
        date = (datetime.now() - timedelta(days=num_entries - i)).strftime('%Y-%m-%d')
        mood = random.choice(moods)
        activity = random.choice(activities)
        note = f"Sample entry for {date} - feeling {mood.lower()}"
        # Determine sentiment based on chosen mood
        sentiment = 'Positive' if mood in ['Happy', 'Excited', 'Content'] else \
                    'Negative' if mood in ['Sad', 'Anxious', 'Frustrated'] else 'Neutral'
        
        sample_data.append({
            'Date': date,
            'Mood': mood,
            'Activities': activity,
            'Notes': note,
            'Sentiment': sentiment
        })
    
    sample_df = pd.DataFrame(sample_data)
    
    try:
        # If data file exists and has data, we won't overwrite with sample data
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            if df.empty:
                sample_df.to_csv(DATA_FILE, index=False)
                print(f"Added {num_entries} sample entries to {DATA_FILE}.")
            else:
                print(f"{DATA_FILE} already contains data. Sample data not added.")
        else:
            # If data file doesn't exist, create and populate it
            os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
            sample_df.to_csv(DATA_FILE, index=False)
            print(f"Created {DATA_FILE} and added {num_entries} sample entries.")
    except Exception as e:
        print(f"Error populating sample data: {e}")

def display_menu():
    """
    Displays the main menu options with improved formatting.
    This function is called before every user choice.
    """
    print("\n=== Personal Mood Tracker ===")
    print("1. Log Today's Mood")
    print("2. View Mood Trends")
    print("3. View Activity Correlations")
    print("4. View Mood Statistics")
    print("5. View Mood Streaks")
    print("6. Export Data")
    print("7. Exit")

def main_menu():
    """
    Main function to run the Mood Tracker application.
    Handles the menu interaction and user input in a loop until 'Exit' is chosen.
    """
    while True:
        try:
            display_menu()
            choice = input("\nSelect an option (1-7): ").strip()
            
            if choice == '1':
                log_mood()
            elif choice == '2':
                view_mood_trends()
            elif choice == '3':
                view_activity_correlation()
            elif choice == '4':
                show_mood_stats()
            elif choice == '5':
                track_mood_streaks()
            elif choice == '6':
                export_csv()
            elif choice == '7':
                print("\nThank you for using the Mood Tracker. Goodbye!")
                break
            else:
                print("Invalid choice. Please select a number between 1 and 7.")
                
        except KeyboardInterrupt:
            # Allow the user to gracefully exit using Ctrl+C
            print("\n\nProgram interrupted by user. Exiting...")
            break
        except Exception as e:
            # Catch unexpected errors without crashing the whole program
            print(f"An unexpected error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    try:
        print("Welcome to the Personal Mood Tracker!")
        print("Initializing...")
        # Populate with sample data for demo; in production, you may skip this step.
        populate_sample_data()
        # Run the main menu loop
        main_menu()
    except Exception as e:
        print(f"Critical error: {e}")
        print("Please check your setup and try again.")


# In[ ]:





# In[ ]:




