import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the recommendation results
df = pd.read_csv("results/final_recommendations.csv")

# Ensure the plots directory exists
os.makedirs("results/plots", exist_ok=True)

# Plot 1: Distribution of Recommendation Scores
plt.figure(figsize=(8, 5))
sns.histplot(df['Recommendation_Score'], bins=20, kde=True)
plt.title("Distribution of Recommendation Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/plots/score_distribution.png")
plt.close()

# Plot 2: Recommendation count per user (Pie Chart)
user_counts = df['User_ID'].value_counts().value_counts().sort_index()
labels = [f'{i} Books' for i in user_counts.index]
sizes = user_counts.values

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Number of Recommendations per User")
plt.tight_layout()
plt.savefig("results/plots/user_recommendation_pie.png")
plt.close()

# Plot 3: Top 10 most frequently recommended books
top_books = df['Book_ID'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_books.sort_values().plot(kind='barh')
plt.title("Top 10 Most Recommended Books")
plt.xlabel("Number of Recommendations")
plt.ylabel("Book ID")
plt.tight_layout()
plt.savefig("results/plots/top_books_bar.png")
plt.close()
