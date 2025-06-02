import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../results/final_recommendations.csv")

plt.figure(figsize=(8, 5))
sns.histplot(df['Recommendation_Score'], bins=20, kde=True)
plt.title("Distribution of Recommendation Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../results/plots/score_distribution.png")

user_counts = df['User_ID'].value_counts().value_counts()
labels = [f'{i} Books' for i in user_counts.index]
sizes = user_counts.values

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Number of Recommendations per User")
plt.tight_layout()
plt.savefig("../results/plots/user_recommendation_pie.png")

top_books = df['Book_ID'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_books.sort_values().plot(kind='barh')
plt.title("Top 10 Most Recommended Books")
plt.xlabel("Number of Recommendations")
plt.ylabel("Book ID")
plt.tight_layout()
plt.savefig("../results/plots/top_books_bar.png")
