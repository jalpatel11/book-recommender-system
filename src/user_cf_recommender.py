import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
from tqdm import tqdm

libsvm_file = "../data/ratings.libsvm"
num_users, max_book_id = 0, 0

with open(libsvm_file, 'r') as file:
    for line in file:
        for part in line.strip().split()[1:]:
            book_id, _ = part.split(":")
            max_book_id = max(max_book_id, int(book_id))
        num_users += 1

rating_matrix = lil_matrix((num_users, max_book_id + 1))
with open(libsvm_file, 'r') as file:
    for i, line in enumerate(file):
        for part in line.strip().split()[1:]:
            book_id, rating = part.split(":")
            rating_matrix[i, int(book_id)] = float(rating)

rating_matrix = rating_matrix.tocsr()
similarity_matrix = cosine_similarity(rating_matrix)

books_df = pd.read_csv("../data/Books.csv", sep=';', low_memory=False)
books_df.columns = books_df.columns.str.strip()
title_col = [col for col in books_df.columns if "title" in col.lower()][0]
isbn_to_title = dict(zip(books_df['ISBN'], books_df[title_col]))
book_id_to_isbn = {idx + 1: isbn for idx, isbn in enumerate(books_df['ISBN'])}

recommendations = []

for user_idx in tqdm(range(num_users), desc="Generating Recommendations", miniters=500):
    user_ratings = rating_matrix[user_idx]
    user_sim = similarity_matrix[user_idx]
    top_users = np.argsort(user_sim)[-11:-1][::-1]

    neighbor_books = set()
    for neighbor_idx in top_users:
        neighbor_books.update(rating_matrix[neighbor_idx].nonzero()[1])

    user_books = set(user_ratings.nonzero()[1])
    recommendable_books = neighbor_books - user_books

    estimated_ratings = {}
    for book_idx in recommendable_books:
        weighted_sum = 0
        sim_sum = 0
        for neighbor_idx in top_users:
            rating = rating_matrix[neighbor_idx, book_idx]
            if rating > 0:
                weighted_sum += user_sim[neighbor_idx] * rating
                sim_sum += user_sim[neighbor_idx]
        if sim_sum > 0:
            estimated_ratings[book_idx] = weighted_sum / sim_sum

    top_books = sorted(estimated_ratings.items(), key=lambda x: -x[1])
    selected_books = []
    for book_idx, score in top_books:
        isbn = book_id_to_isbn.get(book_idx, None)
        title = isbn_to_title.get(isbn, "Unknown Title")
        if title != "Unknown Title":
            selected_books.append((user_idx + 1, book_idx, title, round(score, 2)))
        if len(selected_books) == 5:
            break
    recommendations.extend(selected_books)

recommendations_df = pd.DataFrame(recommendations, columns=["User_ID", "Book_ID", "Book_Title", "Recommendation_Score"])
recommendations_df.sort_values(by=["User_ID", "Recommendation_Score"], ascending=[True, False]).to_csv("../results/final_recommendations.csv", index=False)
