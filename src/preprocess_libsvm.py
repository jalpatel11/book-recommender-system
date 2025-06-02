import pandas as pd
from sklearn.model_selection import train_test_split

def convert_to_libsvm(input_csv, output_file):
    ratings = pd.read_csv(input_csv, sep=';')
    ratings = ratings[ratings['Book-Rating'] > 0]

    with open(output_file, 'w') as f:
        for user_id, group in ratings.groupby('User-ID'):
            line = f"0"
            for _, row in group.iterrows():
                line += f" {int(row['ISBN_ID'])}:{row['Book-Rating']}"
            f.write(line + "\n")

if __name__ == "__main__":
    convert_to_libsvm("../data/Ratings.csv", "../data/ratings.libsvm")
