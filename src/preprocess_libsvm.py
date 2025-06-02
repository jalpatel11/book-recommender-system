import pandas as pd
import argparse
import os


def convert_to_libsvm(ratings_csv_path: str, output_path: str) -> None:
    # Load ratings file
    ratings_df = pd.read_csv(ratings_csv_path, sep=';', low_memory=False)
    ratings_df.columns = ratings_df.columns.str.strip()

    # Drop rows with missing user/book/rating info
    ratings_df = ratings_df.dropna(subset=['User-ID', 'ISBN', 'Rating'])

    # Get sorted unique ISBNs and User-IDs
    unique_isbns = sorted(ratings_df['ISBN'].unique())
    unique_users = sorted(ratings_df['User-ID'].unique())

    # Create mappings
    isbn_to_id = {isbn: idx + 1 for idx, isbn in enumerate(unique_isbns)}  # Book_IDs start from 1
    user_to_id = {uid: idx for idx, uid in enumerate(unique_users)}        # User_IDs start from 0

    # Map original values to IDs
    ratings_df['User_ID'] = ratings_df['User-ID'].map(user_to_id)
    ratings_df['Book_ID'] = ratings_df['ISBN'].map(isbn_to_id)

    # Group by user
    grouped = ratings_df.groupby('User_ID')

    # Format lines
    libsvm_lines = []
    for user_id in range(len(unique_users)):
        user_group = grouped.get_group(user_id).sort_values(by='Book_ID')
        pairs = [f"{int(row['Book_ID'])}:{float(row['Rating'])}" for _, row in user_group.iterrows()]
        line = "0 " + " ".join(pairs)
        libsvm_lines.append(line)

    # Writing to output file
    with open(output_path, 'w') as f:
        for line in libsvm_lines:
            f.write(line + '\n')

    print(f"LibSVM file created at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ratings.csv to LibSVM format.")
    parser.add_argument("--input", required=True, help="Path to Ratings.csv")
    parser.add_argument("--output", default="ratings.libsvm", help="Output path for LibSVM file")

    args = parser.parse_args()
    convert_to_libsvm(args.input, args.output)
