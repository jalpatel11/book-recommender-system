
# 📚 Book Recommender System

This project implements a **user-based collaborative filtering** recommender system using the [Book Crossing dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset). The system recommends personalized book titles by identifying similar users based on rating patterns and generating suggestions accordingly.

---

## 📌 Project Overview

- **Approach:** User-based Collaborative Filtering
- **Similarity Metric:** Cosine Similarity
- **Input Format:** Ratings in LIBSVM format
- **Output:** Top-5 personalized book recommendations per user
- **Visualization:** Distribution of scores, recommendation coverage, most recommended books

---

## 🗂️ Project Structure

```
book-recommender-system/
│
├── data/
│   ├── Books.csv
│   ├── Ratings.csv
│   ├── Users.csv
│   └── ratings.libsvm
│
├── src/
│   ├── preprocess_libsvm.py         # Converts Ratings.csv to libsvm format
│   ├── user_cf_recommender.py      # Main recommendation logic
│   ├── visualize_results.py        # All plots and visual insights
│   └── utils.py                    # Helper functions
│
├── results/
│   ├── final_recommendations.csv
│   └── plots/
│       ├── score_distribution.png
│       ├── user_recommendation_pie.png
│       └── top_books_bar.png
│
├── report/
│   ├── IFT511_Project_Report.pdf
│   └── Screenshots/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/book-recommender-system.git
cd book-recommender-system
```

### 2. Install Required Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Convert Ratings to LIBSVM Format

```bash
python src/preprocess_libsvm.py
```

### Step 2: Run the Recommendation Engine

```bash
python src/user_cf_recommender.py
```

### Step 3: Visualize the Results

```bash
python src/visualize_results.py
```

---

## 📊 Sample Outputs

- `final_recommendations.csv`: Contains `User_ID`, `Book_ID`, `Book_Title`, `Recommendation_Score`
- Visuals include:
  - Distribution of predicted scores
  - Percentage of users receiving 1–5 recommendations
  - Top 10 most recommended books

---

## 🧠 Key Assumptions

- Only books with non-zero ratings are used.
- Books without valid titles (e.g., "Untitled") are excluded from recommendations.
- Each user receives up to 5 recommendations based on their top-10 similar users.
- Book IDs in the LIBSVM file are mapped to sequential integers starting from 1.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 👥 Team Members

**Group 9 - Arizona State University**
- Nikita Kumari
- Nishu Singh
- Jal Dharmendrabhai Patel
- Karthikeya Battula