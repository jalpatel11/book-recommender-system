def safe_title_lookup(book_id, book_id_to_isbn, isbn_to_title):
    isbn = book_id_to_isbn.get(book_id, None)
    return isbn_to_title.get(isbn, "Unknown Title")
