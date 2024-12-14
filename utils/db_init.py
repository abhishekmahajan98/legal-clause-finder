# import sqlite3

# # Connect to SQLite database
# conn = sqlite3.connect('contracts.db')
# cursor = conn.cursor()

# # Create documents table
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS documents (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     title TEXT NOT NULL,
#     category TEXT NOT NULL,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# )
# ''')

# # Create document_pages table
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS document_pages (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     document_id INTEGER,
#     page_number INTEGER NOT NULL,
#     content TEXT,
#     FOREIGN KEY (document_id) REFERENCES documents (id)
# )
# ''')

# # Commit changes and close connection
# conn.commit()
# conn.close()
