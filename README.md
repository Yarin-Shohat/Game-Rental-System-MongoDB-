# Game Rental System (MongoDB)

This project is a **game rental system** implemented in Python using **MongoDB** as the database. The system allows user registration, login, game rental management, and game recommendations.

## Features

- **User Management**
  - Register new users with hashed passwords using `bcrypt`.
  - User authentication with secure password verification.

- **Game Rental System**
  - Load game data from a CSV file into MongoDB.
  - Rent and return games.
  - Prevent duplicate game entries.

- **Game Recommendations**
  - Recommend games based on genre preferences.
  - Recommend games based on name similarity using **TF-IDF** and **cosine similarity**.

- **Game Analytics**
  - Find top-rated games based on user scores.
  - Decrease user scores for games on a specific platform.
  - Compute the average game rating per platform.
  - Analyze the distribution of games across different genres.

## Technologies Used

- **Python**
- **MongoDB** (NoSQL Database)
- **Pandas** (for CSV handling)
- **bcrypt** (for password hashing)
- **scikit-learn** (for text similarity recommendations)

## Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/Yarin-Shohat/Game-Rental-System-MongoDB.git
   cd game-rental-system
   ```

2. **Install dependencies**
   ```sh
   pip install pymongo bcrypt pandas scikit-learn
   ```

3. **Set up MongoDB**
   - Ensure you have MongoDB installed and running locally on `mongodb://localhost:27017/`.

4. **Run the script**
   ```sh
   python app.py
   ```

## Usage

### 1. Register a new user
```python
from app import LoginManager

lm = LoginManager()
lm.register_user("test_user", "password123")
```

### 2. Log in a user
```python
user = lm.login_user("test_user", "password123")
```

### 3. Load game data from CSV
```python
from app import DBManager

dbm = DBManager()
dbm.load_csv()
```

### 4. Rent a game
```python
dbm.rent_game(user, "Super Mario Odyssey")
```

### 5. Return a game
```python
dbm.return_game(user, "Super Mario Odyssey")
```

### 6. Recommend games by genre
```python
dbm.recommend_games_by_genre(user)
```

### 7. Recommend games by name similarity
```python
dbm.recommend_games_by_name(user)
```

### 8. Find top-rated games
```python
dbm.find_top_rated_games(8.0)
```

### 9. Decrease scores for games on a platform
```python
dbm.decrement_scores("Nintendo Switch")
```

### 10. Get average score per platform
```python
dbm.get_average_score_per_platform()
```

### 11. Get genre distribution
```python
dbm.get_genres_distribution()
```
