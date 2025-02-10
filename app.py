import pymongo
import bcrypt
import ast
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LoginManager:
    """
    LoginManager class to manage Users Logins.

    Methods
    -------
    register_user(username: str, password: str) -> None
        Add a new user to users collection.
    login_user(username: str, password: str) -> object
        Log in a user with the provided username and password.
    """
    def __init__(self) -> None:
        """
        Constructs all the necessary attributes for the LoginManager class.
        """
        # MongoDB connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["hw3"]
        self.collection = self.db["users"]
        self.salt = b"$2b$12$ezgTynDsK3pzF8SStLuAPO"  # TODO: if not working, generate a new salt

    def register_user(self, username: str, password: str) -> None:
        """
        Add a new user to users collection.

        :param username: The username for the new user
        :param password: The password for the new user
        :return: None
        """
        # Check if the username and password are not empty strings
        if not username or not password:
            raise ValueError("Username and password are required.")
        if username == '' or password == '':
            raise ValueError("Username and password are required.")
        # Check if the length of both username and password is at least 3 characters
        if len(password) < 3 or len(username) < 3:
            raise ValueError("Username and password must be at least 3 characters.")
        # Check if the username already exists in the database
        if self.collection.find_one({'username': username}):
            raise ValueError(f"User already exists: {username}")
        # Encrypt password
        bytes = password.encode('utf-8')
        enc_password = bcrypt.hashpw(bytes, self.salt)
        # Insert User
        self.collection.insert_one({'username': username, 'password': enc_password, 'rented_game_ids': []})

    def login_user(self, username: str, password: str) -> object:
        """
        Log in a user with the provided username and password

        :param username: The username of the user trying to log in.
        :param password: The password of the user trying to log in.
        :return:
        """
        # Check if the username and password are not empty strings
        if not username or not password:
            raise ValueError("Invalid username or password")
        if username == '' or password == '':
            raise ValueError("Invalid username or password")
        # Check if the length of both username and password is at least 3 characters
        if len(password) < 3 or len(username) < 3:
            raise ValueError("Invalid username or password")
        # Check if the username already exists in the database
        if not self.collection.find_one({'username': username}):
            raise ValueError("Invalid username or password")
        # Get password from DB
        user = self.collection.find_one({'username': username})
        user_pass = user.get('password')
        # Check password match
        inputBytes = password.encode('utf-8')  # encoding inputed password
        if bcrypt.checkpw(inputBytes, user_pass):
            print(f"Logged in successfully as: {username}")
            return user
        else:
            raise ValueError("Invalid username or password")


class DBManager:
    """
    DBManager class to manage users DB.

    Methods
    -------
    load_csv(self) -> None
        Load a csv file into DB.
    rent_game(self, user: dict, game_title: str) -> str
        Rent the game title.
    return_game(self, user: dict, game_title: str) -> str
        Return the game title.
    recommend_games_by_genre(self, user: dict) -> list
        Recommend games by genre.
    recommend_games_by_name(self, user: dict) -> list
        Recommend games by name.
    find_top_rated_games(self, min_score) -> list
        Find top rated games.
    decrement_scores(self, platform_name) -> None
        Decrement scores.
    get_average_score_per_platform(self) -> dict
        Get average score per platform.
    get_genres_distribution(self) -> dict
        Get genres distribution.
    """
    def __init__(self) -> None:
        # MongoDB connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["hw3"]
        self.user_collection = self.db["users"]
        self.game_collection = self.db["games"]

    def load_csv(self) -> None:
        """
        Loads data from a CSV file into games collection.
        It is assumed that this function will be executed once at first.

        :return: None
        """
        # Check file ia valid
        try:
            data = pd.read_csv("NintendoGames.csv")
            data = data.fillna('')
            data = data.astype(str)
        except:
            return
        # Read Data
        games = []
        for i, row in data.iterrows():
            game = {}
            meta_score = row.iloc[0]  # Get meta_score
            if meta_score != '':
                game["meta_score"] = float(meta_score)
            title = row.iloc[1]  # Get title
            if title != '':
                game["title"] = title
            platform = row.iloc[2]  # Get platform
            if platform != '':
                game["platform"] = platform
            date = row.iloc[3]  # Get date
            if date != '':
                game["date"] = date
            user_score = row.iloc[4]  # Get user_score
            if user_score != '':
                game["user_score"] = float(user_score)
            link = row.iloc[5]  # Get link
            if link != '':
                game["link"] = link
            esrb_rating = row.iloc[6]  # Get esrb_rating
            if esrb_rating != '':
                game["esrb_rating"] = esrb_rating
            developers = row.iloc[7]  # Get developers
            if developers != '':
                game["developers"] = ast.literal_eval(developers)
            genres = row.iloc[8]  # Get genres
            if genres != '':
                game["genres"] = ast.literal_eval(genres)
            if not self.game_collection.find_one({'title': game['title']}):
                game["is_rented"] = False
                games.append(game)
        if len(games) > 0:
            self.game_collection.insert_many(games)

    def rent_game(self, user: dict, game_title: str) -> str:
        """
        Rents a game for the user.

        :param user: The user object.
        :param game_title: The title of the game to be rented.
        :return: (str) "success" if the rental process was successful, and "failure" if the rental process failed.
        """
        # Check valid input
        if not game_title or not user:
            return "failure"
        if len(user) == 0 or len(game_title) == 0:
            return "failure"
        user = self.user_collection.find_one({'username': user['username']})
        if not user:
            return "failure"
        # Get the game
        try:
            game = self.game_collection.find_one({'title': game_title})
        except:
            return "failure"
        # Game not found
        if not game:
            return f"{game_title} not found"
        # Game already rented
        rented = game["is_rented"]
        if rented:
            return f"{game_title} is already rented"
        # Rent the game
        try:
            self.game_collection.update_one(game, {"$set": {"is_rented": True}})
            self.user_collection.update_one(user, {"$push": {"rented_game_ids": game["_id"]}})
        except:
            return "failure"

        return f"{game_title} rented successfully"

    def return_game(self, user: dict, game_title: str) -> str:
        """
        Returns a rented game

        :param user: The user object
        :param game_title: The title of the game to be returned
        :return: (str) "success" if the return process was successful, and "failure" if the return process failed.
        """
        # Check valid input
        if not game_title or not user:
            return "failure"
        if len(user) == 0 or len(game_title) == 0:
            return "failure"
        try:
            user = self.user_collection.find_one({'username': user['username']})
            rented = user["rented_game_ids"]
        except:
            if user:
                # No rented games
                return f"{game_title} was not rented by you"
        finally:
            if not user:
                return "failure"
        # Get the game
        try:
            rented_games = user["rented_game_ids"]
            games = self.game_collection.aggregate([
                {
                    "$match": {
                        "_id": {"$in": rented_games},
                        "title": game_title
                    }
                }
            ])
        except:
            return "failure"
        if not games.alive:
            return f"{game_title} was not rented by you"
        # Game found
        game = games.next()
        try:
            # Mark the game as not rented in the game collection
            self.game_collection.update_one(game, {"$set": {"is_rented": False}})
            # Remove the game id from the user's rented games ids list
            self.user_collection.update_one(
                user,
                {"$pull": {"rented_game_ids": game["_id"]}}
            )
        except:
            return "failure"
        return f"{game_title} returned successfully"

    def recommend_games_by_genre(self, user: dict) -> list:
        """
        Recommends games based on the user's rented game genre. Don’t recommend games that
        are already owned, it is allowed to recommend games that are rented by others.

        :param user: dict - The user object.
        :return: list of strings: A list containing recommended game titles based on genre
        """
        # Check valid input
        if user is None:
            return ["No games rented"]
        if len(user) == 0:
            return ["No games rented"]
        user = self.user_collection.find_one({'username': user['username']}, {"_id": 0, "rented_game_ids": 1})
        if not user:
            return ["No games rented"]
        # Get the list of games rented by the user from the user object
        rented_games = user["rented_game_ids"]
        games = self.game_collection.aggregate([
                                    {
                                        "$match": {
                                            "_id": {"$in": rented_games}
                                        }
                                    }
                                    ])
        # If no games are rented, return "No games rented"
        if not games.alive:
            return ["No games rented"]
        # Select a genre randomly from the pool of rented games, taking into account the probability distribution
        genres = []
        owned_titles = []
        for game in games:
            genres.extend(game["genres"])
            owned_titles.append(game["title"])
        random_genre = random.choice(genres)
        # Query the game collection to find 5 random games with the chosen genre
        random_games = self.game_collection.aggregate([
            {"$match": {"genres": random_genre,
                        "title": {"$nin": owned_titles}}},
            {"$sample": {"size": 5}}
        ])
        # Return the titles as a list with 5 random games
        titles = []
        for game in random_games:
            titles.append(game["title"])
        return titles

    def recommend_games_by_name(self, user: dict) -> list:
        """
        Recommends games based on random user's rented game name. Don’t recommend games
        that are already owned, it is allowed to recommend games that are rented by others

        :param user: (dict) The user object.
        :return: list of strings: A list containing recommended game titles based on similarity
        """
        # Check valid input
        if user is None:
            return ["No games rented"]
        if len(user) == 0:
            return ["No games rented"]
        user = self.user_collection.find_one({'username': user['username']}, {"_id": 0, "rented_game_ids": 1})
        if not user:
            return ["No games rented"]
        # Get the list of games rented by the user from the user object
        rented_games = user["rented_game_ids"]
        games = self.game_collection.aggregate([
            {
                "$match": {
                    "_id": {"$in": rented_games}
                }
            }
        ])
        # If no games are rented, return "No games rented"
        if not games.alive:
            return ["No games rented"]
        # Get the list of games rented by the user from the user object
        owned_titles = []
        for game in games:
            owned_titles.append(game["title"])
        # Choose a random game from the rented games.
        random_game = random.choice(owned_titles)
        games_titles = []
        games = self.game_collection.find()
        if games.alive:
            for game in games:
                if game["title"] not in owned_titles:
                    games_titles.append(game["title"])
                # games_titles.append(game["title"])
        if len(games_titles) == 0:
            return []
        # Compute TF-IDF vectors for all game titles and the chosen title
        # vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([random_game] + games_titles)
        # Compute cosine similarity between the TF-IDF vectors of the chosen title and all other games
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        # Create a DataFrame with game titles and their cosine similarities
        df = pd.DataFrame({'title': games_titles, 'cosine_sim': cosine_sim})
        # Sort the titles based on cosine similarity and return the top 5 recommended titles as a list
        df = df.sort_values(by='cosine_sim', ascending=False)
        top_recommendations = df['title'].head(5).tolist()
        return top_recommendations, df

    def find_top_rated_games(self, min_score) -> list:
        """
        Returns games with a user score higher than or equal to the score received
        Returns:
            List: list of all games (only titles and user scores) with a user score of at least
            min_score. Each game is represented as a dictionary and the return format is for
            example: [{'title': 'A', 'user_score': 9.0}, {'title': 'B', 'user_score': 8.9}]

        :param min_score: (double)
        :return: List: list of all games (only titles and user scores) with a user score of at least min_score
        """
        games = self.game_collection.aggregate([
            {"$match": {"user_score": {"$gte": min_score}}},
             {"$project": {"title": 1, "user_score": 1, "_id": 0}},
            {"$sort": {"user_score": -1}}
        ])
        out = []
        for game in games:
            out.append(game)
        return out

    def decrement_scores(self, platform_name) -> None:
        """
        Lowers user score by 1 for games whose platform is the received platform name. It can be
        assumed that platform names whose user scores for their games are at least 1 will be received.

        :param platform_name: (str) the name of the gaming platform
        :return: None
        """
        if platform_name is None:
            return
        # Get the games from that platform
        games = self.game_collection.aggregate([
            {"$match": {"platform": platform_name}},
            {"$project": {"_id": 1}}
        ])
        games_id = [game["_id"] for game in games]
        self.game_collection.update_many(
            {"_id": {"$in": games_id}},
            [{"$set": {"user_score": {"$round": [{"$subtract": ["$user_score", 1]}, 1]}}}])

    def get_average_score_per_platform(self) -> dict:
        """
        Calculate the average user score for games on each platform. The function returns a
        dictionary that maps each platform to its average user score.

        :return: Dictionary that maps each platform name to its average user score
        """
        # Use aggregation - Group the records according to the platform it belong to & Computers per platform average user score
        avg_scores = self.game_collection.aggregate([
            {"$group": {"_id": "$platform", "average_score": {"$avg": "$user_score"}}}
        ])
        # Returns a dictionary that maps the average user score to each platform
        d = dict()
        for game in avg_scores:
            d[game["_id"]] = round(game["average_score"], 3)
        return d

    def get_genres_distribution(self) -> dict:
        """
        Count the number of games in each genre. The function returns a dictionary that maps each genre to its amount games.

        :return: (dict) Dictionary that maps each genre to its amount games.
        """
        # Use aggregation
        geners_count = self.game_collection.aggregate([
            {"$match": {"genres": {"$exists": True}}},
            {"$project": {"genres": 1}},
            {"$unwind": "$genres"},  # The genera arrays should be broken down into individual documents
            {"$group": {"_id": "$genres", "count": {"$sum": 1}}}  # Group the records according to the genres to which they belong.
        ])
        # Returns a dictionary that maps the number of games to each genre.
        d = dict()
        for genre in geners_count:
            d[genre["_id"]] = genre["count"]
        return d
