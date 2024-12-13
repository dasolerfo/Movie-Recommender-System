#import train as t
import numpy as np
import train3 as md
import dataset as dt
import torch

class UserInterface:

    def __init__(self, current_user = 1):
        #self.user_embeddings = t.user_embeddings
        #self.movie_embeddings = t.movie_embeddings
        self.user = 0
        self.movies = dt.readMovies()
        self.current_user = current_user
        self.ratings = dt.readRatings()
        self.model = md.loadModel(len(self.ratings['userId'].unique()), len(self.ratings['movieId'].unique()), 128)
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(self.ratings['movieId'].unique())}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        self.run_ui()


    def run_ui(self):
        """Keeps running the UI until the user decides to quit."""
        while True:
            self.print_main_menu()
            self.get_user_input()

    def change_user(self):
        """Switches users with input validation."""
        while True:
            try:
                new_user = int(input("Enter New User (1-610): "))
                if self.validate_user(new_user):
                    self.current_user = new_user
                    print(f"User changed to {self.current_user}")
                    # Perform a forward pass with the new user embedding and a random movie embedding
                    #user_emb = self.user_embeddings[self.current_user - 1]
                    #movie_idx = np.random.randint(len(self.movie_embeddings))
                    #movie_emb = self.movie_embeddings[movie_idx]
                    #_, _, _, _, _, _, y_pred = t.forward(user_emb, movie_emb)
                    #print(f"Predicted rating for a random movie: {y_pred[0][0]:.2f}")
                    break
                else:
                    print("Invalid user ID. Please enter a number between 1 and 610.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

    def validate_user(self, user):
        """Checks if the user ID is between 1 and 610."""
        return 1 <= user <= 610

    def print_main_menu(self):
        """Displays the main menu."""
        MAIN_MENU = f"""\nCurrent User: {self.current_user}        
1. Give Rating
2. Get Recommendations
3. Change User
4. Quit
"""
        print(MAIN_MENU)

    def get_user_input(self):
        """Handles user navigation from the main menu."""
        while True:
            try:
                user_input = int(input("Please enter an option (1-4): "))
                if self.validate_input(user_input):
                    if user_input == 1:
                        self.give_rating()
                    elif user_input == 2:
                        self.get_recommendation()
                    elif user_input == 3:
                        self.change_user()
                    elif user_input == 4:
                        print("Exiting the program.")
                        return
                else:
                    print("Invalid option. Please enter a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

    def validate_input(self, user_input):
        """Validates that the user input is an integer between 1 and 4."""
        return 1 <= user_input <= 4

    def give_rating(self):
        print("Select which movie yo would like to watch: ( Write the id is showed in the movie dataset :) )")
        id = int(input())
        print("Which rating do you want to give it to the movie?")
        rating = float(input())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_id_tensor = torch.tensor([self.current_user]).to(device)
        idx = self.item_mapping.get(id)
        item_id_tensor = torch.tensor([idx]).to(device)
        with torch.no_grad():  # No necessitem gradients per a la predicció
            y_pred = self.model(user_id_tensor, item_id_tensor).item()  # Obtenim el valor de la predicció
        print(f"Predicted Rating: {y_pred*5:.2f}")
        self.model.adapt_user_embedding(user_id_tensor, item_id_tensor, rating)

        similar_movies, scores = self.model.find_similar_movies(idx, 5)

        print(f"\nSimilar movies to {self.movies.loc[self.movies["movieId"] == id, 'title'].iloc[0]} : ")
        for i, (sim_movie, score) in enumerate(zip(similar_movies, scores)):
            print(f"{i + 1}. Movie Title:  {self.movies.loc[self.movies["movieId"] == self.reverse_item_mapping.get(sim_movie.item()), "title"].iloc[0]} - Similarity: {score.item():.4f}")



        """Placeholder for the give rating functionality."""

        print("Rating Success")

    def get_recommendation(self):
        """Recommend top movies for the current user based on their embedding."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_id_tensor = torch.tensor([self.current_user]).to(device)



        predictions = []

        for index, movie in self.movies.iterrows():
            if index == 0:
                continue

            idx = self.item_mapping.get(movie[0])
            if idx is None:
                continue
            item_id_tensor = torch.tensor([idx]).to(device)

            with torch.no_grad():  # No necessitem gradients per a la predicció
                y_pred = self.model(user_id_tensor, item_id_tensor).item()  # Obtenim el valor de la predicció
            predictions.append((movie[0], y_pred * 5))

        # Sort movies by predicted rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 5 Movie Recommendations:")
        for i in range(5):
            movie_idx, rating = predictions[i]
            peli = self.movies[self.movies["movieId"] == movie_idx]["title"]
            print(f"Movie: {peli.iloc[0]}: ID: {movie_idx}: Predicted Rating: {rating:.2f}")


if __name__ == "__main__":
    test = UserInterface()
