import train as t
import numpy as np


class UserInterface:

    def __init__(self, current_user = 1):
        self.user_embeddings = t.user_embeddings
        self.movie_embeddings = t.movie_embeddings
        self.current_user = current_user
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
                    user_emb = self.user_embeddings[self.current_user - 1]
                    movie_idx = np.random.randint(len(self.movie_embeddings))
                    movie_emb = self.movie_embeddings[movie_idx]
                    _, _, _, _, _, _, y_pred = t.forward(user_emb, movie_emb)
                    print(f"Predicted rating for a random movie: {y_pred[0][0]:.2f}")
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
        """Placeholder for the give rating functionality."""
        print("Rating Success")

    def get_recommendation(self):
        """Recommend top movies for the current user based on their embedding."""
        user_emb = self.user_embeddings[self.current_user - 1]
        predictions = []

        for idx, movie_emb in enumerate(self.movie_embeddings):
            _, _, _, _, _, _, y_pred = t.forward(user_emb, movie_emb)
            predictions.append((idx, y_pred[0][0]))

        # Sort movies by predicted rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 5 Movie Recommendations:")
        for i in range(5):
            movie_idx, rating = predictions[i]
            print(f"Movie {movie_idx + 1}: Predicted Rating: {rating:.2f}")


if __name__ == "__main__":
    test = UserInterface()
