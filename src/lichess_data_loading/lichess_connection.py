import berserk

if __name__ == "__main__":
    # initial testing script for API functionality
    with open('src/lichess_data_loading/api_token.txt') as f:
        token = f.read()

    session = berserk.TokenSession(token)
    client = berserk.Client(session)
    for game in client.games.export_by_player(username="Supersamu", as_pgn=True, max=1, rated=True, 
                                perf_type=["blitz", "classical", "rapid"]):
        x = game.split("\n")
        print("metadata")
        print(x[0])  # Print the first line of the PGN, which contains metadata, example: [Event "Rated Blitz game"]
        print("Link to game")
        print(x[1])  # Print the second line of the PGN, which contains the link to the game, example: [Site "https://lichess.org/zdLZJdPH"]
        print("Date")
        print(x[2])  # Print the third line of the PGN, which contains the Date, example: [Date "2024.11.28"]
        print("White Username")
        print(x[3])  # Print the fourth line of the PGN, which contains the White Username, example: [White "lEONARDOeH"]
        print("Black Username")
        print(x[4])  # Print the fifth line of the PGN, which contains the Black Username, example: [Black "Supersamu"]
        print("Result")
        print(x[5])  # Print the sixth line of the PGN, which contains the Result, example: [Result "1-0"]
        print("Game ID")
        print(x[6])  # Print the seventh line of the PGN, which contains the Game ID, example: [GameId "zdLZJdPH"]
        print("Date of Game")
        print(x[7])  # Print the eighth line of the PGN, which contains the Date of Game, example: [UTCDate "2024.11.28"]
        print("Time of Game")
        print(x[8])  # Print the ninth line of the PGN, which contains the Time of Game, example: [UTCTime "01:40:40"]
        print("White Elo")
        print(x[9])  # Print the tenth line of the PGN, which contains the White Elo, example: [WhiteElo "1249"]
        print("Black Elo")
        print(x[10])  # Print the eleventh line of the PGN, which contains the Black Elo, example: [BlackElo "1265"]
        print("Difference in white player rating after the game")
        print(x[11])  # Print the twelfth line of the PGN, which contains the difference in white player rating after the game, example: [WhiteRatingDiff "+5"]
        print("Difference in black player rating after the game")
        print(x[12])  # Print the thirteenth line of the PGN, which contains the difference in black player rating after the game, example: [BlackRatingDiff "-69"]
        print("Chess Variant")
        print(x[13])  # Print the fourteenth line of the PGN, which contains the chess variant, example: [Variant "Standard"]
        print("Time control")
        print(x[14])  # Print the fifteenth line of the PGN, which contains the time control, example: [TimeControl "60+3"]
        print("Opening Type")
        print(x[15])  # Print the sixteenth line of the PGN, which contains the opening type, example: [ECO "C57"]
        print("Reason of Game end")
        print(x[16])  # Print the seventeenth line of the PGN, which contains the reason of game end, example: [Termination "Time forfeit"]
        print("Empty Line")
        print(x[17])  # Print the eighteenth line of the PGN, which is an empty line: 
        print("Game Moves")
        print(x[18])  # Print the nineteenth line of the PGN, which contains the game moves, example: 1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 d5 5. exd5 Nxd5 6. Qf3 Qxg5 7. Bxd5 Qf6 8. Qb3 1-0

