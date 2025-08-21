"""
get_gm_games.py
Given a list of Chess Grandmasters, this script retrieves and saves an excerpt of their games using the Lichess API.
"""

import berserk
from gm_usernames import Lichess_names
import json
from collections import defaultdict
import os
import time

with open('src/lichess_data_loading/api_token.txt') as f:
    token = f.read()

session = berserk.TokenSession(token)
client = berserk.Client(session)


def get_game_moves(lichess_name, color):
    games_received = False
    fail_counter = 0
    while not games_received and fail_counter < 3:
        game_moves = []
        try:
            games = client.games.export_by_player(username=lichess_name, as_pgn=True, max=50,
                                                   rated=True, perf_type=["blitz"], color=color)
            for game in games:
                game_info = game.split("\n")
                moves = game_info[-1]
                game_moves.append(moves)
            games_received = True
            return game_moves
        except berserk.exceptions.ResponseError as re:
            if re.status_code == 429:
                print(f"Rate limit exceeded for {lichess_name}. Repeating after one minute.")
                time.sleep(61)  # Wait before retrying according to the API rate limit
                fail_counter += 1
            else:
                print(f"Error retrieving games for {lichess_name}: {re}")
                games_received = True  # Stop retrying on other errors
    return game_moves


if os.path.exists('src/lichess_data_loading/gm_games.json'):
    with open('src/lichess_data_loading/gm_games.json') as f:
        data = json.load(f)
        games_white = defaultdict(list, data.get("white", {}))
        games_black = defaultdict(list, data.get("black", {}))
else:
    games_white = defaultdict(list)
    games_black = defaultdict(list)

for lichess_name in Lichess_names:
    # Retrieve white games
    if lichess_name not in games_white:
        print(f"Retrieving white games for {lichess_name}")
        white_games = get_game_moves(lichess_name, "white")
        games_white[lichess_name] = white_games
        # Save progress
        with open('src/lichess_data_loading/gm_games.json', 'w') as f:
            json.dump({"white": games_white, "black": games_black}, f)

    # Retrieve black games
    if lichess_name not in games_black:
        print(f"Retrieving black games for {lichess_name}")
        black_games = get_game_moves(lichess_name, "black")
        games_black[lichess_name] = black_games
        # Save progress
        with open('src/lichess_data_loading/gm_games.json', 'w') as f:
            json.dump({"white": games_white, "black": games_black}, f)
    