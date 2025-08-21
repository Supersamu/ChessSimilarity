"""
get_gm_games.py
Given a list of Chess Grandmasters, this script retrieves and saves an excerpt of their games using the Lichess API.
"""

import berserk
from gm_usernames import Lichess_names

# I did not find a way to get Usernames from FIDE ids, so I will use this list: 


with open('src/lichess_data_loading/api_token.txt') as f:
    token = f.read()

session = berserk.TokenSession(token)
client = berserk.Client(session)

for lichess_name in Lichess_names:
    print(f"Retrieving games for {lichess_name}")
    player = client.lichess.get_user(lichess_name)
    print(player)
