import numpy as np

# Consider a simplified chess game (4x4 board, only pawns):

# Pawn array of white player
pawns_white = np.array([[0, 0, 0, 0],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

pawns_black = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0]])

# the board_array is then created by stacking the two pawn arrays
original_position = board_array = np.stack((pawns_white, pawns_black))
print(board_array.shape)  # 2, 4, 4

# the subsequent position has one pawn of white moved one step forward:
pawns_white_new = np.array([[0, 0, 0, 0],
                            [0, 1, 1, 1],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0]])

# the new board is created by stacking the updated pawn arrays
subsequent_position = board_array_new = np.stack((pawns_white_new, pawns_black))
print(board_array_new.shape)  # 2, 4, 4

# from white's perspective, the board as is is not modified
# from black's perspective, the board needs to be flipped such that the black pawns are shown first

flipped_original_position = np.flip(original_position, axis=[0, 1])
flipped_subsequent_position = np.flip(subsequent_position, axis=[0, 1])
print(flipped_original_position[0])
"""
[[0 0 0 0]
[1 1 1 1]
[0 0 0 0]
[0 0 0 0]]
"""
print(flipped_original_position[1])
"""
[[0 0 0 0]
[0 0 0 0]
[1 1 1 1]
[0 0 0 0]]
"""

print(flipped_subsequent_position[0])
"""
[[0 0 0 0]
[1 1 1 1]
[0 0 0 0]
[0 0 0 0]]
"""
print(flipped_subsequent_position[1])
"""
[[0 0 0 0]
[1 0 0 0]
[0 1 1 1]
[0 0 0 0]]
"""

# Now, games from black's perspective and white's perspective can be handled by the neural network in the exact same way.
# The opponent's pawns are always at the same index along the first axis, no matter the perspective.
# The opponent's pawns are also always at the 'bottom' of the board from the player's perspective.
