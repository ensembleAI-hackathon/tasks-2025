import numpy as np
VERSION = '1.0.1'

# You may change this, to adjust the window into your screen
WINDOW_SIZE = 800


# Gameplay settings
MAX_RESOURCES = 1000
RESOURCE_PRODUCTION_DIVISOR = 4
MAX_SHIPS = 1000
MAP_MAX_VALUE = 255
BASE_SHIP_SPEED = 1
SHIP_COST = np.array([100, 100, 100, 100])
SHIP_DAMAGE = 30
SHIP_HEALING_SPEED = 2
MAX_SHIP_FIRE_RANGE = 8
ASTEROID_SPEED_FACTOR = 0.25
IONIZED_FIELD_SPEED_FACTOR = 3.0
SHIP_OCCUPATION_RANGE = 2
FIRING_COOLDOWN = 10
MOVE_COOLDOWN = 3
ASTEROID_DAMAGE = 3
VISION_RANGE = 5
PLANET_VISION_RANGE = 9

# Map generation settings
BOARD_SIZE = 100
N_PLANETS = 7
FRAC_OF_ASTEROID_AREA = 0.2
FRAC_OF_IONIZED_AREA = 0.003
PLANETS_DIAMETER = 9
PLANETS_OFFSET = 5
PLANETS_DISTANCE = 20
PLAYER_1_LOCATION = PLANETS_OFFSET + PLANETS_DIAMETER//2               # position with regard to the tiles on board
PLAYER_2_LOCATION = BOARD_SIZE - (PLANETS_OFFSET + PLANETS_DIAMETER//2 + 1)
PLAYER_1_ORIGIN = np.array([PLAYER_1_LOCATION, PLAYER_1_LOCATION], dtype=int)
PLAYER_2_ORIGIN = np.array([PLAYER_2_LOCATION, PLAYER_2_LOCATION], dtype=int)

# Rendering settings
BORDER_WIDTH = 75
GUI_SIZE = 110
TILE_SIZE = WINDOW_SIZE // BOARD_SIZE
ASTEROID_SIZE = 25
PLAYER_ICON_SIZE = 50                   # Icon visible on the board
PLAYER_ICON_TEAM_NAME_SIZE = 60         # Icon next to the team name
SHIP_SIZE = 25
SIDE_SHIP_SIZE = 35
EFFECT_IONIZED_FIELD_SIZE = 20
RF_MARKER_SIZE = int(TILE_SIZE * 0.8)
RF_MARKER_ADJUSTMENT = TILE_SIZE//2 - RF_MARKER_SIZE//2
RF_ICON_SIZE = 40
RF_BAR_SIZE = 80
FLAG_SIZE = 30
FLAG_ICON_ADJUSTMENT = TILE_SIZE//2 - FLAG_SIZE//2
OCCUPATION_BAR_SIZE = 60
OCCUPATION_BAR_COLOR_MARGIN = int(OCCUPATION_BAR_SIZE * (3 / 41))
OCCUPATION_SPEED = 2
ABS_PLAYER_1_ICON_POS = PLAYER_1_LOCATION * TILE_SIZE + TILE_SIZE//2 - PLAYER_ICON_SIZE//2
ABS_PLAYER_2_ICON_POS = PLAYER_2_LOCATION * TILE_SIZE + TILE_SIZE//2 - PLAYER_ICON_SIZE//2
TEAM_NAMES_MARGIN = 50
EFFECT_DEATH_SIZE = 50
EFFECT_DEATH_ADJUSTMENT = TILE_SIZE//2 - EFFECT_DEATH_SIZE//2
EFFECT_HEALING_SIZE = 50
EFFECT_HEALING_ADJUSTMENT = TILE_SIZE//2 - EFFECT_HEALING_SIZE//2
EFFECT_FIRING_SIZE = 50
EFFECT_FIRING_ADJUSTMENT = TILE_SIZE//2 - EFFECT_FIRING_SIZE//2
EFFECT_CAPTURE_SIZE = 50
EFFECT_CAPTURE_ADJUSTMENT = TILE_SIZE//2 - EFFECT_CAPTURE_SIZE//2
EFFECT_SPACE_JUMP_SIZE = 40
EFFECT_SPACE_JUMP_ADJUSTMENT = TILE_SIZE//2 - EFFECT_SPACE_JUMP_SIZE//2

"""
Each map field is coded on 8 bits.
xxxxxxx0 - space
xxx000x1 - land
xxxxxx1x - speed decrease modifier (rough terrain or asteroids)
xxxxx1xx - speed boost modifier (ionized field)
xx001xx1 - resource field 1
xx010xx1 - resource field 2
xx011xx1 - resource field 3
xx111xx1 - resource field 4
00xxxxxx - field unoccupied
x1xxxxxx - field captured by player 1
1xxxxxxx - field captured by player 2

Examples:
Regular unoccupied land: 00000001 = 1
Asteroid field: 00000010 = 2
Rough unoccupied land: 00000011 = 3
Ionized field: 0000100 = 4
Unoccupied regular resource field 2: 00010001 = 17

Occupation:
(value) & 64 = 64       - occupied by player 1
(value) & 128 = 128     - occupied by player 2
(value) & 57 = 1        - check if tile is a land without resource field
"""

RF_ID_TO_CODING = {
    0: 9,
    1: 17,
    2: 25,
    3: 57
}

RF_CODING_TO_ID = {
    9: 0,
    17: 1,
    25: 2,
    57: 3
}

# Positions on a planet scheme, where the resource fields are
RF_COORDS = [(1, 4), (2, 3), (2, 4), (2, 5),
             (3, 2), (4, 1), (4, 2), (5, 2),
             (3, 6), (4, 7), (4, 6), (5, 6),
             (6, 3), (6, 4), (6, 5), (7, 4)]

TEAMS_NAMES = {
    i: f'Team {i}' for i in range(50)
}

MOVEMENT_DIRECTIONS = np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
], dtype=np.int8)


VISION_ADD_MASK_OLD = np.array([
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
], dtype=bool)


VISION_ADD_MASK = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
], dtype=bool)