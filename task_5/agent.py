# Skeleton for Agent class
import random
import torch
import numpy as np


class Agent:

    def __init__(self, player_id: int):
        self.player_id = player_id  
        print(f"Player ID: {self.player_id}")
        # Initialize home_planet and enemy_planet as None
        # They will be set on the first call to get_action and never changed again
        self.home_planet = None
        self.enemy_planet = None

    def get_action(self, obs: dict) -> dict:
        """
        Main function, which gets called during step() of the environment.

        Observation space:
            game_map: whole grid of board_size, which already has applied visibility mask on it
            allied_ships: an array of all currently available ships for the player. The ships are represented as a list:
                (ship id, position x, y, current health points, firing_cooldown, move_cooldown)
                - ship id: int [0, 1000]
                - position x: int [0, 100]
                - position y: int [0, 100]
                - health points: int [1, 100]
                - firing_cooldown: int [0, 10]
                - move_cooldown: int [0, 3]
            enemy_ships: same, but for the opposing player ships
            planets_occupation: for each visible planet, it shows the occupation progress:
                - planet_x: int [0, 100]
                - planet_y: int [0, 100]
                - occupation_progress: int [-1, 100]:
                    -1: planet is unoccupied
                    0: planet occupied by the 1st player
                    100: planet occupied by the 2nd player
                    Values between indicate an ongoing conflict for the ownership of the planet
            resources: current resources available for building

        Action space:
            ships_actions: player can provide an action to be executed by every of his ships.
                The command looks as follows:
                - ship_id: int [0, 1000]
                - action_type: int [0, 1]
                    0 - move
                    1 - fire
                - direction: int [0, 3] - direction of movement or firing
                    0 - right
                    1 - down
                    2 - left
                    3 - up
                - speed (not applicable when firing): int [0, 3] - a number of fields to move
            construction: int [0, 10] - a number of ships to be constructed

        :param obs:
        :return:
        """
    
        game_map = obs.get('map')
        allied_ships = obs.get('allied_ships')
        enemy_ships = obs.get('enemy_ships')
        planets_occupation = obs.get('planets_occupation')
        resources = obs.get('resources')

        # Set home_planet and enemy_planet only once on the first call
        if self.home_planet is None and planets_occupation:
            self.home_planet = planets_occupation[0]
            
            # Determine enemy planet based on home planet location
            if self.home_planet[0] == 9:
                self.enemy_planet = (90, 90)
            else:
                self.enemy_planet = (9, 9)
                
            print(f"Home planet set to: {self.home_planet}")
            print(f"Enemy planet set to: {self.enemy_planet}")

        allied_ship_temp = {}
        for ship in allied_ships:
            allied_ship_temp[ship[0]] = ship
        
        obs["allied_ships"] = allied_ship_temp

        action_list = []
        for ship in allied_ships:
            if ship[0] % 2:
                action_list.append(get_defense_action(obs, ship[0], self.home_planet))
            else:
                action_list.append(get_offense_action(obs, ship[0], self.enemy_planet))

        return {
            "ships_actions": action_list,
            "construction": 10
        }

    def load(self, abs_path: str):
        """
        Function for loading all necessary weights for the agent. The abs_path is a path pointing to the directory,
        where the weights for the agent are stored, so remember to join it to any path while loading.

        :param abs_path:
        :return:
        """
        ...
        # self._model = torch.load(
        #     "/home/aleksander/Desktop/dev/tasks-2025/task_5/example_weights/example_weights.pt",
        #     weights_only=True,
        # )

    def eval(self):
        """
        With this function you should switch the agent to inference mode.

        :return:
        """
        ...

    def to(self, device):
        """
        This function allows you to move the agent to a GPU. Please keep that in mind,
        because it can significantly speed up the computations and let you meet the time requirements.

        :param device:
        :return:
        """
        ...


def get_offense_action(obs: dict, idx: int, enemy_planet: tuple) -> list[int]:
    try:
        ship = obs["allied_ships"][idx]
    except IndexError:
        print(f"IndexError: No allied ship at index {idx}")
        return []
    
    ship_id, ship_x, ship_y = ship[0], ship[1], ship[2]
    enemy_x, enemy_y = enemy_planet[0], enemy_planet[1]
    
    # Only try to shoot if firing cooldown is 0
    if ship[4] == 0:  # ship[4] is firing_cooldown
        for enemy in obs["enemy_ships"]:
            choice = shoot_enemy_if_in_range(enemy, ship)
            if choice:
                return choice
    
    # If we can't shoot or firing is on cooldown, move towards enemy planet
    # Determine direction to move towards enemy planet
    dx = enemy_x - ship_x
    dy = enemy_y - ship_y
    
    # Choose direction based on which axis has greater distance
    if abs(dx) > abs(dy):
        # Move horizontally
        if dx > 0:
            direction = 0  # Right
        else:
            direction = 2  # Left
    else:
        # Move vertically
        if dy > 0:
            direction = 1  # Down
        else:
            direction = 3  # Up
    
    # Calculate movement speed based on cooldown
    # Always move with at least speed 1, regardless of cooldown
    # If the ship is in an asteroid field (cooldown > 0), it will move at 1/3 the normal speed
    # due to the game's cooldown mechanics (cooldown of 3 makes ship 3x slower)
    if ship[5] == 0:  # No cooldown - can move at full speed (up to 3)
        speed = 3  # Maximum speed when no cooldown
    else:
        # Ship has cooldown but can still move - just slower
        # We always want to move with speed 1 even with cooldown
        speed = 1
    
    return [ship_id, 0, direction, speed]


def get_defense_action(obs: dict, idx: int, home_planet: tuple) -> list[int]:
    ship = obs["allied_ships"][idx]

    for enemy in obs["enemy_ships"]:
        choice = shoot_enemy_if_in_range(enemy, ship)
        if choice:
            return choice

    return move_randomly_around_home(ship, home_planet[0], home_planet[1])


def shoot_enemy_if_in_range(enemy, ship) -> list[int]:
    """
    Check if an enemy ship is within firing range (3 tiles) and directly aligned
    (same row or column) with our ship.
    
    Ship position: (ship[1], ship[2])
    Enemy position: (enemy[1], enemy[2])
    
    Returns a firing action if enemy is in range, otherwise an empty list.
    """
    ship_x, ship_y = ship[1], ship[2]
    enemy_x, enemy_y = enemy[1], enemy[2]
    
    # Check if ships are in the same row (y-coordinate)
    if ship_y == enemy_y:
        # Enemy is to the right
        if 0 < enemy_x - ship_x <= 3:
            return [ship[0], 1, 0]  # Shoot right
        # Enemy is to the left
        if 0 < ship_x - enemy_x <= 3:
            return [ship[0], 1, 2]  # Shoot left
    
    # Check if ships are in the same column (x-coordinate)
    if ship_x == enemy_x:
        # Enemy is below
        if 0 < enemy_y - ship_y <= 3:
            return [ship[0], 1, 1]  # Shoot down
        # Enemy is above
        if 0 < ship_y - enemy_y <= 3:
            return [ship[0], 1, 3]  # Shoot up
    
    # Enemy not in range or not aligned
    return []


def move_randomly_around_home(ship, home_x, home_y, max_distance=7) -> list[int]:
    """
    Poruszanie się losowo w obszarze max_distance wokół planety macierzystej.
    """
    ship_x, ship_y = ship[1], ship[2]

    # Losowy wybór kierunku
    direction = random.randint(0, 3)

    # Przewidywana nowa pozycja
    new_x = ship_x + (1 if direction == 0 else -1 if direction == 2 else 0)
    new_y = ship_y + (1 if direction == 1 else -1 if direction == 3 else 0)

    # Sprawdzenie, czy nowa pozycja mieści się w dozwolonym obszarze wokół planety
    if abs(new_x - home_x) + abs(new_y - home_y) <= max_distance:
        # Always use speed 1 for defensive ships around home
        return [ship[0], 0, direction, 1]  # Ruch o 1 pole w danym kierunku

    # Jeśli ruch wykracza poza obszar, zostań na miejscu
    return [ship[0], 0, random.randint(0, 3), 0]  # Nie ruszaj się, jeśli brak dobrego ruchu

