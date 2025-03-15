# Skeleton for Agent class
import random
import torch


class Agent:
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

        action_list = []
        for ship in allied_ships:
            if ship[0] % 2:
                action_list.append(get_defense_action(obs, ship[0]))

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


def get_defense_action(obs: dict, idx: int) -> list[int]:
    ship = obs["allied_ships"][idx]

    home_planet = obs["planet_occupation"][0]

    position = obs["map"][ship[1]][ship[2]]

    neighborhood = get_bounds(position, 3)
    visibility = get_bounds(position, 5)

    choice = None

    for enemy in obs["enemy_ships"]:
        choice = shoot_enemy_if_in_range(enemy, ship)
        if choice:
            return choice

    return move_randomly_around_home(ship, home_planet[0], home_planet[1])



def get_bounds(position, size):
    r_min, r_max = max(position[0] - size, 0), min(position[0] + size, 100)
    c_min, c_max = max(position[1] - size, 0), min(position[1] + size, 100)

    return r_min, r_max, c_min, c_max


def shoot_enemy_if_in_range(enemy, ship) -> list[int]:
    if 0 < ship[1] - enemy[1] <= 3:  # enemy on the left
        return [ship[0], 1, 2]

    if 0 < enemy[1] - ship[1] <= 3:  # enemy on the right
        return [ship[0], 1, 0]

    if 0 < ship[2] - enemy[2] <= 3:  # enemy up
        return [ship[0], 1, 3]

    if 0 < enemy[2] - ship[2] <= 3:  # enemy down
        return [ship[0], 1, 1]

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
        return [ship[0], 0, direction, 1]  # Ruch o 1 pole w danym kierunku

    # Jeśli ruch wykracza poza obszar, zostań na miejscu
    return [ship[0], 0, random.randint(0, 3), 0]  # Nie ruszaj się, jeśli brak dobrego ruchu

