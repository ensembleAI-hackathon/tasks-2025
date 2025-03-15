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

        actions = []
        
        for ship in obs['allied_ships']:
            x, y, hp = ship[1], ship[2], ship[3]
            nearest_enemy = min(obs['enemy_ships'], key=lambda s: abs(ship[1] - s[1]) + abs(ship[2] - s[2]), default=(0, 0, 0))
            nearest_planet = min(obs['planets_occupation'].keys(), key=lambda p: abs(ship[1] - p[0]) + abs(ship[2] - p[1]), default=(0, 0))
            
            # Tworzymy tensor wejściowy dla modelu
            inputs = torch.tensor([[x, y, hp, abs(x - nearest_enemy[1]), abs(y - nearest_enemy[2]), obs['resources']]], dtype=torch.float32).to(self.device)
            
            # Model przewiduje najlepszą akcję
            with torch.no_grad():
                action_type = torch.argmax(self._model(inputs)).item()
            
            if action_type == 0:
                actions.append((ship[0], 0, random.randint(0, 3), random.randint(1, 3)))  # Ruch
            elif action_type == 1:
                actions.append((ship[0], 1, random.randint(0, 3)))  # Strzał
            else:
                actions.append((ship[0], 0, 0, 0))  # Kolonizacja

        return {"ships_actions": actions, "construction": min(10, obs['resources'] // 100)}



    def load(self, abs_path: str):
        """
        Function for loading all necessary weights for the agent. The abs_path is a path pointing to the directory,
        where the weights for the agent are stored, so remember to join it to any path while loading.

        :param abs_path:
        :return:
        """
        self._model = torch.load(abs_path)

    def eval(self):
        """
        With this function you should switch the agent to inference mode.

        :return:
        """
        self._model.eval()

    def to(self, device):
        """
        This function allows you to move the agent to a GPU. Please keep that in mind,
        because it can significantly speed up the computations and let you meet the time requirements.

        :param device:
        :return:
        """
        self._model.to(device)