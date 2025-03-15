from typing import Any, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.core import RenderFrame


from octospace.envs.game_config import (MAX_RESOURCES, MAP_MAX_VALUE,
                                        WINDOW_SIZE, MAX_SHIPS, BOARD_SIZE, VERSION, GUI_SIZE, BORDER_WIDTH,
                                        BASE_SHIP_SPEED, SHIP_COST,
                                        PLAYER_1_ORIGIN, PLAYER_2_ORIGIN, N_PLANETS, FIRING_COOLDOWN, MOVE_COOLDOWN,
                                        RESOURCE_PRODUCTION_DIVISOR)
from octospace.envs.map_assets import BORDER, BORDER_SCORE, generate_players_assets
from octospace.envs.map_generation import _generate_map, _generate_state_map, _add_base_planet_occupation, _reset_planets_occupation
from octospace.envs.rendering import (_render_planets, _render_planet_occupation, _render_ongoing_planet_capture, _render_players,
                       _render_ships, _render_turn, _render_background, _render_team_names, _render_resources,
                       _render_effects, _render_vision_debug, _render_score)
from octospace.envs.game_logic import (_ship_firing, _ship_movement, _ship_construction, _occupation_progress,
                        _change_ownership_of_planets, _ship_land_interaction, _decrease_cooldowns, _handle_ship_death,
                        _handle_visibility, _add_planet_visibility, _check_victory_conditions)
from octospace.envs.sound import setup_music_loop, get_new_track


class OctoSpaceEnv(gym.Env):
    """
    Args:
        render_mode: type of visualization, available options: human and rgb_array
        turn_on_music: turn on music and sound effects
        volume: change the volume of music and sound effects

    Observation Space:
        game_map: whole grid of board_size, which already has applied visibility mask on it
        allied_ships: an array of all currently available ships for the player. The ships are represented as a list:
            (ship id, position x, y, current health points, firing_cooldown, move_cooldown)
            - ship id: int [0, MAX_SHIPS]
            - position x: int [0, board_size]
            - position y: int [0, board_size]
            - health points: int [1, 100]
            - firing_cooldown: int [0, FIRING_COOLDOWN]
            - move_cooldown: int [0, MOVE_COOLDOWN]
        enemy_ships: same, but for the opposing player ships
        planets_occupation: for each visible planet, it shows the occupation progress:
            - -1 means no occupation
            - 0 means the planet is occupied by the 1st player
            - 100 means the planet is occupied by the 2nd player
            Planets are represented as: (planet_x, planet_y, occupation_progress)
        resources: current resources available for building

    Action Space:
        ships_actions: player can provide an action to be executed by every of his ships. The command looks as follows:
            (ship id, 0, direction, speed)
            (ship id, 1, direction)

            where 0 - ship movement, 1 - ship firing
        construction: int [0, MAX_RESOURCES // 100] - a number of ships to be constructed

    Construction of a new ship requires 100 units from each resource
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self,
                 player_1_id: int,
                 player_2_id: int,
                 render_mode: Optional[str] = None,
                 max_steps: int = 1000,
                 turn_on_music: bool = False,
                 volume: float = 0.25,
                 seed: Optional[int] = None
                 ):
        assert BOARD_SIZE > 30
        assert N_PLANETS >= 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self._turn_on_music = turn_on_music
        self.player_1_id = player_1_id
        self.player_2_id = player_2_id
        self.player_1_id_original = player_1_id
        self.player_2_id_original = player_2_id

        self._players_rendering_order = sorted([self.player_1_id, self.player_2_id])

        self.max_steps = max_steps
        self.volume = volume
        self.seed = seed
        self.render_mode = render_mode
        self.debug = False

        self.observation_space = spaces.Dict({
            player: spaces.Dict({
                "map": spaces.Box(0, MAP_MAX_VALUE, shape=(BOARD_SIZE, BOARD_SIZE), dtype=int),
                "allied_ships": spaces.Sequence(
                    space=spaces.Tuple(
                        spaces=[
                            spaces.Discrete(n=MAX_SHIPS, start=0),               # ship id
                            spaces.Discrete(n=BOARD_SIZE, start=0),              # position x
                            spaces.Discrete(n=BOARD_SIZE, start=0),              # position y
                            spaces.Discrete(n=100, start=1),                     # health points
                            spaces.Discrete(n=FIRING_COOLDOWN, start=0),         # has recently fired
                            spaces.Discrete(n=MOVE_COOLDOWN, start=0)            # has entered the asteroid field
                        ]
                    )
                ),
                "enemy_ships": spaces.Sequence(
                    space=spaces.Tuple(
                        spaces=[
                            spaces.Discrete(n=MAX_SHIPS, start=0),              # ship id
                            spaces.Discrete(n=BOARD_SIZE, start=0),             # position x
                            spaces.Discrete(n=BOARD_SIZE, start=0),             # position y
                            spaces.Discrete(n=100, start=1),                    # health points
                            spaces.Discrete(n=FIRING_COOLDOWN, start=0),         # has recently fired
                            spaces.Discrete(n=MOVE_COOLDOWN, start=0)            # has entered the asteroid field
                        ]
                    )
                ),
                "planets_occupation": spaces.Sequence(
                    space=spaces.Tuple(
                        spaces=[
                            spaces.Discrete(n=BOARD_SIZE, start=0),             # planet x
                            spaces.Discrete(n=BOARD_SIZE, start=0),             # planet y
                            spaces.Discrete(n=100, start=-1)                    # occupation progress
                        ]
                    )
                ),
                "resources": spaces.Box(0, MAX_RESOURCES)
            }) for player in ["player_1", "player_2"]
        })
        self.action_space = spaces.Dict({
            player: spaces.Dict({
                "ships_actions": spaces.Sequence(
                    space=spaces.OneOf(
                        spaces=[
                            spaces.Tuple(
                                spaces=[
                                    spaces.Discrete(n=MAX_SHIPS, start=0),                  # ship id
                                    spaces.Discrete(n=2, start=0),                          # action
                                    spaces.Discrete(n=4, start=0),                          # direction
                                    spaces.Discrete(n=BASE_SHIP_SPEED, start=0)             # speed
                                ]
                            ),
                            spaces.Tuple(
                                spaces=[
                                    spaces.Discrete(n=MAX_SHIPS, start=0),                  # ship id
                                    spaces.Discrete(n=2, start=0),                          # action
                                    spaces.Discrete(n=4, start=0),                          # direction
                                ]
                            )
                        ]
                    )
                ),
                "construction": spaces.Discrete(n=MAX_RESOURCES // np.max(SHIP_COST), start=0)      # number of ships to be constructed
            })
            for player in ["player_1", "player_2"]
        })

        self._map: np.ndarray = None
        self._state_ids = None
        self._planets_centers: np.ndarray = None

        # Contain the values between 0 and 100, indicating the occupation progress
        # 0 means the whole planet belongs to 1st player
        # 100 means the whole planet belongs to 2nd player
        # Values in between indicate, that there is an ongoing fight for the leadership
        self._planets_occupation_progress: np.ndarray = None

        # Occupation occurs, when a ship of one player enters the other one's planet
        # Value of 0 means, there's no ongoing occupation (there should be no progress to ownership)
        # Values < 0 mean, that the occupation progress is going in favor of the 1st player
        # Values > 0 mean, that the occupation is going in favor of the 2nd player
        # The higher the absolute value, the faster is the occupation progress
        self._planets_ongoing_occupation: np.ndarray = None

        self._player_1_visibility_mask: np.ndarray = None
        self._player_2_visibility_mask: np.ndarray = None
        self.ionized_field_id: dict = None

        self._player_1_score = 0
        self._player_2_score = 0

        # On start both players have 1 battleship at their base
        self._player_1_ships: dict = None
        self._player_2_ships: dict = None

        self._player_1_ships_next_id: int = None
        self._player_2_ships_next_id: int = None

        # 0 - right, 1 - down, 2 - left, 3 - up
        self._player_1_ships_facing: dict = None
        self._player_2_ships_facing: dict = None

        self._player_1_resources: np.ndarray = None
        self._player_2_resources: np.ndarray = None

        self._player_1_occupied_rf: np.ndarray = None
        self._player_2_occupied_rf: np.ndarray = None

        # self._player_1_origin = None
        # self._player_2_origin = None

        self.victorious_player = [False, False]
        self.terminated = False

        self.window: pygame.Surface = None
        self.clock: pygame.time.Clock = None

        """
        Death effect: (0, pos_x, pos_y, frame)
        Healing effect: (1, player, ship_id, frame)
        Firing effect: (2, ship_x, ship_y, facing, frame)
        Capture effect: (3, pos_x, pos_y, frame)
        Space jump effect: (4, pos_x, pos_y, frame)
        """
        self.effects = None

        self.turn: int = None
        self._round = 0

        if self._turn_on_music:
            setup_music_loop(volume=volume)

        pygame.display.set_caption(f"Octospace {VERSION}")

    def _get_info(self):
        return {}

    def _get_reward(self):
        # Draw may occur in 2 cases: when the match reaches max number of steps or both players are victorious
        # (It is possible, that both players capture each other's bases on the same turn)
        if self.turn == self.max_steps or all(self.victorious_player):
            return {
                "player_1": 0.5,
                "player_2": 0.5
            }

        # Under usual circumstances one of the players is the winner or none of them yet
        if self.player_1_id == self.player_1_id_original:
            return {
                "player_1": 1 if self.victorious_player[0] else 0,
                "player_2": 1 if self.victorious_player[1] else 0
            }
        else:
            return {
                "player_1": 1 if self.victorious_player[1] else 0,
                "player_2": 1 if self.victorious_player[0] else 0
            }

    def _get_obs(self):
        player_1_map = self._map.copy()
        player_2_map = self._map.copy()
        player_1_map[~(self._player_1_visibility_mask.astype(bool))] = -1
        player_2_map[~(self._player_2_visibility_mask.astype(bool))] = -1

        return {
            "player_1": {
                "map": player_1_map,
                "allied_ships": [[ship_id] + ship for ship_id, ship in self._player_1_ships.items()],
                "enemy_ships": [[ship_id] + ship for ship_id, ship in self._player_2_ships.items() if self._player_1_visibility_mask[ship[0], ship[1]]],
                "planets_occupation": [(planet_x, planet_y, occupation) for (planet_x, planet_y), occupation in
                                       zip(self._planets_centers, self._planets_occupation_progress) if
                                       self._player_1_visibility_mask[planet_y, planet_x]],
                "resources": self._player_1_resources
            },
            "player_2": {
                "map": player_2_map,
                "allied_ships": [[ship_id] + ship for ship_id, ship in self._player_2_ships.items()],
                "enemy_ships": [[ship_id] + ship for ship_id, ship in self._player_1_ships.items() if self._player_2_visibility_mask[ship[0], ship[1]]],
                "planets_occupation": [(planet_x, planet_y, occupation) for (planet_x, planet_y), occupation in
                                       zip(self._planets_centers, self._planets_occupation_progress) if
                                       self._player_2_visibility_mask[planet_y, planet_x]],
                "resources": self._player_2_resources
            }
        }

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> Tuple[dict, dict]:
        # On start both players have 1 battleship at their base
        self._player_1_ships = {0: [PLAYER_1_ORIGIN[0] + 7, PLAYER_1_ORIGIN[1], 100, 0, 0]}
        self._player_2_ships = {0: [PLAYER_2_ORIGIN[0] - 8, PLAYER_2_ORIGIN[1], 100, 0, 0]}

        self._player_1_ships_next_id = 1
        self._player_2_ships_next_id = 1

        # 0 - right, 1 - down, 2 - left, 3 - up
        self._player_1_ships_facing = {0: 1}
        self._player_2_ships_facing = {0: 3}

        self._player_1_visibility_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self._player_2_visibility_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)

        self._player_1_resources = np.array([100, 100, 100, 100], dtype=int)
        self._player_2_resources = np.array([100, 100, 100, 100], dtype=int)

        self._player_1_occupied_rf = np.array([4, 4, 4, 4], dtype=int)
        self._player_2_occupied_rf = np.array([4, 4, 4, 4], dtype=int)

        self.victorious_player = [False, False]
        self.terminated = False

        self.effects = []

        self.turn = 1

        # If it is the 2nd round, then don't generate a new map
        if self._round % 2 == 0:
            self._generate_map()

        # Handle planets occupation
        _reset_planets_occupation(game_map=self._map)
        _add_base_planet_occupation(game_map=self._map, centers=self._planets_centers)

        if self._round != 0:
            self._change_sides()

        generate_players_assets(player_1_id=self.player_1_id, player_2_id=self.player_2_id)

        self._reset_planets_occupation_state()

        self._round += 1

        _add_planet_visibility(self._planets_centers[0][1], self._planets_centers[0][0], self._player_1_visibility_mask)
        _add_planet_visibility(self._planets_centers[1][1], self._planets_centers[1][0], self._player_2_visibility_mask)

        return self._get_obs(), self._get_info()

    def _generate_map(self):
        self._map, new_planet_centers, ionized_field_id = _generate_map()
        self._state_ids = _generate_state_map(game_map=self._map)
        self._planets_centers = [PLAYER_1_ORIGIN, PLAYER_2_ORIGIN]
        self._planets_centers.extend(new_planet_centers)
        self._planets_centers = np.array(self._planets_centers, dtype=int)
        self.ionized_field_id = ionized_field_id

    def _reset_planets_occupation_state(self):
        self._planets_occupation_progress = [-1 for _ in range(len(self._planets_centers))]
        self._planets_occupation_progress[0] = 0
        self._planets_occupation_progress[1] = 100
        self._planets_ongoing_occupation = [0 for _ in range(len(self._planets_centers))]

    def step(
        self, actions: dict
    ) -> Tuple[dict, dict, bool, bool, dict]:
        self.turn += 1
        # If the song has ended, play another one
        if self._turn_on_music and not pygame.mixer.music.get_busy():
            get_new_track()

        # Decrease cooldowns
        _decrease_cooldowns(player_1_ships=self._player_1_ships, player_2_ships=self._player_2_ships)

        # Ships firing
        _ship_firing(actions=actions, player_1_ships=self._player_1_ships, player_2_ships=self._player_2_ships,
                     player_1_ships_facing=self._player_1_ships_facing, player_2_ships_facing=self._player_2_ships_facing,
                     effects=self.effects, turn_on_music=self._turn_on_music, volume=self.volume)

        # Ship movement
        _ship_movement(game_map=self._map, actions=actions, player_1_ships=self._player_1_ships,
                       player_2_ships=self._player_2_ships, player_1_ships_facing=self._player_1_ships_facing,
                       player_2_ships_facing=self._player_2_ships_facing, effects=self.effects, turn_on_music=self._turn_on_music,
                       volume=self.volume)

        # Construction
        _ship_construction(actions=actions, player_1_ships=self._player_1_ships, player_2_ships=self._player_2_ships,
                           player_1_ships_facing=self._player_1_ships_facing, player_2_ships_facing=self._player_2_ships_facing,
                           player_1_resources=self._player_1_resources, player_2_resources=self._player_2_resources)

        # Change the ownership of newly captured planets
        _change_ownership_of_planets(game_map=self._map, planets_centers=self._planets_centers,
                                     planets_occupation_progress=self._planets_occupation_progress, player_1_occupied_rf=self._player_1_occupied_rf,
                                     player_2_occupied_rf=self._player_2_occupied_rf, player_1_visibility_mask=self._player_1_visibility_mask,
                                     player_2_visibility_mask=self._player_2_visibility_mask, effects=self.effects,
                                     turn_on_music=self._turn_on_music, volume=self.volume)

        # Resource production
        self._player_1_resources = np.clip(self._player_1_resources + self._player_1_occupied_rf // RESOURCE_PRODUCTION_DIVISOR, 0, MAX_RESOURCES)
        self._player_2_resources = np.clip(self._player_2_resources + self._player_2_occupied_rf // RESOURCE_PRODUCTION_DIVISOR, 0, MAX_RESOURCES)

        # Occupation progress
        _occupation_progress(planets_centers=self._planets_centers, planets_occupation_progress=self._planets_occupation_progress,
                             planets_ongoing_occupation=self._planets_ongoing_occupation)

        # Planet capture and ship healing
        _ship_land_interaction(game_map=self._map, planets_centers=self._planets_centers, planets_occupation_progress=self._planets_occupation_progress,
                               planets_ongoing_occupation=self._planets_ongoing_occupation,
                               player_1_ships=self._player_1_ships, player_2_ships=self._player_2_ships,
                               player_1_ships_facing=self._player_1_ships_facing, player_2_ships_facing=self._player_2_ships_facing,
                               effects=self.effects)

        _handle_ship_death(player_1_ships=self._player_1_ships, player_2_ships=self._player_2_ships, player_1_ships_facing=self._player_1_ships_facing,
                           player_2_ships_facing=self._player_2_ships_facing, effects=self.effects, turn_on_music=self._turn_on_music, volume=self.volume)

        _handle_visibility(player_1_ships=self._player_1_ships, player_2_ships=self._player_2_ships, player_1_visibility_mask=self._player_1_visibility_mask,
                           player_2_visibility_mask=self._player_2_visibility_mask)

        self._victory_conditions()

        return self._get_obs(), self._get_reward(), self.terminated, False, self._get_info()

    def render(self) -> RenderFrame:
        if self.render_mode == "rgb_array":
            return self._render_frame()

        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE + 2*GUI_SIZE + 2*BORDER_WIDTH, WINDOW_SIZE))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        _render_background(canvas)

        # Render planets
        _render_planets(canvas, game_map=self._map, state_ids_map=self._state_ids, ionized_field_id=self.ionized_field_id)

        # Render planets occupation
        _render_planet_occupation(canvas, game_map=self._map, planets_centers=self._planets_centers, player_1_id=self.player_1_id, player_2_id=self.player_2_id)

        # Render ongoing planet capture
        _render_ongoing_planet_capture(canvas, planets_occupation=self._planets_occupation_progress,
                                       planets_centers=self._planets_centers, player_1_id=self.player_1_id, player_2_id=self.player_2_id)

        # Render players
        _render_players(canvas, player_1_id=self.player_1_id, player_2_id=self.player_2_id)

        # Render ships
        _render_ships(canvas, player_1_ships=self._player_1_ships, player_2_ships=self._player_2_ships,
                      player_1_ships_facing=self._player_1_ships_facing, player_2_ships_facing=self._player_2_ships_facing)

        # Display which turn currently is it
        _render_turn(canvas, turn=self.turn)

        # Render effects
        _render_effects(canvas, game_map=self._map, effects=self.effects, player_1_ships=self._player_1_ships, player_2_ships=self._player_2_ships,
                        player_1_ships_facing=self._player_1_ships_facing, player_2_ships_facing=self._player_2_ships_facing)

        # Vision debug
        if self.debug:
            _render_vision_debug(canvas, player_1_visibility_mask=self._player_1_visibility_mask,
                                 player_2_visibility_mask=self._player_2_visibility_mask, player_1_id=self.player_1_id,
                                 player_2_id=self.player_2_id)

        if self.render_mode == "human":
            self.window.blit(canvas, (GUI_SIZE + BORDER_WIDTH, 0))
            self.window.blit(BORDER, (GUI_SIZE, 0))
            self.window.blit(BORDER_SCORE, (GUI_SIZE+25, 0))

            if self.player_1_id == self._players_rendering_order[0]:
                _render_resources(self.window, player_1_resources=self._player_1_resources, player_2_resources=self._player_2_resources)
            else:
                _render_resources(self.window, player_1_resources=self._player_2_resources,
                                 player_2_resources=self._player_1_resources)
            _render_team_names(self.window, player_ids=[self.player_1_id, self.player_2_id])
            _render_score(self.window, player_1_score=self._player_1_score, player_2_score=self._player_2_score)

            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _victory_conditions(self):
        self.victorious_player = _check_victory_conditions(game_map=self._map, planets_centers=self._planets_centers)
        if self.turn == self.max_steps:
            self.terminated = True
            self._player_1_score += 0.5
            self._player_2_score += 0.5
        if any(self.victorious_player):
            if self.player_1_id == self.player_1_id_original:
                self._player_1_score += int(self.victorious_player[0]) / sum(self.victorious_player)
                self._player_2_score += int(self.victorious_player[1]) / sum(self.victorious_player)
            else:
                self._player_1_score += int(self.victorious_player[1]) / sum(self.victorious_player)
                self._player_2_score += int(self.victorious_player[0]) / sum(self.victorious_player)

    def _change_sides(self):
        self.player_1_id, self.player_2_id = self.player_2_id, self.player_1_id

    def close(self):
        if self.window is not None:
            if self._turn_on_music:
                pygame.mixer.music.stop()
            pygame.display.quit()
            pygame.font.quit()
            pygame.quit()
