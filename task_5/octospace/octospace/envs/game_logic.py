import numpy as np

from octospace.envs.game_config import (MAX_SHIP_FIRE_RANGE, SHIP_DAMAGE, BASE_SHIP_SPEED,
                         IONIZED_FIELD_SPEED_FACTOR, BOARD_SIZE, MOVEMENT_DIRECTIONS, SHIP_COST, PLAYER_1_ORIGIN, \
    PLAYER_2_ORIGIN, OCCUPATION_SPEED, SHIP_HEALING_SPEED, SHIP_OCCUPATION_RANGE, FIRING_COOLDOWN, MOVE_COOLDOWN,
                         ASTEROID_DAMAGE, VISION_RANGE, VISION_ADD_MASK)
from octospace.envs.schemes import PLANET_MASK
from octospace.envs.sound import play_space_jump_sound, play_capture_sound, play_ship_explosion_sound, play_shoot_sound


player_1_ships_next_id = 1
player_2_ships_next_id = 1


def _ship_firing(
    actions: dict,
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_ships_facing: dict,
    player_2_ships_facing: dict,
    effects: list,
    turn_on_music: bool,
    volume: float
):
    for command in actions["player_1"]["ships_actions"]:
        if command[1] == 1:
            ship_id, act, direction = command

            # If there is not such ship or the ship has an active firing cooldown
            if ship_id not in player_1_ships.keys() or player_1_ships[ship_id][4] > 0:
                continue

            # Play shoot sound
            if turn_on_music:
                play_shoot_sound(volume=volume)

            target_id = _get_target(
                ship_x=player_1_ships[ship_id][0],
                ship_y=player_1_ships[ship_id][1],
                direction=direction,
                player=0,
                player_1_ships=player_1_ships,
                player_2_ships=player_2_ships
            )

            effects.append([2, player_1_ships[ship_id][0], player_1_ships[ship_id][1], player_1_ships_facing[ship_id], 0])
            player_1_ships[ship_id][3] = FIRING_COOLDOWN    # Set firing cooldown for this ship

            if target_id == -1:
                continue
            player_2_ships[target_id][2] -= SHIP_DAMAGE  # Damage the enemy ship
            player_1_ships_facing[ship_id] = direction

    for command in actions["player_2"]["ships_actions"]:
        if command[1] == 1:
            ship_id, act, direction = command

            # If there is not such ship or the ship has an active firing cooldown
            if ship_id not in player_2_ships.keys() or player_2_ships[ship_id][4] > 0:
                continue

            # Play shoot sound
            if turn_on_music:
                play_shoot_sound(volume=volume)

            target_id = _get_target(
                ship_x=player_2_ships[ship_id][0],
                ship_y=player_2_ships[ship_id][1],
                direction=direction,
                player=1,
                player_1_ships=player_1_ships,
                player_2_ships=player_2_ships
            )

            effects.append([2, player_2_ships[ship_id][0], player_2_ships[ship_id][1], player_2_ships_facing[ship_id], 0])
            player_2_ships[ship_id][3] = FIRING_COOLDOWN

            if target_id == -1:
                continue
            player_1_ships[target_id][2] -= SHIP_DAMAGE
            player_2_ships_facing[ship_id] = direction


def _handle_ship_death(
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_ships_facing: dict,
    player_2_ships_facing: dict,
    effects: list,
    turn_on_music: bool,
    volume: float
):
    for ship_id in list(player_1_ships.keys()):
        # If the damaged ship's health points are below 0, then remove it from the board
        if player_1_ships[ship_id][2] <= 0:
            _delete_ship(player_1_ships=player_1_ships, player_2_ships=player_2_ships, player_1_ships_facing=player_1_ships_facing,
                         player_2_ships_facing=player_2_ships_facing, player=0, ship_id=ship_id, turn_on_music=turn_on_music,
                         volume=volume, effects=effects)

    for ship_id in list(player_2_ships.keys()):
        # If the damaged ship's health points are below 0, then remove it from the board
        if player_2_ships[ship_id][2] <= 0:
            _delete_ship(player_1_ships=player_1_ships, player_2_ships=player_2_ships,
                         player_1_ships_facing=player_1_ships_facing,
                         player_2_ships_facing=player_2_ships_facing, player=1, ship_id=ship_id,
                         turn_on_music=turn_on_music,
                         volume=volume, effects=effects)


def _ship_movement(
    game_map: np.ndarray,
    actions: dict,
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_ships_facing: dict,
    player_2_ships_facing: dict,
    effects: list,
    turn_on_music: bool,
    volume: float
):
    for command in actions["player_1"]["ships_actions"]:
        if command[1] == 0:
            ship_id, act, direction, velocity = command
            if ship_id not in player_1_ships.keys() or player_1_ships[ship_id][4] > 0:
                continue
            ship_x, ship_y = player_1_ships[ship_id][0], player_1_ships[ship_id][1]

            # Calculate max distance the ship can travel
            max_movement = BASE_SHIP_SPEED
            if game_map[ship_y, ship_x] == 4:
                max_movement = int(max_movement * IONIZED_FIELD_SPEED_FACTOR)
                if velocity == max_movement:
                    effects.append([4, ship_x, ship_y, 0])
                    if turn_on_music:
                        play_space_jump_sound(volume=volume)

            # If it's too far, then clip it to the maximum speed for the ship
            velocity = np.clip(velocity, 0, max_movement)
            movement_vec = MOVEMENT_DIRECTIONS[direction] * velocity

            # Move the ship in that direction
            player_1_ships[ship_id][0] = np.clip(player_1_ships[ship_id][0] + movement_vec[0], 0, BOARD_SIZE - 1)
            player_1_ships[ship_id][1] = np.clip(player_1_ships[ship_id][1] + movement_vec[1], 0, BOARD_SIZE - 1)

            # Update ship's direction
            player_1_ships_facing[ship_id] = direction

            # If the ship stumbled upon asteroid field, add a move cooldown
            if game_map[player_1_ships[ship_id][1], player_1_ships[ship_id][0]] == 2:
                player_1_ships[ship_id][4] = MOVE_COOLDOWN
                player_1_ships[ship_id][2] -= ASTEROID_DAMAGE

            # If the ship entered one of player's tiles, start the healing effect
            if game_map[player_1_ships[ship_id][1], player_1_ships[ship_id][0]] & 64 == 64 and game_map[ship_y, ship_x] & 64 != 64:
                effects.append([1, 0, ship_id, 0])

            # If the ship left player's tile, stop the healing effect
            if game_map[player_1_ships[ship_id][1], player_1_ships[ship_id][0]] & 64 != 64 and game_map[ship_y, ship_x] & 64 == 64:
                _delete_healing_effect(0, ship_id, effects)

    for command in actions["player_2"]["ships_actions"]:
        if command[1] == 0:
            ship_id, act, direction, velocity = command
            if ship_id not in player_2_ships.keys() or player_2_ships[ship_id][4] > 0:
                continue
            ship_x, ship_y = player_2_ships[ship_id][0], player_2_ships[ship_id][1]

            # Calculate max distance the ship can travel
            max_movement = BASE_SHIP_SPEED
            if game_map[ship_y, ship_x] == 4:
                max_movement = int(max_movement * IONIZED_FIELD_SPEED_FACTOR)
                if velocity == max_movement:
                    effects.append([4, ship_x, ship_y, 0])
                    if turn_on_music:
                        play_space_jump_sound(volume=volume)

            # If it's too far, then don't do anything
            velocity = np.clip(velocity, 0, max_movement)
            movement_vec = MOVEMENT_DIRECTIONS[direction] * velocity

            # Move the ship in that direction
            player_2_ships[ship_id][0] = np.clip(player_2_ships[ship_id][0] + movement_vec[0], 0, BOARD_SIZE - 1)
            player_2_ships[ship_id][1] = np.clip(player_2_ships[ship_id][1] + movement_vec[1], 0, BOARD_SIZE - 1)

            # Update ship's direction
            player_2_ships_facing[ship_id] = direction

            # If the ship stumbled upon asteroid field, add a move cooldown
            if game_map[player_2_ships[ship_id][1], player_2_ships[ship_id][0]] == 2:
                player_2_ships[ship_id][4] = MOVE_COOLDOWN
                player_2_ships[ship_id][2] -= ASTEROID_DAMAGE

            # If the ship entered one of player's tiles, start the healing effect
            if game_map[player_2_ships[ship_id][1], player_2_ships[ship_id][0]] & 128 == 128 and game_map[ship_y, ship_x] & 128 != 128:
                effects.append([1, 1, ship_id, 0])

            # If the ship left player's tile, stop the healing effect
            if game_map[player_2_ships[ship_id][1], player_2_ships[ship_id][0]] & 128 != 128 and game_map[ship_y, ship_x] & 128 == 128:
                _delete_healing_effect(1, ship_id, effects)


def _ship_construction(
    actions: dict,
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_ships_facing: dict,
    player_2_ships_facing: dict,
    player_1_resources: np.ndarray,
    player_2_resources: np.ndarray,
):
    if actions["player_1"]["construction"] > 0:
        for i in range(actions["player_1"]["construction"]):
            if np.all([player_1_resources[i] >= SHIP_COST[i] for i in range(4)]):
                new_ship_id = _get_player_next_id(0)
                player_1_ships[new_ship_id] = [PLAYER_1_ORIGIN[0], PLAYER_1_ORIGIN[1], 100, 0, 0]
                player_1_ships_facing[new_ship_id] = 0
                player_1_resources -= SHIP_COST

    if actions["player_2"]["construction"] > 0:
        for i in range(actions["player_2"]["construction"]):
            if np.all([player_2_resources[i] >= SHIP_COST[i] for i in range(4)]):
                new_ship_id = _get_player_next_id(1)
                player_2_ships[new_ship_id] = [PLAYER_2_ORIGIN[0], PLAYER_2_ORIGIN[1], 100, 0, 0]
                player_2_ships_facing[new_ship_id] = 0
                player_2_resources -= SHIP_COST


def _occupation_progress(
    planets_centers: np.ndarray,
    planets_occupation_progress: np.ndarray,
    planets_ongoing_occupation: np.ndarray
):
    for e, center in enumerate(planets_centers):
        # If there is an ongoing occupation of the planet
        if planets_ongoing_occupation[e] != 0:
            planets_occupation_progress[e] = np.clip(planets_occupation_progress[e] + planets_ongoing_occupation[e] * OCCUPATION_SPEED, 0, 100)

            # If the planet got occupied, reset the occupation speed counter
            if planets_occupation_progress[e] in [0, 100]:
                planets_ongoing_occupation[e] = 0


def _change_ownership_of_planets(
    game_map: np.ndarray,
    planets_centers: np.ndarray,
    planets_occupation_progress: np.ndarray,
    player_1_occupied_rf: np.ndarray,
    player_2_occupied_rf: np.ndarray,
    player_1_visibility_mask: np.ndarray,
    player_2_visibility_mask: np.ndarray,
    effects: list,
    turn_on_music: bool,
    volume: float
):
    for e, center in enumerate(planets_centers):
        if planets_occupation_progress[e] == 0 and game_map[center[0], center[1]] & 64 != 64:
            map_mask = np.zeros((BOARD_SIZE, BOARD_SIZE))
            map_mask[center[0] - 4: center[0] + 5, center[1] - 4: center[1] + 5] = PLANET_MASK

            # If the planet was already occupied by the other player, delete his ownership
            if game_map[center[0], center[1]] & 128 == 128:
                game_map[map_mask.astype(bool)] -= 128

            val, rf_cnt = np.unique(game_map[center[0] - 4: center[0] + 5, center[1] - 4: center[1] + 5],
                                    return_counts=True)
            rf_counts = dict(zip(val, rf_cnt))
            try:
                rf_counts = np.array([rf_counts[9], rf_counts[17], rf_counts[25], rf_counts[57]])
            except KeyError:
                print(val, rf_cnt)
                exit()

            if game_map[center[1], center[0]] & 128 == 128:
                player_2_occupied_rf -= rf_counts

            player_1_occupied_rf += rf_counts

            # Add planet ownership to player_1
            game_map[map_mask.astype(bool)] |= 64

            # Add capture effect
            effects.append([3, center[1], center[0], 0])
            if turn_on_music:
                play_capture_sound(volume=volume)

            # Add area around the planet to the player's visibility mask
            _add_planet_visibility(center[1], center[0], player_1_visibility_mask)

        elif planets_occupation_progress[e] == 100 and game_map[center[0], center[1]] & 128 != 128:
            map_mask = np.zeros((BOARD_SIZE, BOARD_SIZE))
            map_mask[center[0] - 4: center[0] + 5, center[1] - 4: center[1] + 5] = PLANET_MASK

            # If the planet was already occupied by the other player, delete his ownership
            if game_map[center[0], center[1]] & 64 == 64:
                game_map[map_mask.astype(bool)] -= 64

            val, rf_cnt = np.unique(game_map[center[0] - 4: center[0] + 5, center[1] - 4: center[1] + 5],
                                    return_counts=True)
            rf_counts = dict(zip(val, rf_cnt))
            rf_counts = np.array([rf_counts[9], rf_counts[17], rf_counts[25], rf_counts[57]])

            if game_map[center[1], center[0]] & 64 == 64:
                player_1_occupied_rf -= rf_counts

            player_2_occupied_rf += rf_counts

            # Add planet ownership to player_2
            game_map[map_mask.astype(bool)] |= 128

            # Add capture effect
            effects.append([3, center[1], center[0], 0])
            if turn_on_music:
                play_capture_sound(volume=volume)

            _add_planet_visibility(center[1], center[0], player_2_visibility_mask)


def _ship_land_interaction(
    game_map: np.ndarray,
    planets_centers: np.ndarray,
    planets_occupation_progress: np.ndarray,
    planets_ongoing_occupation: np.ndarray,
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_ships_facing: dict,
    player_2_ships_facing: dict,
    effects: list
):
    ship_ids_to_delete_1 = []
    for ship_id, (ship_x, ship_y, hp, firing_cooldown, move_cooldown) in player_1_ships.items():
        if game_map[ship_y, ship_x] & 64 == 64 and hp != 100:
            player_1_ships[ship_id][2] = np.clip(player_1_ships[ship_id][2] + SHIP_HEALING_SPEED, 1, 100)

        planet_id = _get_planet_id_by_ship_position(ship_x, ship_y, planets_centers=planets_centers)
        if planet_id != -1:
            # If there is an ongoing fight for this planet
            if planets_ongoing_occupation[planet_id] != 0 or planets_occupation_progress[planet_id] not in [-1, 0, 100]:
                planets_ongoing_occupation[planet_id] -= 1
                ship_ids_to_delete_1.append(ship_id)

            # If planet is unoccupied
            elif planets_occupation_progress[planet_id] == -1:
                planets_occupation_progress[planet_id] = 0
                ship_ids_to_delete_1.append(ship_id)

            # If the planet belongs to the other player
            elif planets_occupation_progress[planet_id] == 100:
                planets_occupation_progress[planet_id] = 100 - OCCUPATION_SPEED
                planets_ongoing_occupation[planet_id] -= 1
                ship_ids_to_delete_1.append(ship_id)

    # Delete the ship afterward
    for ship_id in ship_ids_to_delete_1:
        _delete_ship(player_1_ships=player_1_ships, player_2_ships=player_2_ships,
                     player_1_ships_facing=player_1_ships_facing,
                     player_2_ships_facing=player_2_ships_facing, player=0, ship_id=ship_id,
                     turn_on_music=False,
                     volume=0.0, effects=effects, death_effect=False)

    ship_ids_to_delete_2 = []
    for ship_id, (ship_x, ship_y, hp, firing_cooldown, move_cooldown) in player_2_ships.items():
        if game_map[ship_y, ship_x] & 128 == 128 and hp != 100:
            player_2_ships[ship_id][2] = np.clip(player_2_ships[ship_id][2] + SHIP_HEALING_SPEED, 1, 100)

        planet_id = _get_planet_id_by_ship_position(ship_x, ship_y, planets_centers=planets_centers)
        if planet_id != -1:
            # If there is an ongoing fight for this planet
            if planets_ongoing_occupation[planet_id] != 0 or planets_occupation_progress[planet_id] not in [-1, 0, 100]:
                planets_ongoing_occupation[planet_id] += 1
                ship_ids_to_delete_2.append(ship_id)

            # If planet is unoccupied
            elif planets_occupation_progress[planet_id] == -1:
                planets_occupation_progress[planet_id] = 100
                ship_ids_to_delete_2.append(ship_id)

            # If the planet belongs to the other player
            elif planets_occupation_progress[planet_id] == 0:
                planets_occupation_progress[planet_id] = OCCUPATION_SPEED
                planets_ongoing_occupation[planet_id] += 1
                ship_ids_to_delete_2.append(ship_id)

    # Delete the ship afterward
    for ship_id in ship_ids_to_delete_2:
        _delete_ship(player_1_ships=player_1_ships, player_2_ships=player_2_ships,
                     player_1_ships_facing=player_1_ships_facing,
                     player_2_ships_facing=player_2_ships_facing, player=1, ship_id=ship_id,
                     turn_on_music=False,
                     volume=0.0, effects=effects, death_effect=False)


def _decrease_cooldowns(
    player_1_ships: dict,
    player_2_ships: dict,
):
    for key in player_1_ships.keys():
        player_1_ships[key][3] = max(player_1_ships[key][3] - 1, 0)
        player_1_ships[key][4] = max(player_1_ships[key][4] - 1, 0)

    for key in player_2_ships.keys():
        player_2_ships[key][3] = max(player_2_ships[key][3] - 1, 0)
        player_2_ships[key][4] = max(player_2_ships[key][4] - 1, 0)


def _handle_visibility(
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_visibility_mask: np.ndarray,
    player_2_visibility_mask: np.ndarray
):
    for ship_id in player_1_ships.keys():
        ship_x, ship_y = player_1_ships[ship_id][0], player_1_ships[ship_id][1]
        start_x = max(ship_x - VISION_RANGE, 0)
        end_x = min(ship_x + VISION_RANGE + 1, BOARD_SIZE)
        start_y = max(ship_y - VISION_RANGE, 0)
        end_y = min(ship_y + VISION_RANGE + 1, BOARD_SIZE)

        vision_add_start_x = start_x - ship_x + VISION_RANGE
        vision_add_end_x = end_x - ship_x + VISION_RANGE
        vision_add_start_y = start_y - ship_y + VISION_RANGE
        vision_add_end_y = end_y - ship_y + VISION_RANGE

        player_1_visibility_mask[start_y:end_y, start_x:end_x] = player_1_visibility_mask[start_y:end_y, start_x:end_x] | VISION_ADD_MASK[vision_add_start_y:vision_add_end_y, vision_add_start_x:vision_add_end_x]

    for ship_id in player_2_ships.keys():
        ship_x, ship_y = player_2_ships[ship_id][0], player_2_ships[ship_id][1]
        start_x = max(ship_x - VISION_RANGE, 0)
        end_x = min(ship_x + VISION_RANGE + 1, BOARD_SIZE)
        start_y = max(ship_y - VISION_RANGE, 0)
        end_y = min(ship_y + VISION_RANGE + 1, BOARD_SIZE)

        vision_add_start_x = start_x - ship_x + VISION_RANGE
        vision_add_end_x = end_x - ship_x + VISION_RANGE
        vision_add_start_y = start_y - ship_y + VISION_RANGE
        vision_add_end_y = end_y - ship_y + VISION_RANGE

        player_2_visibility_mask[start_y:end_y, start_x:end_x] = player_2_visibility_mask[start_y:end_y, start_x:end_x] | VISION_ADD_MASK[vision_add_start_y:vision_add_end_y,
                                                                                                                          vision_add_start_x:vision_add_end_x]


def _check_victory_conditions(
    game_map: np.ndarray,
    planets_centers: np.ndarray
):
    player_1_center = planets_centers[0]
    player_2_center = planets_centers[1]

    player_1_victory = game_map[player_2_center[0], player_2_center[1]] & 128 != 128
    player_2_victory = game_map[player_1_center[0], player_1_center[1]] & 64 != 64
    return player_1_victory, player_2_victory


def _add_planet_visibility(
    planet_x: int,
    planet_y: int,
    visibility_mask: np.ndarray
):
    start_x = max(planet_x - VISION_RANGE, 0)
    end_x = min(planet_x + VISION_RANGE + 1, BOARD_SIZE)
    start_y = max(planet_y - VISION_RANGE, 0)
    end_y = min(planet_y + VISION_RANGE + 1, BOARD_SIZE)

    vision_add_start_x = start_x - planet_x + VISION_RANGE
    vision_add_end_x = end_x - planet_x + VISION_RANGE
    vision_add_start_y = start_y - planet_y + VISION_RANGE
    vision_add_end_y = end_y - planet_y + VISION_RANGE

    visibility_mask[start_x:end_x, start_y:end_y] = visibility_mask[start_x:end_x, start_y:end_y] | VISION_ADD_MASK[vision_add_start_x:vision_add_end_x,
                                                                  vision_add_start_y:vision_add_end_y]


def _get_target(
    ship_x: int,
    ship_y: int,
    direction: int,
    player: int,
    player_1_ships: dict,
    player_2_ships: dict
):
    """
    Returns id of the first ship, that player's ship is facing, if it is in firing range
    """
    if player == 0:
        enemy_ships = player_2_ships
    else:
        enemy_ships = player_1_ships

    if len(enemy_ships.keys()) == 0:
        return -1

    ship_vec = np.array([ship_x, ship_y], dtype=int)
    target_vec = np.array([ship_x, ship_y], dtype=int) + MOVEMENT_DIRECTIONS[direction] * MAX_SHIP_FIRE_RANGE
    target_vec -= ship_vec
    vec_to_other_ships = np.array([(x, y) for ship_id, (x, y, hp, firing_cooldown, move_cooldown) in enemy_ships.items()], dtype=int)
    vec_to_other_ships = vec_to_other_ships - ship_vec
    vec_angles = [np.arccos(np.clip(np.dot(vec/np.linalg.norm(vec), target_vec/np.linalg.norm(target_vec)), -1.0, 1.0)) if np.linalg.norm(vec) != 0 else 0 for vec in vec_to_other_ships]

    target_id = -1
    min_dist = MAX_SHIP_FIRE_RANGE + 1
    for i in range(len(enemy_ships.keys())):

        # Get all ships between -15 and 15 degrees
        if -np.pi/12 <= vec_angles[i] <= np.pi/12 and min_dist > np.linalg.norm(vec_to_other_ships[i]):
            target_id = list(enemy_ships.keys())[i]
            min_dist = np.linalg.norm(vec_to_other_ships[i])
    return target_id


def _get_player_next_id(player: int):
    global player_1_ships_next_id, player_2_ships_next_id
    if player == 0:
        player_1_ships_next_id += 1
        return player_1_ships_next_id - 1
    else:
        player_2_ships_next_id += 1
        return player_2_ships_next_id - 1


def _get_planet_id_by_ship_position(ship_x, ship_y, planets_centers):
    for e, center in enumerate(planets_centers):
        if np.linalg.norm([ship_x - center[1], ship_y - center[0]]) <= SHIP_OCCUPATION_RANGE:
            return e
    return -1


def _delete_healing_effect(
        player: int,
        ship_id: int,
        effects: list
):
    i = 0
    while i < len(effects):
        if effects[i][0] == 1 and effects[i][1] == player and effects[i][2] == ship_id:
            effects.pop(i)
        i += 1


def _delete_ship(
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_ships_facing: dict,
    player_2_ships_facing: dict,
    player: int,
    ship_id: int,
    turn_on_music: bool,
    volume: float,
    effects: list,
    death_effect: bool = True
):
    if player == 0:
        ship_list = player_1_ships
        ship_list_facing = player_1_ships_facing
    else:
        ship_list = player_2_ships
        ship_list_facing = player_2_ships_facing

    if death_effect:
        effects.append([0, ship_list[ship_id][0], ship_list[ship_id][1], 0])

    _delete_healing_effect(player, ship_id, effects)

    del ship_list[ship_id]
    del ship_list_facing[ship_id]

    if turn_on_music:
        play_ship_explosion_sound(volume=volume)

