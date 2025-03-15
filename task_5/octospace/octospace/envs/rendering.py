import numpy as np
import pygame
from pygame import BLEND_MULT

from octospace.envs.game_config import (BOARD_SIZE, TILE_SIZE, RF_CODING_TO_ID, RF_MARKER_ADJUSTMENT, EFFECT_IONIZED_FIELD_SIZE,
                                        FLAG_ICON_ADJUSTMENT, OCCUPATION_BAR_SIZE, OCCUPATION_BAR_COLOR_MARGIN, ABS_PLAYER_1_ICON_POS, \
                                        ABS_PLAYER_2_ICON_POS, SHIP_SIZE, SIDE_SHIP_SIZE, WINDOW_SIZE, TEAM_NAMES_MARGIN, GUI_SIZE,
                                        BORDER_WIDTH, MAX_RESOURCES, EFFECT_DEATH_ADJUSTMENT, EFFECT_FIRING_ADJUSTMENT, EFFECT_HEALING_ADJUSTMENT,
                                        EFFECT_CAPTURE_ADJUSTMENT, EFFECT_SPACE_JUMP_ADJUSTMENT)
from octospace.envs.map_assets import (LAND, RESOURCE_FIELDS_MARKERS, ASTEROIDS, IONIZED_FIELDS, OCCUPATION_FLAG, \
    TEAM_COLORS, OCCUPATION_FLAG_CROSSED, BAR_EMPTY, TEAM_ICONS, SHIP_ORIENTATIONS_1, SHIP_ORIENTATIONS_2, BACKGROUND, \
    PLAYER_ICON, RESOURCE_FIELDS_ICONS, RESOURCE_FIELDS_BARS, DEATH_EFFECT_ANIMATION, HEALING_EFFECT_ANIMATION,
                        FIRING_EFFECT_ANIMATION, CAPTURE_EFFECT_ANIMATION, SPACE_JUMP_EFFECT_ANIMATION, ROUGH_TERRAIN,
                        ROUGH_TERRAIN_FLAG, ROUGH_TERRAIN_CORNER)

from matches_config import TEAMS_ABBREVIATIONS


pygame.font.init()
ship_font = pygame.font.SysFont("Arial", size=11)
turn_counter_font = pygame.font.SysFont("Arial", size=35)
teams_names_font = pygame.font.SysFont("Bricolage Grotesque", size=75)
resource_font = pygame.font.SysFont("Arial", size=12)
scoreboard_font = pygame.font.SysFont("Arial", size=40)


def _render_background(canvas: pygame.Surface):
    canvas.blit(pygame.transform.scale(BACKGROUND, (WINDOW_SIZE, WINDOW_SIZE)), BACKGROUND.get_rect())


def _render_planets(
        canvas: pygame.Surface,
        game_map: np.ndarray,
        state_ids_map: np.ndarray,
        ionized_field_id: dict):
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            block = game_map[y, x]
            if block & 3 == 1:
                canvas.blit(LAND[state_ids_map[y, x]], (x * TILE_SIZE, y * TILE_SIZE))
                # Render resource fields marks
                if block & 57 != 1:
                    rf_coding = block & 57
                    canvas.blit(RESOURCE_FIELDS_MARKERS[RF_CODING_TO_ID[rf_coding]],
                                (x * TILE_SIZE + RF_MARKER_ADJUSTMENT, y * TILE_SIZE + RF_MARKER_ADJUSTMENT))
            elif block & 3 == 2:
                canvas.blit(ASTEROIDS[state_ids_map[y, x]], (x * TILE_SIZE, y * TILE_SIZE))
            elif block & 3 == 3:
                if game_map[y-1, x] & 57 != 1:
                    if game_map[y, x-1] & 57 != 1:
                        canvas.blit(ROUGH_TERRAIN_CORNER[3], (x * TILE_SIZE, y * TILE_SIZE))
                    elif game_map[y, x+1] & 57 != 1:
                        canvas.blit(ROUGH_TERRAIN_CORNER[2], (x * TILE_SIZE, y * TILE_SIZE))
                    else:
                        canvas.blit(ROUGH_TERRAIN[1], (x * TILE_SIZE, y * TILE_SIZE))
                elif game_map[y+1, x] & 57 != 1:
                    if game_map[y, x-1] & 57 != 1:
                        canvas.blit(ROUGH_TERRAIN_CORNER[0], (x * TILE_SIZE, y * TILE_SIZE))
                    elif game_map[y, x+1] & 57 != 1:
                        canvas.blit(ROUGH_TERRAIN_CORNER[1], (x * TILE_SIZE, y * TILE_SIZE))
                    else:
                        canvas.blit(ROUGH_TERRAIN[3], (x * TILE_SIZE, y * TILE_SIZE))
                elif game_map[y, x-1] & 57 != 1:
                    canvas.blit(ROUGH_TERRAIN[2], (x * TILE_SIZE, y * TILE_SIZE))
                elif game_map[y, x+1] & 57 != 1:
                    canvas.blit(ROUGH_TERRAIN[0], (x * TILE_SIZE, y * TILE_SIZE))
                else:
                    canvas.blit(ROUGH_TERRAIN_FLAG, (x * TILE_SIZE, y * TILE_SIZE))

            # Render ionized fields
            elif block & 4 == 4:
                effect_loc_adjustment = TILE_SIZE // 2 - EFFECT_IONIZED_FIELD_SIZE // 2
                canvas.blit(IONIZED_FIELDS[ionized_field_id[(y, x)]],
                            (x * TILE_SIZE + effect_loc_adjustment, y * TILE_SIZE + effect_loc_adjustment))
                ionized_field_id[(y, x)] = (ionized_field_id[(y, x)] + 1) % len(IONIZED_FIELDS)

def _render_planet_occupation(
    canvas: pygame.Surface,
    game_map: np.ndarray,
    planets_centers: list,
    player_1_id: int,
    player_2_id: int
):
    for center in planets_centers:
        if game_map[center[0], center[1]] & 64 == 64:
            flag_image = OCCUPATION_FLAG.copy()
            flag_image.fill(TEAM_COLORS[player_1_id], special_flags=BLEND_MULT)
        elif game_map[center[0], center[1]] & 128 == 128:
            flag_image = OCCUPATION_FLAG.copy()
            flag_image.fill(TEAM_COLORS[player_2_id], special_flags=BLEND_MULT)
        else:
            flag_image = OCCUPATION_FLAG_CROSSED

        canvas.blit(flag_image,(center[1] * TILE_SIZE + FLAG_ICON_ADJUSTMENT, center[0] * TILE_SIZE + FLAG_ICON_ADJUSTMENT))

def _render_ongoing_planet_capture(
    canvas: pygame.Surface,
    planets_occupation: np.ndarray,
    planets_centers: np.ndarray,
    player_1_id: int,
    player_2_id: int
):
    for e, occupation in enumerate(planets_occupation):
        if occupation not in [-1, 0, 100]:
            planet_y, planet_x = planets_centers[e]

            color_surface_height = OCCUPATION_BAR_SIZE // 5 - 2 * OCCUPATION_BAR_COLOR_MARGIN
            player_2_color_surface_width = int(
                (OCCUPATION_BAR_SIZE - 2 * OCCUPATION_BAR_COLOR_MARGIN) * occupation / 100)
            player_1_color_surface_width = (OCCUPATION_BAR_SIZE - 2 * OCCUPATION_BAR_COLOR_MARGIN) - player_2_color_surface_width

            color_surface_1 = pygame.Surface((player_1_color_surface_width, color_surface_height))
            color_surface_1.fill(TEAM_COLORS[player_1_id])
            color_surface_2 = pygame.Surface((player_2_color_surface_width, color_surface_height))
            color_surface_2.fill(TEAM_COLORS[player_2_id])

            bar_filling = BAR_EMPTY.copy()
            bar_filling.blit(color_surface_1, (OCCUPATION_BAR_COLOR_MARGIN, OCCUPATION_BAR_COLOR_MARGIN),
                             special_flags=BLEND_MULT)
            bar_filling.blit(color_surface_2,
                             (OCCUPATION_BAR_COLOR_MARGIN + player_1_color_surface_width, OCCUPATION_BAR_COLOR_MARGIN),
                             special_flags=BLEND_MULT)

            canvas.blit(bar_filling, ((planet_x - 4) * TILE_SIZE + 15, (planet_y + 5) * TILE_SIZE + 5))

def _render_players(
    canvas: pygame.Surface,
    player_1_id: int,
    player_2_id: int
):
    canvas.blit(TEAM_ICONS[player_1_id], (ABS_PLAYER_1_ICON_POS, ABS_PLAYER_1_ICON_POS))
    canvas.blit(TEAM_ICONS[player_2_id], (ABS_PLAYER_2_ICON_POS, ABS_PLAYER_2_ICON_POS))


def _render_ships(
    canvas: pygame.Surface,
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_ships_facing: dict,
    player_2_ships_facing: dict,
):
    for key in player_1_ships.keys():
        ship = player_1_ships[key]
        x, y = ship[0], ship[1]
        ship_loc_adjustment = TILE_SIZE // 2 - (
            SHIP_SIZE // 2 if player_1_ships_facing[key] in [1, 3] else SIDE_SHIP_SIZE // 2)
        ship_x = x * TILE_SIZE + ship_loc_adjustment
        ship_y = y * TILE_SIZE + ship_loc_adjustment
        canvas.blit(SHIP_ORIENTATIONS_1[player_1_ships_facing[key]], (ship_x, ship_y))
        ship_text = ship_font.render(f"{ship[2]}%", False, _get_ship_text_color(ship))
        canvas.blit(ship_text, (ship_x, ship_y - 12))

    for key in player_2_ships.keys():
        ship = player_2_ships[key]
        x, y = ship[0], ship[1]
        ship_loc_adjustment = TILE_SIZE // 2 - (
            SHIP_SIZE // 2 if player_2_ships_facing[key] in [1, 3] else SIDE_SHIP_SIZE // 2)
        ship_x = x * TILE_SIZE + ship_loc_adjustment
        ship_y = y * TILE_SIZE + ship_loc_adjustment
        canvas.blit(SHIP_ORIENTATIONS_2[player_2_ships_facing[key]], (ship_x, ship_y))
        ship_text = ship_font.render(f"{ship[2]}%", False, _get_ship_text_color(ship))
        canvas.blit(ship_text, (ship_x, ship_y - 12))


def _render_turn(canvas, turn):
    turn_text = turn_counter_font.render(f'TURN: {turn}', False, (255, 255, 255))
    canvas.blit(turn_text, (WINDOW_SIZE - 200, 50))


# Team rendering is always performed in the same order
def _render_team_names(
    window: pygame.Surface,
    player_ids: list
):
    player_ids = sorted(player_ids)
    player_1_id, player_2_id = player_ids[0], player_ids[1]

    team_1_text_size = teams_names_font.size(TEAMS_ABBREVIATIONS[player_1_id])
    team_2_text_size = teams_names_font.size(TEAMS_ABBREVIATIONS[player_2_id])

    team_1_text = pygame.transform.rotate(teams_names_font.render(TEAMS_ABBREVIATIONS[player_1_id], False, (255, 255, 255)), angle=90)
    team_2_text = pygame.transform.rotate(teams_names_font.render(TEAMS_ABBREVIATIONS[player_2_id], False, (255, 255, 255)), angle=270)
    window.blit(team_1_text, (TEAM_NAMES_MARGIN, WINDOW_SIZE - TEAM_NAMES_MARGIN - team_1_text_size[0]))
    window.blit(team_2_text, (WINDOW_SIZE + 2 * GUI_SIZE + 2 * BORDER_WIDTH - 100, TEAM_NAMES_MARGIN))

    tentacle_1 = pygame.transform.rotate(PLAYER_ICON, angle=90)
    tentacle_1.fill(TEAM_COLORS[player_1_id], special_flags=BLEND_MULT)
    tentacle_2 = pygame.transform.rotate(PLAYER_ICON, angle=270)
    tentacle_2.fill(TEAM_COLORS[player_2_id], special_flags=BLEND_MULT)

    window.blit(tentacle_1, (TEAM_NAMES_MARGIN - 5, WINDOW_SIZE - team_1_text_size[0] - 3 * TEAM_NAMES_MARGIN))
    window.blit(tentacle_2, (WINDOW_SIZE + 2 * GUI_SIZE + 2 * BORDER_WIDTH - 100, 2 * TEAM_NAMES_MARGIN + team_2_text_size[0]))


def _render_resources(
    window: pygame.Surface,
    player_1_resources: np.ndarray,
    player_2_resources: np.ndarray
):
    for player in range(2):
        resource_surface = pygame.Surface((GUI_SIZE+20, WINDOW_SIZE))

        for rf_id in range(4):
            if player == 0:
                resource_amount = player_1_resources[rf_id]
            else:
                resource_amount = player_2_resources[rf_id]
            bar_state_id = np.clip(10 - int(resource_amount / MAX_RESOURCES * 10), 0, 9)
            resource_text = resource_font.render(f'{resource_amount}/{MAX_RESOURCES}', False, (255, 255, 255))
            resource_surface.blit(RESOURCE_FIELDS_ICONS[rf_id], (50, 50 + 100 * rf_id))
            resource_surface.blit(RESOURCE_FIELDS_BARS[rf_id][bar_state_id], (30, 100 + 100 * rf_id))
            resource_surface.blit(resource_text, (42, 120 + rf_id * 100))

        window.blit(resource_surface, (player * (WINDOW_SIZE + GUI_SIZE + 2*BORDER_WIDTH - 30), player * (WINDOW_SIZE - 500)))


def _render_score(
    window: pygame.Surface,
    player_1_score: float,
    player_2_score: float
):

    if player_1_score % 1 == 0 and player_2_score % 1 == 0:
        player_1_score = int(player_1_score)
        player_2_score = int(player_2_score)
    elif player_1_score % 1 != 0:
        player_2_score = int(player_2_score)
    else:
        player_1_score = int(player_1_score)

    left_dist = WINDOW_SIZE//2 + GUI_SIZE + BORDER_WIDTH - scoreboard_font.size(f"{player_1_score} : {player_2_score}")[0]//2
    upper_dist = int(WINDOW_SIZE * 0.01)

    score_text = scoreboard_font.render(f"{player_1_score} : {player_2_score}", False, (255, 255, 255))
    window.blit(score_text, (left_dist, upper_dist))



def _render_effects(
    canvas: pygame.Surface,
    game_map: np.ndarray,
    effects: list,
    player_1_ships: dict,
    player_2_ships: dict,
    player_1_ships_facing: dict,
    player_2_ships_facing: dict
):
    """
    Effects
    """
    # Delete all expired effects
    i = 0
    while i < len(effects):
        if effects[i][0] == 0 and effects[i][3] == 15:
            effects.pop(i)
        elif effects[i][0] == 1 and effects[i][3] == 15:
            # Reset frame counter
            effects[i][3] %= 15
        elif effects[i][0] == 2 and effects[i][4] == 5:
            effects.pop(i)
        elif effects[i][0] == 3 and effects[i][3] == 12:
            effects.pop(i)
        elif effects[i][0] == 4 and effects[i][3] == 9:
            effects.pop(i)
        else:
            i += 1

    for e, effect in enumerate(effects):
        effect_id = effect[0]

        # Death effect
        if effect_id == 0:
            pos_x, pos_y, frame = effect[1], effect[2], effect[3]
            canvas.blit(DEATH_EFFECT_ANIMATION[frame], (pos_x*TILE_SIZE+EFFECT_DEATH_ADJUSTMENT, pos_y*TILE_SIZE+EFFECT_DEATH_ADJUSTMENT))

            # Proceed to the next frame
            effects[e][3] += 1

        # Healing effect
        elif effect_id == 1:
            player, ship_id, frame = effect[1], effect[2], effect[3]
            if player == 0:
                ally_ships = player_1_ships
            else:
                ally_ships = player_2_ships

            if ship_id in ally_ships.keys():
                pos_x, pos_y = ally_ships[ship_id][0], ally_ships[ship_id][1]
                canvas.blit(HEALING_EFFECT_ANIMATION[frame], (pos_x*TILE_SIZE+EFFECT_HEALING_ADJUSTMENT, pos_y*TILE_SIZE+EFFECT_HEALING_ADJUSTMENT))

                # Next frame
                effects[e][3] += 1

        # Firing effect:
        elif effect_id == 2:
            ship_x = effect[1]
            ship_y = effect[2]
            facing = effect[3]
            frame = effect[4]

            if facing == 0:
                facing_adjustment = 15+SHIP_SIZE, -12
            elif facing == 1:
                facing_adjustment = -2, SHIP_SIZE + 10
            elif facing == 2:
                facing_adjustment = -20-SIDE_SHIP_SIZE, 0
            else:
                facing_adjustment = -12, -SHIP_SIZE - 27

            canvas.blit(FIRING_EFFECT_ANIMATION[facing][frame],
                        (ship_x*TILE_SIZE+EFFECT_FIRING_ADJUSTMENT + facing_adjustment[0],
                         ship_y*TILE_SIZE+EFFECT_FIRING_ADJUSTMENT + facing_adjustment[1]))

            effects[e][4] += 1

        # Capture effect
        elif effect_id == 3:
            pos_x, pos_y, frame = effect[1], effect[2], effect[3]
            canvas.blit(CAPTURE_EFFECT_ANIMATION[frame], (pos_x*TILE_SIZE+EFFECT_CAPTURE_ADJUSTMENT, pos_y*TILE_SIZE+EFFECT_CAPTURE_ADJUSTMENT))

            effects[e][3] += 1

        # Space jump effect
        elif effect_id == 4:
            pos_x, pos_y, frame = effect[1], effect[2], effect[3]
            canvas.blit(SPACE_JUMP_EFFECT_ANIMATION[frame], (pos_x*TILE_SIZE+EFFECT_SPACE_JUMP_ADJUSTMENT, pos_y*TILE_SIZE+EFFECT_SPACE_JUMP_ADJUSTMENT))

            effects[e][3] += 1


def _render_vision_debug(
    canvas: pygame.Surface,
    player_1_visibility_mask: np.ndarray,
    player_2_visibility_mask: np.ndarray,
    player_1_id: int,
    player_2_id: int
):
    for player in range(2):
        vision_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        player_rectangle = pygame.Surface((TILE_SIZE, TILE_SIZE))
        if player == 0:
            team_color = TEAM_COLORS[player_1_id]
            visibility_mask = player_1_visibility_mask
        else:
            team_color = TEAM_COLORS[player_2_id]
            visibility_mask = player_2_visibility_mask

        player_rectangle.fill(team_color)

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if visibility_mask[y, x]:
                    vision_surface.blit(player_rectangle, (x*TILE_SIZE, y*TILE_SIZE))

        vision_surface.set_alpha(64)
        vision_surface = vision_surface.convert_alpha()
        canvas.blit(vision_surface, (0, 0))


def _get_ship_text_color(ship: list):
    if ship[2] <= 33:
        return (255, 0, 0)
    elif ship[2] >= 66:
        return (255, 255, 255)
    else:
        return (255, 255, 0)