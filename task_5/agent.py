# Skeleton for Agent class
import random
import torch
import numpy as np

# Define role constants for better code readability
ROLE_EXPLORE = "explore"
ROLE_ATTACK = "attack"
ROLE_DEFEND = "defend"

class Agent:

    def load(self, abs_path: str):
        pass

    def eval(self):
        pass

    def to(self, device):
        pass

    def __init__(self, player_id: int):
        self.player_id = player_id
        # Initialize home_planet and enemy_planet as None
        # They will be set on the first call to get_action and never changed again
        self.home_planet = None
        self.enemy_planet = None
        
        # Initialize the ship roles dictionary
        self.ship_roles = {}
        
        # Game state tracking
        self.turn_counter = 0
        self.last_known_enemy_positions = {}
        self.planet_targets = {}  # Track which neutral planets are being targeted

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
        
        # Increment turn counter
        self.turn_counter += 1

        # Set home_planet and enemy_planet only once on the first call
        if self.home_planet is None and planets_occupation:
            self.home_planet = planets_occupation[0]

            # Determine enemy planet based on home planet location
            if self.home_planet[0] == 9:
                self.enemy_planet = (90, 90)
            else:
                self.enemy_planet = (9, 9)

        # Create dictionary for easier access to ships by ID
        allied_ship_dict = {}
        for ship in allied_ships:
            ship_id = ship[0]
            allied_ship_dict[ship_id] = ship
        
        # Store the ship dictionary in obs for use by action functions
        obs["allied_ships_dict"] = allied_ship_dict
        
        # Update the enemy ship positions for tracking
        if enemy_ships:
            for enemy in enemy_ships:
                self.last_known_enemy_positions[enemy[0]] = (enemy[1], enemy[2], self.turn_counter)
        
        # Run the scheduler to update ship roles - pass the original allied_ships list
        # ALL role management should happen in the scheduler
        self.scheduler(obs, allied_ships)

        action_list = []
        # Iterate through the original allied_ships list
        for ship in allied_ships:
            ship_id = ship[0]
            
            # Get the ship's role from the dictionary - the scheduler should have assigned one
            role = self.ship_roles[ship_id]  # Using direct lookup since scheduler ensures all ships have roles
            
            # Execute the appropriate action based on the ship's role
            if role == ROLE_DEFEND:
                action_list.append(get_defense_action(obs, ship_id, self.home_planet))
            elif role == ROLE_EXPLORE:
                action_list.append(get_explore_action(obs, ship_id, self.home_planet))
            elif role == ROLE_ATTACK:
                action_list.append(get_offense_action(obs, ship_id, self.enemy_planet))
            else:
                # This should never happen if scheduler is working correctly
                print(f"Warning: Ship {ship_id} has unknown role: {role}. Defaulting to explorer.")
                action_list.append(get_explore_action(obs, ship_id, self.home_planet))

        return {
            "ships_actions": action_list,
            "construction": 10
        }
    
    def scheduler(self, obs: dict, allied_ships):
        """
        Assigns and updates roles for ships based on the current game state.
        This function dynamically manages ship roles throughout the game.
        
        Logic:
        1. Ensure all ships have an initial role
        2. Reassign roles based on game state (enemy presence, planets, etc.)
        3. Balance roles to maintain a good distribution of explorers, attackers, and defenders
        
        :param obs: The current game observation
        :param allied_ships: List of allied ships
        """
        enemy_ships = obs.get('enemy_ships')
        planets_occupation = obs.get('planets_occupation')
        
        # Track the count of each role for balancing
        role_counts = {ROLE_EXPLORE: 0, ROLE_ATTACK: 0, ROLE_DEFEND: 0}
        
        
        # Step 1: ALWAYS ensure all ships have an initial role
        for ship in allied_ships:
            ship_id = ship[0]  # Extract the ID from the ship
            
            # Check if the ship already has a role
            if ship_id not in self.ship_roles:
                # Assign initial role based on ID (for compatibility with original logic)
                if ship_id % 3 == 2:
                    self.ship_roles[ship_id] = ROLE_DEFEND
                elif ship_id % 3 == 0:
                    self.ship_roles[ship_id] = ROLE_EXPLORE
                else:
                    self.ship_roles[ship_id] = ROLE_ATTACK

            
            # Count the current distribution of roles
            current_role = self.ship_roles[ship_id]
            role_counts[current_role] = role_counts.get(current_role, 0) + 1
            print(role_counts)
        
        # Step 2: Dynamic role reassignment based on game conditions
        
        # Check for nearby enemy ships that might require more defenders
        # enemy_near_home = False
        # if enemy_ships and self.home_planet:
        #     home_x, home_y = self.home_planet[0], self.home_planet[1]
        #     for enemy in enemy_ships:
        #         enemy_x, enemy_y = enemy[1], enemy[2]
        #         distance_to_home = abs(enemy_x - home_x) + abs(enemy_y - home_y)
        #         if distance_to_home < 20:  # Threshold distance to be considered "near"
        #             enemy_near_home = True
        #             break
        
        # # Get the total number of ships
        total_ships = len(allied_ships)
        
        # Step 3: Balance roles based on game state
        
        # Early game strategy (first 50 turns)
        if self.turn_counter < 250:
            target_distribution = {
                ROLE_EXPLORE: max(1, int(total_ships * 1.0)),  # 0% explorers
                ROLE_ATTACK: max(0, int(total_ships * 0)),   # 0% attackers
                ROLE_DEFEND: max(0, int(total_ships * 0))    # 0% defenders
            }
        # Mid game strategy
        elif 250 <= self.turn_counter < 500:
            target_distribution = {
                ROLE_EXPLORE: max(0, int(total_ships * 0)),  # 30% explorers
                ROLE_ATTACK: max(1, int(total_ships * 1.0)),   # 40% attackers
                ROLE_DEFEND: max(0, int(total_ships * 0))    # 30% defenders
            }
        # Late game strategy
        else:
            target_distribution = {
                ROLE_EXPLORE: max(1, int(total_ships * 0.5)),  # 20% explorers
                ROLE_ATTACK: max(0, int(total_ships * 0)),   # 50% attackers
                ROLE_DEFEND: max(1, int(total_ships * 0.5))    # 30% defenders
            }
        
        # Adjust for enemy presence near home base
        # if enemy_near_home:
        #     # Allocate more ships to defense if enemies are near home
        #     target_distribution[ROLE_DEFEND] = max(2, int(total_ships * 0.5))
        #     target_distribution[ROLE_ATTACK] = max(1, int(total_ships * 0.3))
        #     target_distribution[ROLE_EXPLORE] = max(1, total_ships - target_distribution[ROLE_DEFEND] - target_distribution[ROLE_ATTACK])
        
        # Make adjustments to achieve the target distribution
        for role, target_count in target_distribution.items():
            current_count = role_counts.get(role, 0)
            
            # If we need more ships in this role
            while current_count < target_count:
                # Find a role that has excess ships
                for excess_role in role_counts:
                    if excess_role != role and role_counts[excess_role] > target_distribution[excess_role]:
                        # Find a ship to reassign
                        for ship in allied_ships:
                            ship_id = ship[0]
                            if self.ship_roles.get(ship_id) == excess_role:
                                # Consider health before reassigning
                                ship_health = ship[3]
                                
                                # Don't reassign critically damaged ships to attack
                                if role == ROLE_ATTACK and ship_health < 30:
                                    continue
                                    
                                # Reassign the ship
                                self.ship_roles[ship_id] = role
                                role_counts[excess_role] -= 1
                                role_counts[role] += 1
                                current_count += 1
                                
                                # Print role change notification for debugging
                                print(f"Turn {self.turn_counter}: Ship {ship_id} reassigned from {excess_role} to {role}")
                                
                                break
                        
                        # If we've reached the target, break out
                        if current_count >= target_count:
                            break
                
                # If we can't find any more ships to reassign, just break
                if current_count < target_count:
                    break
        
        # Cleanup: Remove entries for ships that no longer exist
        ship_ids = [ship[0] for ship in allied_ships]
        for ship_id in list(self.ship_roles.keys()):
            if ship_id not in ship_ids:
                del self.ship_roles[ship_id]

def get_offense_action(obs: dict, idx: int, enemy_planet: tuple) -> list[int]:
    ship = obs["allied_ships_dict"][idx]
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

def get_explore_action(obs: dict, idx: int, home_planet: tuple, ) -> list[int]:
    """
    Function to explore the map looking for neutral planets to capture.
    Searches for clusters of valuable tiles and moves toward them.
    If none found, moves in a direction away from home planet.
    """
    ship = obs["allied_ships_dict"][idx]
    found = False
    target_x, target_y = None, None
    max_ones_count = -1
    
    # Only try to shoot if firing cooldown is 0
    if ship[4] == 0:  # ship[4] is firing_cooldown
        for enemy in obs["enemy_ships"]:
            choice = shoot_enemy_if_in_range(enemy, ship)
            if choice:
                return choice

    # Look for clusters of valuable tiles (planets/resources)
    for i in range(len(obs['map'])):
        for j in range(len(obs['map'][i])):
            # Check if this is a valuable tile (indicated by specific bit patterns)
            if format(obs['map'][i][j], '08b')[-1] == '1' and format(obs['map'][i][j], '08b')[0:2] == '00':
                # Count nearby valuable tiles to find clusters
                ones_count = sum(
                    1 for x in range(max(0, i-3), min(len(obs['map']), i+3))
                    for y in range(max(0, j-3), min(len(obs['map'][i]), j+3))
                    if format(obs['map'][x][y], '08b')[-1] == '1' and format(obs['map'][x][y], '08b')[0:2] == '00'
                )
                if ones_count > max_ones_count:
                    max_ones_count = ones_count
                    target_x, target_y = i, j
                    found = True

    if not found:
        # If no valuable targets found, move away from home planet
        if home_planet[0] == 9:  # If home is at (9,9), move right or down
            return [ship[0], 0, random.choice([0, 1]), 1]
        else:  # If home is at (90,90), move left or up
            return [ship[0], 0, random.choice([2, 3]), 1]
        

    else:
        # Go towards the identified target
        # Note: The map coordinates and ship coordinates might be flipped (x,y vs y,x)
        dx = ship[1] - target_y  # X distance (ship x - target y)
        dy = ship[2] - target_x  # Y distance (ship y - target x)

        if abs(dx) > abs(dy):
            if dx > 0:
                return [ship[0], 0, 2, min(3, abs(dx))]  # Move left
            else:
                return [ship[0], 0, 0, min(3, abs(dx))]  # Move right
        else:
            if dy > 0:
                return [ship[0], 0, 3, min(3, abs(dy))]  # Move up
            else:
                return [ship[0], 0, 1, min(3, abs(dy))]  # Move down


def get_defense_action(obs: dict, idx: int, home_planet: tuple) -> list[int]:
    ship = obs["allied_ships_dict"][idx]

    for enemy in obs["enemy_ships"]:
        choice = shoot_enemy_if_in_range(enemy, ship)
        if choice:
            return choice
        
    if ship[3] <= 30:
        return return_home_on_low_hp(ship, home_planet[0], home_planet[1])

    return move_randomly_around_home(obs, ship, home_planet[0], home_planet[1])



def shoot_enemy_if_in_range(enemy, ship) -> list[int]:
    """
    Check if an enemy ship is within firing range (8 tiles) and directly aligned
    (same row or column) with our ship.
    
    Ship position: (ship[1], ship[2])
    Enemy position: (enemy[1], enemy[2])
    
    Returns a firing action if enemy is in range, otherwise an empty list.
    """
    ship_x, ship_y = ship[1], ship[2]
    enemy_x, enemy_y = enemy[1], enemy[2]
    
    # Check if ships are in the same row (y-coordinate)
    if ship_y == enemy_y:
        # Enemy is to the right of our ship
        if enemy_x > ship_x and enemy_x - ship_x <= 8:
            return [ship[0], 1, 0]  # Shoot right
        
        # Enemy is to the left of our ship
        if enemy_x < ship_x and ship_x - enemy_x <= 8:
            return [ship[0], 1, 2]  # Shoot left
    
    # Check if ships are in the same column (x-coordinate)
    if ship_x == enemy_x:
        # Enemy is below our ship
        if enemy_y > ship_y and enemy_y - ship_y <= 8:
            return [ship[0], 1, 1]  # Shoot down
        
        # Enemy is above our ship
        if enemy_y < ship_y and ship_y - enemy_y <= 8:
            return [ship[0], 1, 3]  # Shoot up
    
    # Enemy not in range or not aligned
    return []

def move_randomly_around_home(obs : dict, ship, home_x, home_y, max_distance=15) -> list[int]:
    """
    Poruszanie się losowo w obszarze max_distance wokół planety macierzystej.
    """
    ship_x, ship_y = ship[1], ship[2]

    for _ in range(10):
        if home_x == 9 and ship_x <= home_x:
            direction = 0
        elif home_y == 9 and ship_y <= home_y:
            direction = 1
        elif home_x == 90 and ship_x >= home_x:
            direction = 2
        elif home_y == 90 and ship_y >= home_y:
            direction = 3
        else:
            # Losowy wybór kierunku
            direction = random.randint(0, 3)

        # Przewidywana nowa pozycja
        new_x = ship_x + (1 if direction == 0 else -1 if direction == 2 else 0)
        new_y = ship_y + (1 if direction == 1 else -1 if direction == 3 else 0)

        if not (0 <= new_x < 100 and 0 <= new_y < 100):
            continue  # Jeśli poza mapą, ponawiamy próbę

        # Sprawdzenie, czy nowa pozycja mieści się w dozwolonym obszarze wokół planety
        if abs(new_x - home_x) + abs(new_y - home_y) > max_distance:
            continue  # Jeśli poza zakresem, ponawiamy próbę

        # Sprawdzenie, czy pole NIE jest asteroidą
        if is_asteroid(obs, new_x, new_y):
            continue  # Jeśli to asteroida, ponawiamy próbę

        # Jeśli pole jest poprawne, wykonaj ruch
        return [ship[0], 0, direction, 1]  # Ruch o 1 pole w danym kierunku
    
    return [ship[0], 0, direction, 1] 


def return_home_on_low_hp(ship, home_x, home_y) -> list[int]:
    dx = ship[1] - home_x
    dy = ship[2] - home_y

    if abs(dx) > abs(dy):
        # need to move in X direction first
        if dx > 0:
            # need to move left
            return [ship[0], 0, 2, min(3, abs(dx))]
        else:
            # need to move right
            return [ship[0], 0, 0, min(3, abs(dx))]
    else:
        # need to move in Y direction first
        if dy > 0:
            # need to move up
            return [ship[0], 0, 3, min(3, abs(dy))]
        else:
            # need to move down
            return [ship[0], 0, 1, min(3, abs(dy))]
        
def is_asteroid(obs: dict, x, y) -> bool:

    point = obs['map'][x][y]
    
    if format(point, '08b')[-2] == '1':
        return True

    return False
