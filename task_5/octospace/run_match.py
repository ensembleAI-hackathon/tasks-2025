import argparse
import gymnasium as gym
import numpy as np

from importlib.machinery import SourceFileLoader

from matches_config import TEAMS
from simulation import simulate_game



def get_parser():
    parser = argparse.ArgumentParser(description='Run matches between agents')
    parser.add_argument('path_to_agent_1', type=str, help="Path to the 1st player's agent")
    parser.add_argument('path_to_agent_2', type=str, help="Path to the 2nd player's agent")
    parser.add_argument('--n_matches', type=int, default=3, help='Number of matches to run')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--render_mode', type=str, default=None, help='Render mode')
    parser.add_argument('--turn_on_music', type=bool, default=False, help='Music')
    return parser



def run_match(
        n_matches: int,
        agent_1_path: str,
        agent_2_path: str,
        render_mode: str = None,
        verbose: bool = False,
        turn_on_music: bool = False
):
    # Disable warnings in the gym
    if not verbose:
        gym.logger.min_level = 40

    player_1_id = 46
    player_2_id = 47

    agent_1 = SourceFileLoader('agent_1', agent_1_path).load_module()
    agent_2 = SourceFileLoader('agent_2', agent_2_path).load_module()

    score = simulate_game(player_1_id=player_1_id, player_2_id=player_2_id, player_1_agent_class=agent_1.Agent,
                            player_2_agent_class=agent_2.Agent, n_games=n_matches,
                            render_mode=render_mode, verbose=False, turn_on_music=turn_on_music)

    print(f'{TEAMS[player_1_id]} vs {TEAMS[player_2_id]}: {score}')


if __name__ == '__main__':
    parse = get_parser()
    args = parse.parse_args()

    run_match(n_matches=args.n_matches, agent_1_path=args.path_to_agent_1, agent_2_path=args.path_to_agent_2,
              verbose=args.verbose, render_mode=args.render_mode, turn_on_music=args.turn_on_music)

    """
    Example execution:
        python run_match.py ../agent.py ../agent.py --n_matches=1 --render_mode=human --turn_on_music=True
        
    !IMPORTANT!
    If it happens, that you have a smaller screen on your computer and the game window doesn't render correctly,
    go to the octospace/envs/game_config.py and change the WINDOW_SIZE value (around 800 should be fine).
    """