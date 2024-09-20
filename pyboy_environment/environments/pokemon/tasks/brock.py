from functools import cached_property

import numpy as np
import time
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

class Location:
    def __init__(self, x: int, y: int, map_id: int, map_name: str) -> None:
        self.x = x
        self.y = y
        self.map_id = map_id
        self.map_name = map_name

    def __eq__(self, other):
        if isinstance(other, Location):
            return self.x == other.x and self.y == other.y and self.map_id == other.map_id and self.map_name == other.map_name
        return False

class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

        self.prev_locations: list[Location] = []
        self.penalties = 0
        self.prev_state = None

    def has_not_been_on_map(self, curent_loc: Location) -> bool:
        for loc in self.prev_locations:
            if curent_loc.map_id == loc.map_id:
                return False
        return True
    
    def last_x_locs_the_same(self, x: int):
        # checks if the last x locations are the same
        if len(self.prev_locations) < x:
            return False
        for i in range(x-2):
            if self.prev_locations[-1 - i] != self.prev_locations[-2 - i]:
                return False
        return True
    
    def has_higher_y(self, loc: Location):
        max_y_for_map = 999
        for prev_loc in self.prev_locations:
            if prev_loc.map_id == loc.map_id and prev_loc.y < max_y_for_map:
                max_y_for_map = prev_loc.y
        return loc.y < max_y_for_map

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        data = self._generate_game_stats()

        x: int = data['location']['x']
        y: int = data['location']['y']
        map_id: int = data['location']['map_id']
        map_name: str = data['location']['map']
        party_size: int = data['party_size']
        ids: list[int] = data['ids']
        pokemon: list[str] = data['pokemon']
        levels: list[int] = data['levels']
        type_id: list[int] = data['type_id']
        types: list[str] = data['type']
        hp_current: list[int] = data['hp']['current']
        hp_max: list[int] = data['hp']['max']
        xp: list[int] = data['xp']
        status: list[int] = data['status']
        badges: int = data['badges']
        caught_pokemon: int = data['caught_pokemon']
        seen_pokemon: int = data['seen_pokemon']
        money: int = data['money']
        events: list[int] = data['events']

        # observation = {
        #     "screens": self.recent_screens,
        #     "health": np.array([self.read_hp_fraction()]),
        #     "level": self.fourier_encode(level_sum),
        #     "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
        #     "events": np.array(self.read_event_bits(), dtype=np.int8),
        #     "map": self.get_explore_map()[:, :, None],
        #     "recent_actions": self.recent_actions
        # }

        return hp_current + levels + [badges] + [map_id] + events

    def _calculate_reward(self, cur_state: dict) -> float:
        # Implement your reward calculation logic here

        # ALL FAIL
        # background = self._get_screen_background_tilemap()
        # walkable = self._get_screen_walkable_matrix()
        # collision = self.game_area_collision()
        
        x: int = cur_state['location']['x']
        y: int = cur_state['location']['y']
        map_id: int = cur_state['location']['map_id']
        map_name: str = cur_state['location']['map']
        party_size: int = cur_state['party_size']
        ids: list[int] = cur_state['ids']
        pokemon: list[str] = cur_state['pokemon']
        levels: list[int] = cur_state['levels']
        type_id: list[int] = cur_state['type_id']
        types: list[str] = cur_state['type']
        hp_current: list[int] = cur_state['hp']['current']
        hp_max: list[int] = cur_state['hp']['max']
        xp: list[int] = cur_state['xp']
        status: list[int] = cur_state['status']
        badges: int = cur_state['badges']
        caught_pokemon: int = cur_state['caught_pokemon']
        seen_pokemon: int = cur_state['seen_pokemon']
        money: int = cur_state['money']
        events: list[int] = cur_state['events']

        reward = -0.1
        
        # Movement
        loc = Location(x, y, map_id, map_name)
        if self.has_not_been_on_map(loc) and loc.map_name != 'OAKS_LAB,':
            print("NEW MAPPPPPPPPP")
            time.sleep(1)
            reward += 10
        elif loc not in self.prev_locations and self.has_higher_y(loc):
            print("Higher Y")
            # time.sleep(1)
            reward += 2
        elif loc not in self.prev_locations:
            print("REWARD")
            reward += 1

        self.prev_locations.append(loc)

        if self.prev_state is None:
            self.prev_state = cur_state

        if sum(hp_current) <= 0:
            print("DEAD")
            time.sleep(2)
            reward -= 10

        if sum(cur_state['xp']) > sum(self.prev_state['xp']):
            print("LEVEL UP")
            time.sleep(2)
            reward += 10
        
        if sum(cur_state['levels']) > sum(self.prev_state['levels']):
            print("POKEMONS LEVEL UP")
            time.sleep(2)
            reward += 20
        
        self.prev_state = cur_state
        len(self.prev_locations)
            
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 500
