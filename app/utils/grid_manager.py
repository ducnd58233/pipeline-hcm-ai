from typing import Dict, Tuple


class GridManager:
    def __init__(self):
        self.grid_state: Dict[Tuple[int, str], str] = {}

    def add_object(self, row: int, col: str, category: str):
        self.grid_state[(row, col)] = category

    def remove_object(self, row: int, col: str):
        position = (row, col)
        if position in self.grid_state:
            del self.grid_state[position]

    def get_object(self, row: int, col: str) -> str:
        return self.grid_state.get((row, col))

    def get_state(self) -> Dict[Tuple[int, str], str]:
        return self.grid_state

grid_manager = GridManager()