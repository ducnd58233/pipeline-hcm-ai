from typing import Any, Dict, Tuple, Optional
from app.models import Category


class GridManager:
    def __init__(self):
        self.grid_state: Dict[Tuple[int, str], Category] = {}
        self.panel_logic: str = "and"
        self.max_objects: Optional[str] = None

    def add_object(self, row: int, col: str, category: Category):
        self.grid_state[(row, col)] = category

    def remove_object(self, row: int, col: str):
        position = (row, col)
        if position in self.grid_state:
            del self.grid_state[position]

    def get_object(self, row: int, col: str) -> Optional[Category]:
        return self.grid_state.get((row, col))

    def get_state(self) -> Dict[Tuple[int, str], Category]:
        return self.grid_state

    def clear(self):
        self.grid_state.clear()

    def set_panel_logic(self, logic: str):
        if logic not in ["and", "or"]:
            raise ValueError("Panel logic must be 'and' or 'or'")
        self.panel_logic = logic

    def get_panel_logic(self) -> str:
        return self.panel_logic

    def set_max_objects(self, max_objects: Optional[str]):
        self.max_objects = max_objects

    def get_max_objects(self) -> Optional[str]:
        return self.max_objects

    def get_query_dict(self) -> Dict[str, Any]:
        return {
            'objects': self.grid_state,
            'logic': self.panel_logic,
            'max_objects': self.max_objects
        }


grid_manager = GridManager()
