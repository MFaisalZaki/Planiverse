
from copy import deepcopy
from collections import defaultdict
from typing import Tuple, List

from planiverse.problems.retro_games.base import RetroGame

class Element:
    def __init__(self, letter:str, pos:Tuple):
        self.letter = letter
        self.pos = pos
    
    def __eq__(self, value):
        return self.letter == value.letter and self.pos == value.pos

    def __str__(self):
        return self.letter

class Box(Element):
    def __init__(self, letter:str, pos:Tuple):
        super().__init__(letter, pos)

    def __add__(self, other):
        if isinstance(other, self.__class__): return self.pos[0] + other.pos[0], self.pos[1] + other.pos[1]
        if isinstance(other, tuple): return self.pos[0] + other[0], self.pos[1] + other[1]
        raise ValueError("Invalid type for addition.")
    
    def update(self, pos:Tuple):
        self.pos = pos

class Cursor(Element):
    def __init__(self, pos:Tuple):
        super().__init__('c', pos)
    
    def __add__(self, other):
        if isinstance(other, self.__class__): return self.pos[0] + other.pos[0], self.pos[1] + other.pos[1]
        if isinstance(other, tuple): return self.pos[0] + other[0], self.pos[1] + other[1]
        raise ValueError("Invalid type for addition.")
    
    def update(self, pos:Tuple):
        self.pos = pos

class Wall(Element):
    def __init__(self, pos:Tuple):
        super().__init__('#', pos)

class EmptySpace(Element):
    def __init__(self, pos:Tuple):
        super().__init__(' ', pos)
    
class PuzznicState:
    def __init__(self, grid:List[List[Element]], cursor:Cursor, score:float):
        self.grid          = grid
        self.cursor        = cursor
        self.score         = score
        self.cleared_boxes = []

        self.shape       = (len(grid), len(grid[0]))
        self.action_map  = {
            'left':       (0, -1),
            'right':      (0,  1),
            'up':         (-1, 0),
            'down':       (1,  0),
            'left-hold':  (0, -1),
            'right-hold': (0,  1),
        }

        self.inbound_check  = lambda pos: 0 <= pos[0] < self.shape[0] and 0 <= pos[1] < self.shape[1]
        self.isvalid_action = lambda action: action in self.action_map.keys()
        self.literals = frozenset([])
        self.__update__()
    
    def __str__(self):
        _map = [[str(cell) for cell in row] for row in self.grid]
        _map[self.cursor.pos[0]][self.cursor.pos[1]] = 'c' if _map[self.cursor.pos[0]][self.cursor.pos[1]] == ' ' else '¢'
        return '\n'.join([''.join(row) for row in _map])

    def __eq__(self, value):
        return self.grid == value.grid and self.cursor == value.cursor

    def __update__(self):
        # this function update the boolean predicates of the state.
        # the representation is simple for now, 
        for ridx, row in enumerate(self.grid):
            for cidx, cell in enumerate(row):
                if isinstance(cell, Wall) or isinstance(cell, EmptySpace): continue
                self.literals |= frozenset([f"at(box-{cell.letter}, {ridx}, {cidx})"])
        self.literals |= frozenset([f"at(cursor, {self.cursor.pos[0]}, {self.cursor.pos[1]})"])
        self.literals |= frozenset([f"score({self.score})"])
        for cleared_box in self.cleared_boxes:
            self.literals |= frozenset([f"cleared(box-{cleared_box.letter}, {cleared_box.pos[0]}, {cleared_box.pos[1]})"])
        # reached goal
        if self.is_goal(): self.literals     |= frozenset(["goal-reached"])
        if self.is_terminal(): self.literals |= frozenset(["terminal-state"])

    def apply_action(self, action:str):
        hold = 'hold' in action
        if hold and action in ['up', 'down']: return False
        new_x, new_y = self.cursor + self.action_map[action]
        if not self.inbound_check((new_x, new_y)): return False
        # don't allow the cursor to move to a wall cell.
        if isinstance(self.grid[new_x][new_y], Wall): return False
        # move box if we are holding it and the next cell is empty
        if hold and\
           isinstance(self.grid[self.cursor.pos[0]][self.cursor.pos[1]], Box) and\
           isinstance(self.grid[new_x][new_y], EmptySpace):
            # move box
            self.grid[new_x][new_y] = self.grid[self.cursor.pos[0]][self.cursor.pos[1]]
            self.grid[self.cursor.pos[0]][self.cursor.pos[1]].update((new_x, new_y))
            # clear old box position
            self.grid[self.cursor.pos[0]][self.cursor.pos[1]] = EmptySpace(self.cursor.pos)
        self.cursor.update((new_x, new_y))
        self.__update__()
    
    def clear_boxes(self, boxes:List[Element]):
        self.cleared_boxes += boxes
        for box in boxes:
            self.grid[box.pos[0]][box.pos[1]] = EmptySpace(box.pos)
        self.__update__()

    def is_goal(self):
        # check that we dont have any boxes left.
        return not any([isinstance(item, Box) for sublist in self.grid for item in sublist])
    
    def is_terminal(self):
        # check if there are no pairs of boxes left.
        current_boxes = list(filter(lambda o: isinstance(o, Box), [item for sublist in self.grid for item in sublist]))
        letter_counter = defaultdict(int)
        for letter in map(lambda o: o.letter, current_boxes):
            letter_counter[letter] += 1
        return 1 in set(letter_counter.values())

class Level:
    def __init__(self, levelstr:str):
        self.levelstr = levelstr.strip().split('\n')
        self.grid     = self._parse_level(self.levelstr)
        self.cursor   = self._locate_cursor(self.levelstr)
        self.state    = PuzznicState(deepcopy(self.grid), deepcopy(self.cursor), 0)

    def _parse_level(self, level):
        """!
        This function parses the level string and returns a 2D grid.
        Each cell is represented by an integer:
        -1: Wall
         0: Empty cell
         1: Block type 1
         2: Block type 2
         ....
         N: Cursor
        """
        row = []
        for x, rowstr in enumerate(level):
            currrow = []
            for y, cell in enumerate(rowstr):
                if cell == '#': currrow.append(Wall((x, y)))
                elif cell in [' ', 'c']: currrow.append(EmptySpace((x, y))) # The cursor is an empty space
                elif cell.isdigit(): 
                    if int(cell) == 0: currrow.append(EmptySpace((x, y)))
                    else: currrow.append(Box(cell, (x, y)))
            row.append(currrow)
        return row

    def _locate_cursor(self, grid):
        """!
        This function locates the cursor in the grid and returns its position.
        The cursor is represented by the highest number in the grid.
        """
        for x, row in enumerate(grid):
            if not 'c' in row: continue
            for y, cell in enumerate(row):
                if cell == 'c':
                    return Cursor((x, y))
        raise ValueError("Cursor not found in the grid.")
    
    def __str__(self):
        return str(self.state)
    
    def reset(self):
        return PuzznicState(deepcopy(self.grid), deepcopy(self.cursor), 0), {}
    
class PuzznicGame(RetroGame):
    def __init__(self, levelstr:str):
        self.state_history = []
        self.state    = None
        self.level    = Level(levelstr)
    
    def __str__(self):
        return str(self.state)

    def _apply_gravity_(self, state:PuzznicState):
        """!
        This function applies gravity to the boxes in the level.
        """
        successor_state = deepcopy(state)
        for ridx, row in enumerate(successor_state.grid):
            # skip all wall rows
            if all(isinstance(cell, Wall) for cell in row): continue
            for yidx, cell in enumerate(row):
                # We will move the boxes with empty spaces below them.
                if isinstance(cell, Box):
                    # check if the box has empty spaces below it.
                    if ridx + 1 < successor_state.shape[0] and isinstance(successor_state.grid[ridx + 1][yidx], EmptySpace):
                        successor_state.grid[ridx + 1][yidx] = cell
                        successor_state.grid[ridx][yidx] = EmptySpace((ridx, yidx))
                        cell.update((ridx + 1, yidx))
        return successor_state

    def _check_and_remove_matches_(self, state:PuzznicState):
        """!
        This function checks and removes all horizontal/vertical matches of 2+ blocks.
        """
        matched_successor_state = deepcopy(state)
        to_remove = set()

        # Check horizontal matches
        for ridx, row in enumerate(matched_successor_state.grid):
            # skip all wall rows
            if all(isinstance(cell, Wall) for cell in row): continue
            for cidx, cell in enumerate(row):
                if isinstance(cell, EmptySpace) or isinstance(cell, Wall): continue
                # for every box we need to check the four directions if there are any matches.
                for dir in ['left', 'right', 'up', 'down']:
                    newx, newy = cell + matched_successor_state.action_map[dir]
                    if not matched_successor_state.grid[newx][newy].letter == cell.letter: continue
                    to_remove.add((cell.letter, (ridx, cidx)))

        matched_successor_state.clear_boxes([Box(letter, pos) for letter, pos in to_remove])
        return matched_successor_state

    def _compute_score_(self, newgrid, oldgrid):
        removed_boxes = set()
        for (ridx, row_newgird), (_, row_oldgrid) in zip(enumerate(newgrid), enumerate(oldgrid)):
            if all(isinstance(cell, Wall) for cell in row_newgird): continue
            for (cidx, cell_newgrid), (_, cell_oldgrid) in zip(enumerate(row_newgird), enumerate(row_oldgrid)):
                if isinstance(cell_newgrid, Wall): continue
                if cell_newgrid == cell_oldgrid: continue
                removed_boxes.add((cell_oldgrid.letter, (ridx, cidx)))
                pass
            pass

        # scoring logic (assumed)
        # Each cleared block awards points (e.g., 10 points per block).
        # Consecutive matches caused by cascading blocks (due to gravity) increase a multiplier
        # Matching more than 2 blocks adds a bonus (e.g., +50 points per extra block).

        each_block_score    = len(removed_boxes) * 10  
        cascaded_blocks     = set(map(lambda o:o[0], removed_boxes))
        each_casecade_score = each_block_score * len(cascaded_blocks) * 1.5 if len(cascaded_blocks) > 1 else each_block_score

        more_than_two_blocks_score = 0
        letters_list = list(map(lambda o:o[0], removed_boxes))
        for l in cascaded_blocks:
            if letters_list.count(l) > 2: more_than_two_blocks_score += 50
        return each_casecade_score + more_than_two_blocks_score

    def _commit_state_(self):
        self.state_history += [deepcopy(self.state)]

    def reset(self):
        self.state, info= self.level.reset()
        self.state_history = [deepcopy(self.state)]
        return self.state, info
    
    def step(self, action:str):
        if self.state is None: raise ValueError("Game not initialized.")
        assert self.state.isvalid_action(action), "Invalid action."
        self._commit_state_() # save a copy of the state.
        self.state.apply_action(action) # apply the action to the state.
        self.state = self._apply_gravity_(self.state)
        self.state = self._check_and_remove_matches_(self.state)
        self.state.score += self._compute_score_(self.state.grid, self.state_history[-1].grid)
        self._commit_state_()
        return self.state, self.state.score

    def render(self):
        # first remove the duplicate states
        unique_states = []
        for state in self.state_history:
            if state not in unique_states: unique_states.append(state)
        for t, state in enumerate(unique_states):
            print(f"Step: {t}")
            print(state)
            print('--------------')

    def is_goal(self, state):
        return state.is_goal()
    
    def is_terminal(self, state):
        return state.is_terminal()
    
    def successors(self, state):
        ret_successors = []
        for action in state.action_map.keys():
            new_state = deepcopy(state)
            new_state.apply_action(action)
            new_state = self._apply_gravity_(new_state)
            new_state = self._check_and_remove_matches_(new_state)
            new_state.score += self._compute_score_(new_state.grid, state.grid)
            if state == new_state: continue # skip the state if it is the same as the current state.
            ret_successors.append((action, new_state))
        return ret_successors

    def simulate(self, plan):
        raise NotImplementedError("Not implemented yet.")

    def validate(self, plan):
        raise NotImplementedError("Not implemented yet.")
    
    def goal(self):
        raise NotImplementedError("Not implemented yet.")
