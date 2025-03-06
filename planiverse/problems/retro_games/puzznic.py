
from copy import deepcopy
from collections import defaultdict
from typing import Tuple

from planiverse.problems.retro_games.base import RetroGame


class Element:
    def __init__(self, letter, pos:Tuple):
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
    
class Level:
    def __init__(self, levelstr):
        self.levelstr = levelstr.strip().split('\n')
        self.grid     = self._parse_level(self.levelstr)
        self.cursor   = self._locate_cursor(self.levelstr)

        self.elements   = set([str(item) for sublist in self.grid for item in sublist])
    
    def __str__(self):
        _map = [[str(cell) for cell in row] for row in self.grid]
        _map[self.cursor.pos[0]][self.cursor.pos[1]] = 'c' if _map[self.cursor.pos[0]][self.cursor.pos[1]] == ' ' else '¢'
        return '\n'.join([''.join(row) for row in _map])
        
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
                # elif cell == 'c': currrow.append(Cursor((x, y)))
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
    
    def _parse_direction(self, direction):
        """!
        This function converts a direction string to a delta (dx, dy).
        """
        # Note the direction is reversed because left and right are preformed on the columns
        # and up and down are performed on the rows.
        directions = {
            'left':  (0, -1),
            'right': (0,  1),
            'up':    (-1, 0),
            'down':  (1,  0),
            'nop':   (0,  0)
        }
        return directions[direction]
    
    def _is_inbound_(self, x, y):
        """!
        This function checks if the position (x, y) is within the grid bounds.
        """
        return 0 <= x < len(self.grid[0]) and 0 <= y < len(self.grid)


    def move(self, direction, hold):
        """!
        This function moves the cursor in the specified direction.
        If hold is True, the cursor will hold the block in its current position.
        """
        # We cannot hold a box and move up or down.
        if hold and direction in ['up', 'down']: return False

        # first check if the cursor is holding a block.
        # if yes then mark this block as marked.
        box_to_move = []
        if hold and isinstance(self.grid[self.cursor.pos[0]][self.cursor.pos[1]], Box):
            box_to_move.append(self.cursor.pos)

        new_x, new_y = self.cursor + self._parse_direction(direction)
        if not self._is_inbound_(new_x, new_y): return False

        for box in box_to_move:
            # we need to update the grid to move the block in the direction
            if isinstance(self.grid[new_x][new_y], EmptySpace):
                self.grid[new_x][new_y] = self.grid[box[0]][box[1]]
                self.grid[box[0]][box[1]].update((new_x, new_y))
                self.grid[box[0]][box[1]] = EmptySpace((box[0], box[1]))

        self.cursor.update((new_x, new_y))
        return True
    
    def update(self, new_grid):
        self.grid = new_grid

class PuzznicGame(RetroGame):
    def __init__(self, levelstr:str):
        self.current_level = None
        self.current_score = 0
        self.levelstr      = levelstr
        self.grid_history  = []
        self.action_space  = [
            'up', 
            'down', 
            'left', 
            'right', 
            'nop',
            'left-hold', 
            'right-hold'
        ]


    # now we need a function that applies gravity to the boxes.
    def _apply_gravity_(self, level:Level):
        """!
        This function applies gravity to the boxes in the level.
        """
        updated_grid = deepcopy(level.grid)
        for ridx, row in enumerate(updated_grid):
            # skip all wall rows
            if all(isinstance(cell, Wall) for cell in row): continue
            for yidx, cell in enumerate(row):
                # We will move the boxes with empty spaces below them.
                if isinstance(cell, Box):
                    # check if the box has empty spaces below it.
                    if ridx + 1 < len(updated_grid) and isinstance(updated_grid[ridx + 1][yidx], EmptySpace):
                        updated_grid[ridx + 1][yidx] = cell
                        updated_grid[ridx][yidx] = EmptySpace((ridx, yidx))
                        cell.update((ridx + 1, yidx))
        return updated_grid

    def _check_and_remove_matches_(self, level:Level):
        """!
        This function checks and removes all horizontal/vertical matches of 2+ blocks.
        """
        updated_grid = deepcopy(level.grid)
        to_remove = set()

        # Check horizontal matches
        for ridx, row in enumerate(updated_grid):
            # skip all wall rows
            if all(isinstance(cell, Wall) for cell in row): continue
            for cidx, cell in enumerate(row):
                if isinstance(cell, EmptySpace) or isinstance(cell, Wall): continue
                # for every box we need to check the four directions if there are any matches.
                for dir in ['left', 'right', 'up', 'down']:
                    newx, newy = cell + level._parse_direction(dir)
                    if not updated_grid[newx][newy].letter == cell.letter: continue
                    to_remove.add((cell.letter, (ridx, cidx)))
                    pass
        
        # remove the matched blocks
        for letter, pos in to_remove:
            updated_grid[pos[0]][pos[1]] = EmptySpace(pos)
        
        return updated_grid

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
            if letters_list.count(l) > 2:
                more_than_two_blocks_score += 50
        return each_casecade_score + more_than_two_blocks_score

    def _parse_action_(self, action):
        if 'hold' in action: return action.split('-')[0], True
        return action, False

    def levels(self):
        return [
            """######\n#12c0#\n###00#\n#0000#\n#2001#\n##21##\n######"""
        ]

    def get_action_space(self):
        return self.action_space
    
    def score(self):
        return self.current_score

    def reset(self):
        """!
        Initialize the game with the specified level.
        """
        self.current_level = Level(self.levelstr)
        self.current_score = 0
        self.grid_history  = []
        self.grid_history  += [deepcopy(self.current_level)]
    
    def step(self, action):
        if self.current_level is None: raise ValueError("Game not initialized.")
        assert action in self.get_action_space(), "Invalid action."
        self.current_level.move(*self._parse_action_(action))
        self.current_level.update(self._apply_gravity_(self.current_level))
        self.grid_history  += [deepcopy(self.current_level)]
        self.current_level.update(self._check_and_remove_matches_(self.current_level))
        self.current_score += self._compute_score_(self.current_level.grid, self.grid_history[-1].grid)
        self.grid_history  += [deepcopy(self.current_level)]
    
    def is_goal(self):
        """!
        A goal state is reached when there are no boxes in the grid.
        """
        return not any([isinstance(item, Box) for sublist in self.current_level.grid for item in sublist])

    def is_terminal(self):
        """!
        A terminal state is a state where no pairs of blocks exist.
        """
        current_boxes = list(filter(lambda o: isinstance(o, Box), [item for sublist in self.current_level.grid for item in sublist]))
        letter_counter = defaultdict(int)
        for letter in map(lambda o: o.letter, current_boxes):
            letter_counter[letter] += 1
        return 1 in set(letter_counter.values())
    
    def successors(self, state):
        raise NotImplementedError("Not implemented yet.")

    def simulate(self, plan):
        raise NotImplementedError("Not implemented yet.")

    def validate(self, plan):
        raise NotImplementedError("Not implemented yet.")



# game = PuzznicGame()
# for i, lvl in enumerate(game.levels()):
#     game.start(lvl)
#     # apply this plan.
#     plan = ['left', 'right-hold', 'left', 'left', 'right-hold', 'right-hold', 'down', 'down', 'down', 'left-hold']
#     plan = ['left', 'right-hold', 'down', 'down', 'down', 'left-hold', 'right', 'up', 'up', 'up', 'left', 'left', 'right-hold', 'right-hold']
#     for action in plan:
#         game.step(action)
    
#     # if game.goal_reached():
#     #     pass
#     # elif game.terminal_state():
#     #     pass

#     # for dev only.
#     for i, grid in enumerate(game.grid_history):
#         print(f"Step {i}")
#         print(grid)
#         print()
#     pass



# level = Level(level1)

# print(level)

# level.move(, False)
# level.update(apply_gravity(level))
# print("After moving left")
# print(level)
# level.move(, True)
# level.update(apply_gravity(level))
# print("After moving right")
# print(level)

# updated_level = deepcopy(level)
# updated_level.update(apply_gravity(updated_level))
# oldgrid = deepcopy(updated_level.grid)
# updated_level.update(check_and_remove_matches(updated_level))
# score = compute_score(updated_level.grid, oldgrid)
# print("After removing matches")
# print(updated_level)
# print("Score:", score)
pass

