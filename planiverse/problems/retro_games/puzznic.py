
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
    
    def __hash__(self):
        return hash((self.letter, self.pos))

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
    def __init__(self, grid:List[List[Element]], cursor:Cursor, score:List[float], cleared_boxes:List[Element]=[]):
        self.grid          = [row[:] for row in grid]
        self.cursor        = Cursor(cursor.pos)  
        self.score         = score[:]  
        self.cleared_boxes = cleared_boxes[:]

        self.shape       = (len(grid), len(grid[0]))
        
        self.action_map  = {
            'left':       (0, -1),
            'right':      (0,  1),
            'up':         (-1, 0),
            'down':       (1,  0),
            'left-hold':  (0, -1),
            'right-hold': (0,  1),
        }

        self.inbound_check  = lambda pos: 0 <= pos[0] < self.shape[0]-1 and 0 <= pos[1] < self.shape[1]-1
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
        
        self.literals = frozenset([])
        
        self.literals |= frozenset([f"at(cursor, {self.cursor.pos[0]}, {self.cursor.pos[1]})"])

        for ridx, row in enumerate(self.grid):
            for cidx, cell in enumerate(row):
                if isinstance(cell, Wall) or isinstance(cell, EmptySpace): continue
                self.literals |= frozenset([f"at(box-{cell.letter}, {ridx}, {cidx})"])
        
        for cleared_box in self.cleared_boxes:
            self.literals |= frozenset([f"cleared(box-{cleared_box.letter}, {cleared_box.pos[0]}, {cleared_box.pos[1]})"])

        if self.is_goal(): 
            self.literals |= frozenset(["goal-reached"])
            self.literals |= frozenset([f"score({sum(self.score)})"])

        if self.is_terminal(): 
            self.literals |= frozenset(["terminal-state"])
            self.literals |= frozenset([f"score({sum(self.score)})"])

    def apply_action(self, action:str):
        hold = 'hold' in action
        if hold and action in ['up', 'down']: return False
        marked_cell = self.grid[self.cursor.pos[0]][self.cursor.pos[1]]
        # do nothing if hold is true and the cursor is not on a box.
        if hold and not isinstance(marked_cell, Box):  return False
        new_x, new_y = self.cursor + self.action_map[action]
        if not self.inbound_check((new_x, new_y)): return False
        # don't allow the cursor to move to a wall cell.
        # if isinstance(self.grid[new_x][new_y], Wall): return False
        # move box if we are holding it and the next cell is empty
        if hold and\
           isinstance(self.grid[new_x][new_y], EmptySpace):
            self.grid[new_x][new_y] = Box(marked_cell.letter, (new_x, new_y))
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
        self.state    = PuzznicState(self.grid, self.cursor, [])

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
        return PuzznicState(self.grid, self.cursor, []), {}
    
class PuzznicGame(RetroGame):
    def __init__(self):
        self.state_history = []
        self.state     = None
        self.level     = None
        self.index     = 0
        self.levelsstr = [
            """######\n#12c #\n###  #\n#    #\n#2  1#\n##21##\n######""",
            """#######\n#  c ##\n#  1  #\n#  2  #\n# 13  #\n# 24  #\n#243 3#\n#######""",
            """########\n###  ###\n##  c ##\n#1 78 1#\n#8 ## 7#\n##8  7##\n###  ###\n########""",
            """#######\n##8####\n#67c###\n##6 6 #\n##7 7##\n#####8#\n#######""",
            """#####\n#3c1#\n#2 2#\n## 4#\n#  2#\n#  4#\n#1#3#\n#####""",
            """#######\n#c  ###\n#57  ##\n##67  #\n###5 6#\n#### 5#\n#######""",
            """#######\n#     #\n#c    #\n#2   8#\n##1  ##\n#18 78#\n#21 87#\n##2878#\n#######""",
            """######\n#c 21#\n#  13#\n#  32#\n#  21#\n######""",
            """########\n#c    2#\n#     3#\n#5  4 5#\n#43 3 ##\n#352#5##\n########""",
            """#######\n###c5##\n#7  65#\n##7 56#\n### 6##\n#######""",
            """########\n#c   27#\n#  8 ###\n#  #   #\n#7 #   #\n###  12#\n#2  821#\n########""",
            """#########\n#654c456#\n#### ####\n###   ###\n##5   5##\n###   ###\n###   ###\n###654###\n#########""",
            """##########\n#4323c234#\n##### ####\n#####  ###\n#####  ###\n####2  ###\n##### ####\n##########""",
            """#######\n##1c ##\n##2  ##\n# 3 31#\n##2#1##\n#######""",
            """#######\n###2c##\n#2 1 2#\n## 2 1#\n#1 #12#\n##1####\n#######""",
            """##########\n#        #\n#  c632  #\n#   5#8  #\n#   ###  #\n#686  4 3#\n#8#7  8#2#\n####  ####\n#  565   #\n#  7#4   #\n#  ###   #\n##########""",
            """#########\n#343c231#\n##31 14##\n###2 4###\n####1####\n#########""",
            """########\n##2#8#2#\n##1#1#1#\n##8c8 2#\n### ####\n#  1  ##\n#1 #  ##\n##   ###\n### ####\n###1####\n########""",
            """#######\n#456c4#\n#3#346#\n#2 235#\n## ####\n## ####\n#32####\n#24####\n#######""",
            """#######\n###6c##\n###5  #\n#7 7 ##\n#5 6 ##\n## 5  #\n###6###\n#######""",
            """########\n###3 ###\n#### 3##\n#65  4##\n#46c 35#\n#54 45##\n### ####\n########""",
            """#######\n### ###\n### 8##\n## c7 #\n#6  1 #\n##87#1#\n#####6#\n#######""",
            """#######\n## 8###\n###17##\n#7c81##\n#6 76 #\n#######""",
            """########\n#4 c   #\n####   #\n#5   45#\n###  ###\n###5  ##\n###4  ##\n########""",
            """#########\n#######4#\n#   c7 2#\n#8 3 5 7#\n#2 4 6 6#\n#8 5 3 3#\n#########""",
            """#######\n#4c4  #\n## 3 3#\n#3 5 5#\n#4 ####\n#######""",
            """########\n#3 c  1#\n#4    3#\n### 23##\n##484###\n###1####\n###8 ###\n## 1####\n###2####\n########""",
            """#######\n#47 c##\n###4 ##\n###5 ##\n#### ##\n#### ##\n#675 ##\n###6 ##\n#47576#\n#######""",
            """########\n#c2341 #\n# ###5 #\n#  # 2 #\n#1   #5#\n#43# 21#\n########""",
            """######\n#34 c#\n#23  #\n#42 1#\n### 2#\n#24 3#\n#1# 4#\n######""",
            """########\n#      #\n# 1  3 #\n# 5  1 #\n# 1 c3 #\n#1#  #3#\n###1 ###\n## 42 ##\n#2 3425#\n##1235##\n########""",
            """########\n##67####\n###1####\n###8####\n###2####\n##c1 ###\n#  7  2#\n#7 #1 8#\n#676821#\n########""",
            """#######\n### 2##\n##  12#\n#1c38##\n##282 #\n###31##\n#######""",
            """########\n####c6##\n####7###\n##8 8  #\n##68#16#\n#67###1#\n########""",
            """#########\n#8c8 ####\n#787 51 #\n##78 18 #\n# 81 87 #\n# 81 76 #\n# 87 65 #\n# ## ## #\n#       #\n#51 18 8#\n#78#6##5#\n#########""",
            """##########\n##      ##\n#6   c6  #\n#787  73 #\n#816  361#\n#387 #182#\n#138##827#\n##########""",
            """#######\n##1c###\n###  6#\n#6   7#\n#7  61#\n#8 87##\n#7 ####\n#######""",
            """#########\n####c 87#\n####  ###\n####  18#\n#### 867#\n##   ####\n#17######\n#68######\#76######\n#########""",
            """########\n#4#c 7 #\n#7#  5 #\n#6 8 65#\n###7 #7#\n####4#8#\n########""",
            """######\n#65c #\n#43  #\n#21 ##\n##8 ##\n### ##\n## 8##\n#  31#\n#  54#\n#  62#\n######"""
        ]
    
    def __str__(self):
        return str(self.state)

    def _apply_gravity_(self, state:PuzznicState):
        """!
        This function applies gravity to the boxes in the level.
        """
        successor_state = PuzznicState(state.grid, state.cursor, state.score, state.cleared_boxes)
        for ridx, row in enumerate(successor_state.grid):
            # skip all wall rows
            if all(isinstance(cell, Wall) for cell in row): continue
            if all(isinstance(cell, EmptySpace) for cell in row): continue
            for yidx, cell in enumerate(row):
                if not isinstance(cell, Box): continue
                # We will move the boxes with empty spaces below them.
                # check if the box has empty spaces below it.
                if isinstance(successor_state.grid[ridx + 1][yidx], EmptySpace):
                    successor_state.grid[ridx + 1][yidx] = Box(cell.letter, (ridx + 1, yidx))
                    successor_state.grid[ridx][yidx]     = EmptySpace((ridx, yidx))
        return successor_state

    def _check_and_remove_matches_(self, state:PuzznicState):
        """!
        This function checks and removes all horizontal/vertical matches of 2+ blocks.
        """
        matched_successor_state = PuzznicState(state.grid, state.cursor, state.score, state.cleared_boxes) #deepcopy(state)
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
                    to_remove.add(cell)
        assert len(to_remove) != 1, "Invalid state, more than one box to remove."
        matched_successor_state.clear_boxes(to_remove)
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
        return [each_casecade_score + more_than_two_blocks_score]

    def _commit_state_(self):
        self.state_history += [PuzznicState(self.state.grid, self.state.cursor, self.state.score, self.state.cleared_boxes)]
    
    def _compute_successor_state_(self, state:PuzznicState, action:str):
        # don't generate successors for goal/terminal states.
        successor_state = PuzznicState(state.grid, state.cursor, state.score, state.cleared_boxes)
        if state.is_goal() or state.is_terminal(): return successor_state
        successor_state.apply_action(action)
        successor_state = self._apply_gravity_(successor_state)
        successor_state = self._check_and_remove_matches_(successor_state)
        successor_state.score += self._compute_score_(successor_state.grid, state.grid)
        return successor_state

    def _levels_str_(self, index):
        assert 0 <= index < len(self.levelsstr), "Invalid level index."
        return self.levelsstr[index]

    def fix_index(self, index:int):
        self.index = index

    def reset(self):
        self.level = Level(self._levels_str_(self.index))
        self.state, info = self.level.reset()
        self.state_history = [PuzznicState(self.state.grid, self.state.cursor, self.state.score, self.state.cleared_boxes)]
        return self.state, info
    
    def step(self, action:str):
        if self.state is None: raise ValueError("Game not initialized.")
        assert self.state.isvalid_action(action), "Invalid action."
        self._commit_state_() # save a copy of the state.
        self.state = self._compute_successor_state_(self.state, action)
        self._commit_state_()
        return self.state, self.state.score

    def render(self):
        # first remove the duplicate states
        ret_render_txt = []
        unique_states = []
        for state in self.state_history:
            if state not in unique_states: unique_states.append(state)
        for t, state in enumerate(unique_states):
            print(f"Step: {t}")
            print(state)
            ret_render_txt.append(str(state))
            print('--------------')
        return ret_render_txt
    
    def is_goal(self, state):
        return state.is_goal()
    
    def is_terminal(self, state):
        return state.is_terminal()
    
    def successors(self, state):
        ret_successors = []
        for action in state.action_map.keys():
            new_state = self._compute_successor_state_(state, action)
            if state == new_state: continue # skip the state if it is the same as the current state.
            ret_successors.append((action, new_state))
        return ret_successors

    def simulate(self, plan):
        state, _ = self.level.reset()
        ret_states_trace = [PuzznicState(state.grid, state.cursor, state.score, state.cleared_boxes)]
        for action in plan:
            ret_states_trace.append(self._compute_successor_state_(ret_states_trace[-1], action))
        return ret_states_trace

    def validate(self, plan):
        return self.simulate(plan)[-1].is_goal()
    
    def get_actions(self):
        return list(self.state.action_map.keys())        

