"""Puzznic in pure Python: no ROM, no emulator, no dependencies.

The sibling [`puzznic_gb`](../gameboy/puzznic_gb.py) drives the real cartridge; this one
implements the rules directly, so it is the dependency-free way to plan in this game.

## Where the levels came from

All 128 of them are the cartridge's 128 rounds, at matching indices: `set_index(7)` here
and on `puzznic_gb` are the same board. The first 50 were transcribed by hand; the rest
were read out of `Puzznic (J)` at `$DF00` by booting each round through `PuzznicGBEnv`.

Hand transcription is why the first 50 are the ones that had errors: level 23 was missing
two walls and level 34 had two pairs of block types transposed, both found by reading the
cartridge back and both confirmed independently against a third-party port of the same
game. `tests/test_puzznic.py` pins those cells, and `tests/test_solutions.py` replays a
known solution per level so a level that silently changes fails loudly.

## Where this differs from the cartridge

Two known gaps, neither yet resolved:

1. **The cursor starts in the wrong place.** The cartridge starts it on a block in 127 of
   the 128 rounds; a `c` in a level string marks an *empty* cell, so it cannot be written
   down. Every level here starts the cursor on the nearest empty cell instead. Harmless
   for solvability (the cursor moves freely), but plans do not transfer move-for-move.
2. **Matching is more eager than the cartridge's.** `_check_and_remove_matches_` rescans
   the whole board after every step and clears *every* adjacent same-type pair, while the
   cartridge will leave such a pair sitting untouched. Replaying cartridge-validated plans
   here clears 30 of 39; the 9 failures are all this, and always in the same direction:
   this version clearing blocks the cartridge keeps.
"""
from itertools import chain
from copy import deepcopy
from collections import defaultdict
from typing import Tuple, List

from planiverse.environments.base import Environment

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
        # this function updates the boolean predicates of the state.
        # the representation is simple for now,
        self.literals = frozenset([f"at(cursor, {self.cursor.pos[0]}, {self.cursor.pos[1]})"])
        
        # for box in filter(lambda o: isinstance(o, Box), [item for sublist in self.grid for item in sublist]):
        for box in filter(lambda o: isinstance(o, Box), chain.from_iterable(self.grid)):
            self.literals |= frozenset([f"at(box-{box.letter}, {box.pos[0]}, {box.pos[1]})"])
        
        # boxes_in_grid = set(map(lambda e:e.letter, filter(lambda o: isinstance(o, Box), [item for sublist in self.grid for item in sublist])))
        boxes_in_grid = set(map(lambda e:e.letter, filter(lambda o: isinstance(o, Box), chain.from_iterable(self.grid))))
        for cleared_box in self.cleared_boxes:
            self.literals |= frozenset([f"cleared(box-{cleared_box.letter}, {cleared_box.pos[0]}, {cleared_box.pos[1]})"])
            if not cleared_box.letter in boxes_in_grid:
                self.literals |= frozenset([f"all-boxes-matched(box-{cleared_box.letter})"])

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
        # do not allow the cursor to move to a wall cell.
        # if isinstance(self.grid[new_x][new_y], Wall): return False
        # move box if we are holding it and the next cell is empty
        if hold and \
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
        # check that we do not have any boxes left.
        # return not any([isinstance(item, Box) for sublist in self.grid for item in sublist])
        return not any([isinstance(item, Box) for item in chain.from_iterable(self.grid)])
    
    def is_terminal(self):
        # check if there are no pairs of boxes left.
        letter_counter = defaultdict(int)
        # for letter in map(lambda o: o.letter, filter(lambda o: isinstance(o, Box), [item for sublist in self.grid for item in sublist])):
        for letter in map(lambda o: o.letter, filter(lambda o: isinstance(o, Box), chain.from_iterable(self.grid))):
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
    
class PuzznicGame(Environment):
    def __init__(self):
        self.state_history = []
        self.state     = None
        self.level     = None
        self.index     = 0
        self.levelsstr = [
            """######\n#12c #\n###  #\n#    #\n#2  1#\n##21##\n######""",
            """#######\n#  c ##\n#  1  #\n#  2  #\n# 13  #\n# 24  #\n#243 3#\n#######""",
            """########\n###  ###\n##  c ##\n#1 78 1#\n#8 ## 7#\n##8  7##\n###  ###\n########""",
            """#######\n##8####\n#67c###\n##6 6 #\n##7 7##\n####8##\n#######""",
            """#####\n#3 1#\n#2c2#\n## 4#\n#  2#\n#  4#\n#1#3#\n#####""",
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
            """########\n#4 c  ##\n####  ##\n#5   45#\n###  ###\n###5  ##\n###4  ##\n########""",
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
            """#########\n#8c8 ####\n#787 51 #\n##78 18 #\n# 81 87 #\n# 18 76 #\n# 87 65 #\n# ## ## #\n#       #\n#51 18 8#\n#87#6##5#\n#########""",
            """##########\n##      ##\n#6   c6  #\n#787  73 #\n#816  361#\n#387 #182#\n#138##827#\n##########""",
            """#######\n##1c###\n###  6#\n#6   7#\n#7  61#\n#8 87##\n#7 ####\n#######""",
            """#########\n####c 87#\n####  ###\n####  18#\n#### 867#\n##   ####\n#17######\n#68######\n#76######\n#########""",
            """########\n#4#c 7 #\n#7#  5 #\n#6 8 65#\n###7 #7#\n####4#8#\n########""",
            """######\n#65c #\n#43  #\n#21 ##\n##8 ##\n### ##\n## 8##\n#  31#\n#  54#\n#  62#\n######""",
            """#######\n#c8 ###\n# 7 2 #\n# 273 #\n# 321 #\n# 738 #\n#81712#\n####8##\n#######""",
            """#######\n###c6##\n# 4 5 #\n# 5 8 #\n# 857 #\n#4785 #\n#56746#\n##4658#\n###4###\n#######""",
            """##########\n###   ####\n#2  18  ##\n##  8#   #\n#7c 7   2#\n##1 #  ###\n##7###8###\n##########""",
            """########\n###67###\n##c215##\n#  3785#\n#  2143#\n## 483##\n###56###\n########""",
            """########\n##c 52##\n##4 2###\n##3 3 3#\n#32 4 2#\n#25 32##\n#32 23##\n###2####\n########""",
            """########\n#1c ####\n#4 #4###\n##2 23##\n##3 123#\n### 21##\n####1###\n########""",
            """#########\n##2 1 c8#\n# 8 2  ##\n# 1212 ##\n###### ##\n#########""",
            """########\n#34537##\n#####6 #\n#c345# #\n# ###7 #\n#    3 #\n##   # #\n#    5 #\n# 65645#\n########""",
            """########\n#21c  7#\n#878  8#\n##271 7#\n##### ##\n##### ##\n########""",
            """#########\n###4#2###\n###6#5###\n###2#4###\n#353c365#\n#### ####\n## 1 81##\n#### ####\n## 71  ##\n####8####\n####7####\n#########""",
            """#######\n#8# c7#\n#7# 82#\n#3 218#\n## 381#\n## 81##\n## 7###\n#######""",
            """########\n###  ###\n###  ###\n#78  c6#\n#542 82#\n#637231#\n###54###\n###35###\n###16###\n###23###\n###61###\n########""",
            """#######\n#c ####\n#7 ####\n#1 ####\n#8  ###\n#7    #\n## 178#\n#  7#7#\n#  8#8#\n#  787#\n#######""",
            """########\n# 4#c6##\n# 3# ###\n# ##  ##\n# 25 45#\n# 43 36#\n# ## ###\n#432 4##\n#### 2##\n########""",
            """########\n##c2  ##\n## 7  4#\n## 8  6#\n#45#  5#\n#68# 3##\n#7## 4##\n#6   ###\n#2 245##\n##3#####\n########""",
            """#######\n#5c3  #\n#6 4  #\n## #  #\n#     #\n#     #\n#  4  #\n#6 6#4#\n#3 3###\n#5 5###\n##5####\n#######""",
            """##########\n##78c#####\n##476  ###\n#4####  ##\n#5    5  #\n#4 676787#\n##6#######\n##########""",
            """##########\n###3c2####\n#  2 #####\n#3#1    4#\n####  41##\n####  8###\n###1 8####\n###8 2####\n##########""",
            """##########\n##########\n##########\n##3   c2##\n#18 32 71#\n#47 ## 18#\n###    ###\n###   4###\n####47####\n##########\n##########""",
            """##########\n####c1####\n#### 5####\n#### 6####\n#    5 42#\n#7   6327#\n#### 5####\n#### 3####\n#### 4####\n#### 3####\n#### 1####\n##########""",
            """#########\n####c####\n###128###\n###312###\n## 8#3 ##\n## ### ##\n#  313 2#\n# 3### 8#\n#38###81#\n#########""",
            """#########\n# 8548c##\n# #### 4#\n#   2  1#\n#1  1  5#\n#8 58  3#\n#581#432#\n###8#####\n#########""",
            """##########\n#65   c13#\n#321  452#\n#263  234#\n#436  #43#\n##########""",
            """########\n#  #####\n# c#####\n# 7#####\n# 4#####\n# 8#####\n#31#####\n#12  ###\n#35# ###\n#528  7#\n#683436#\n########""",
            """##########\n#    c4  #\n#3  2 1  #\n#8  1234 #\n#7  2145 #\n#8  76546#\n##########""",
            """#########\n###c1####\n### ## 8#\n##     6#\n##   8 ##\n#7 281 ##\n##61#7 7#\n##27#818#\n#########""",
            """#########\n####6c6##\n### 4 5##\n### 3 3##\n#35 4 42#\n##3 # 23#\n##4 # ###\n#########""",
            """##########\n#####  c4#\n####   28#\n###   68##\n##    8###\n# 7   ####\n# 81     #\n# 7254   #\n###7615  #\n#####686 #\n#######4 #\n##########""",
            """##########\n###1216###\n## 7##8c##\n#  #  #  #\n#  4  2  #\n# 251542 #\n# #6##2# #\n# #2##1# #\n#8#3  3#7#\n##########""",
            """##########\n#8  c6785#\n##   765##\n### ##5###\n###  61###\n#### 1####\n##########""",
            """##########\n#     c  #\n#  31 1  #\n#  18 8  #\n# 537 735#\n# 246 642#\n##########""",
            """##########\n#8 7857#4#\n#7 #####5#\n#6   67#4#\n#54  ###6#\n###  5 c5#\n#6#  4  6#\n#5#  # ###\n#4   # ###\n##########""",
            """########\n##c6####\n## 54 3#\n## 75 5#\n##68# 4#\n# 75# ##\n# 83####\n# ######\n########""",
            """##########\n###### c #\n###### 8 #\n##8# #12 #\n##1   #8 #\n##8 2  # #\n# # 8   3#\n#8 #13  8#\n##8 283###\n##########""",
            """#######\n###7###\n##c1 ##\n## 5###\n## 6 ##\n#  5 ##\n# 61 ##\n# 76 5#\n# 871##\n##18###\n#######""",
            """########\n#23##18#\n#524582#\n##3##7##\n#56 c67#\n##4##1##\n########""",
            """#########\n##1c8####\n##8 #7###\n###1 68 #\n##68 #7 #\n# 7# ## #\n# 6# # 6#\n# ## # 1#\n#76# # ##\n##7    ##\n#########""",
            """########\n#7######\n#2c#####\n#8 7 ###\n## 2 ###\n## 7 8##\n#  # 1##\n##2  8##\n###  2 #\n#####1 #\n########""",
            """##########\n# 76 583c#\n# 3# ### #\n# ## 373 #\n#  #6### #\n#   52 5 #\n#  #1713 #\n#3####253#\n#8 # 5424#\n##3#####5#\n###   4 4#\n##########""",
            """#########\n## c4354#\n## 26435#\n#  ######\n#     62#\n# 3   34#\n#2#   ###\n#########""",
            """##########\n#### c2  #\n#### 367 #\n#####245 #\n#### ###4#\n#    3 #6#\n# 45 ## 5#\n# 54 3  7#\n##########""",
            """########\n###  ###\n##    ##\n#21  c2#\n#323424#\n#146365#\n#431632#\n##1232##\n###56###\n########""",
            """########\n###  ###\n#8 78c6#\n## ## ##\n#8    1#\n##7  6##\n#18  71#\n###67###\n########""",
            """##########\n# 13c5   #\n# 245#5  #\n# ###### #\n#   5#6  #\n# #5###36#\n# 1# # #3#\n# #1##2 ##\n# #2 ##2##\n#  # # # #\n#7 7 3 4 #\n##########""",
            """########\n#32#####\n#13c####\n#72  ###\n###  8 #\n### 328#\n### 1#7#\n### 3###\n########""",
            """##########\n#   6  c2#\n#   ###43#\n##85#  5##\n###41  ###\n##### ####\n#### 4####\n###4 5 ###\n## 6#1 5##\n#6 7#4#6 #\n#273#6#86#\n##########""",
            """########\n###  ###\n##    ##\n#7c    #\n#2182 7#\n##8217##\n###87###\n########""",
            """##########\n#####c5###\n#### 76###\n#  675# ##\n#  5###7 #\n#  #4# #4#\n##7#6 5 7#\n##### 4 5#\n#####565##\n##########""",
            """##########\n##########\n### 343c##\n##3 ### ##\n##4 ### ##\n### 343 ##\n####### ##\n#######3##\n#######4##\n#######3##\n##########\n##########""",
            """##########\n#######c4#\n###2    3#\n#47#   5##\n#76    ###\n####  6 ##\n##  4 4 3#\n##4 5 5 ##\n#63 4 4 ##\n##5 # 3 ##\n##35# #32#\n##########""",
            """##########\n# 1423c3 #\n# ###### #\n#4   1## #\n#2   2## #\n## ##### #\n#     ## #\n#    1## #\n#1###### #\n#2    3  #\n##### 2 ##\n##########""",
            """########\n########\n## 2c2##\n## 4 4##\n#3 2 2##\n#243 42#\n#42# 24#\n#### ###\n####4###\n########""",
            """##########\n#######38#\n#  187##5#\n# 7823#c4#\n# 6134##3#\n# ##### 5#\n#    ## ##\n#    #  1#\n# 4  #  3#\n# #     4#\n# 2# #  6#\n##########""",
            """##########\n#17   c2##\n#687   5##\n#2785 16##\n##### 657#\n##### ####\n##### ####\n##########""",
            """##########\n#  c     #\n#  8     #\n#  5     #\n#  8#    #\n#  6     #\n#  1  #  #\n# 56     #\n#371     #\n#745  #7##\n#374  #5##\n##########""",
            """##########\n##     c7#\n#18     2#\n###  8 28#\n#818 1281#\n#18#12##8#\n#7########\n##########""",
            """##########\n#c1  #####\n# ## #####\n#4  4#####\n##  ###8##\n#   2  7 #\n#   6  3 #\n#7 2## 7 #\n#673## 8 #\n#568  175#\n##########""",
            """##########\n#   57c  #\n#34364 56#\n####73####\n# 4 47 3 #\n# #7## # #\n# 4####3 #\n##########""",
            """########\n####5#c#\n####6#4#\n####3#2#\n#63 183#\n### ####\n##5 ####\n### ####\n###7   #\n####1#7#\n####248#\n########""",
            """##########\n#87  c7###\n###  24###\n#1   813##\n#2   3723#\n#42878184#\n#74#13417#\n##########""",
            """#########\n##c1#####\n##2## ###\n##6   ###\n##2 6 ###\n##7 7  ##\n##8 8  6#\n#87 #718#\n#########""",
            """########\n#5# 4c5#\n#4  3 4#\n#5 4# 5#\n## #4 4#\n#5 #345#\n## #####\n########""",
            """##########\n#67 5c6 ##\n#5# # # ##\n#6# # # 5#\n#7    # ##\n###7# # ##\n###8 8  ##\n##########""",
            """##########\n## 2#1c###\n## ##3 ###\n##6 5#5###\n### 6 3###\n### 5 6 ##\n##  # 3 2#\n##1 # 5 6#\n##### 36##\n##########""",
            """##########\n##4 ##c3##\n#2# ## #2#\n#1  34  1#\n##  ##  ##\n##########""",
            """########\n###c1###\n##  41##\n#   345#\n#   532#\n##  21##\n### 5###\n########""",
            """#########\n#c54#####\n# #### 5#\n# #585 7#\n# #678 6#\n#7#### ##\n#6   5  #\n#8  7857#\n#6 765#4#\n#########""",
            """#########\n###c7####\n#3# #####\n#4#   7 #\n#65 5 5 #\n####4 #4#\n#5763  6#\n######4##\n#########""",
            """#######\n#254c3#\n#425 5#\n#6## 2#\n#3 3 ##\n#### ##\n###6 ##\n#######""",
            """########\n#7c8416#\n## #####\n#  6 ###\n# 678 ##\n# 356  #\n#82841 #\n#76#186#\n#2###75#\n######3#\n########""",
            """########\n## ##c4#\n#  #5 7#\n#6  4 8#\n#756# 5#\n#54##8##\n########""",
            """########\n#7   c5#\n#6   54#\n#75 ##6#\n##7   5#\n### 4 6#\n###4####\n########""",
            """##########\n####4# c2#\n####3#  5#\n####7# 34#\n## 56# 43#\n## ##  65#\n#6  3  #2#\n#7  5  ###\n##  3   ##\n#  5#34 ##\n#3 ##4# ##\n##########""",
            """#######\n#4  c8#\n#3   4#\n#2   8#\n#1   ##\n##    #\n#  3 4#\n#  2 3#\n#  1 2#\n#  8 1#\n#  #48#\n#######""",
            """#########\n####3 c2#\n##  4  5#\n##  6  ##\n#5  # 4##\n#3 43 56#\n#42#5232#\n#########""",
            """##########\n#  ####c6#\n#6 6##861#\n##6## 71##\n#### 18###\n###  8####\n##1  ##8##\n# 7 ## 78#\n# 6####87#\n##########""",
            """#########\n####3c###\n##2#45###\n##3#1####\n##4#2#12#\n# 1#3#45#\n# # 2 23#\n#   #####\n#   1####\n#########""",
            """##########\n#######81#\n### 4c#12#\n##  1 #25#\n#   3 ##6#\n# 214  45#\n#8### ####\n##      ##\n#8      3#\n#186   1##\n#21#   3##\n##########""",
            """##########\n# #613c#1#\n#8#542 #5#\n#4#### #4#\n#5 4 1 #2#\n#3## 4 #3#\n#65  3  2#\n###  2 #1#\n###  # #8#\n##########""",
            """##########\n#4343582c#\n#######1 #\n#8  6  # #\n#7  7   5#\n#1  6  83#\n#8  7653##\n#212######\n###1######\n##########""",
            """##########\n####  ####\n##51 c7 ##\n# 76 16 8#\n#7##16##6#\n#6 8587 7#\n##15##87##\n##8#  #6##\n# ###### #\n##########""",
            """########\n###c6###\n### ####\n#6#7 7 #\n#7#6 86#\n#685 5##\n#### 8##\n#### ###\n########""",
            """##########\n####87 c7#\n####76  5#\n# 5#65  6#\n# 6#58  8#\n# 7#171 ##\n# ##### ##\n#  4    ##\n#  67 4 ##\n#  7# 6 ##\n#5 5#4#4##\n##########""",
            """##########\n#4#78145c#\n#3###### #\n#2    87 #\n#6168 7# #\n#52#7 #3 #\n#65#8 6# #\n##########""",
            """########\n#6#c87##\n#1# ####\n#2#  2##\n#1 8 1##\n#3 # 26#\n#7 #####\n#1 6 ###\n#7613###\n########""",
            """#########\n##   ####\n## 2c1###\n#8 1 382#\n## 3 231#\n## 1 ####\n## # ####\n#########""",
            """##########\n#45 c3   #\n#56  7   #\n#64  #   #\n###     8#\n###     1#\n###     8#\n###     1#\n#6#  3 12#\n#5#  # 27#\n#43 ##171#\n##########""",
            """#######\n###c2##\n#1# #3#\n#8#  4#\n#4  13#\n####31#\n### 2##\n### 8##\n#######"""
        ]
    
    def __str__(self):
        return str(self.state)

    def _apply_gravity_(self, state:PuzznicState):
        """!
        This function applies gravity to the boxes in the level, until they settle.

        A single top-down pass is not enough: a box resting on another box is processed
        before the one below it moves away, and would be left floating.
        """
        successor_state = PuzznicState(state.grid, state.cursor, state.score, state.cleared_boxes)
        moved = True
        while moved:
            moved = False
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
                        moved = True
        return successor_state

    def _check_and_remove_matches_(self, state:PuzznicState):
        """!
        This function checks and removes all horizontal/vertical matches of 2+ blocks.

        Returns the successor state and the set of boxes that were cleared.
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

        return matched_successor_state, to_remove

    def _compute_score_(self, removed_boxes):
        """!
        Scores the boxes cleared by a single match.

        Takes the cleared boxes rather than diffing two grids: boxes are identified by
        letter *and* position, so diffing counted a box that merely fell as cleared twice.
        """
        # scoring logic (assumed)
        # Each cleared block awards points (e.g., 10 points per block).
        # Consecutive matches caused by cascading blocks (due to gravity) increase a multiplier
        # Matching more than 2 blocks adds a bonus (e.g., +50 points per extra block).

        each_block_score    = len(removed_boxes) * 10
        cascaded_blocks     = set(map(lambda o:o.letter, removed_boxes))
        each_casecade_score = each_block_score * len(cascaded_blocks) * 1.5 if len(cascaded_blocks) > 1 else each_block_score

        more_than_two_blocks_score = 0
        letters_list = list(map(lambda o:o.letter, removed_boxes))
        for l in cascaded_blocks:
            if letters_list.count(l) > 2: more_than_two_blocks_score += 50
        return [each_casecade_score + more_than_two_blocks_score]

    def _commit_state_(self):
        self.state_history += [PuzznicState(self.state.grid, self.state.cursor, self.state.score, self.state.cleared_boxes)]
    
    def _compute_successor_state_(self, state:PuzznicState, action:str):
        # do not generate successors for goal/terminal states.
        successor_state = PuzznicState(state.grid, state.cursor, state.score, state.cleared_boxes)
        if state.is_goal() or state.is_terminal(): return successor_state
        successor_state.apply_action(action)
        # Gravity and matching cascade: clearing a match lets the boxes above fall, which can
        # form a new match, which clears, and so on. Repeat until the grid settles.
        while True:
            settled_state = self._apply_gravity_(successor_state)
            matched_state, removed_boxes = self._check_and_remove_matches_(settled_state)
            if not removed_boxes:
                # Nothing cleared, so nothing can fall: the grid has settled.
                successor_state = settled_state
                break
            matched_state.score += self._compute_score_(removed_boxes)
            successor_state = matched_state
        return successor_state

    def _levels_str_(self, index):
        assert 0 <= index < len(self.levelsstr), "Invalid level index."
        return self.levelsstr[index]

    def set_index(self, index:int):
        """Select a level. Out of range is refused here, not later.

        It used to be accepted and only rejected at `reset()`, by an assertion inside
        `_levels_str_`, which `python -O` strips, and which meant anything asking the
        environment how many levels it has by walking `set_index` upwards (the benchmark
        harness does exactly that) got told there was no limit.
        """
        if not 0 <= index < len(self.levelsstr):
            raise IndexError(
                f"Invalid index: {index}. There are {len(self.levelsstr)} levels, so the "
                f"index must be 0-{len(self.levelsstr) - 1}.")
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

