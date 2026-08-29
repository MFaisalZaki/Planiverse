"""Amazing Tater in pure Python: no ROM, no emulator, no dependencies.

The sibling [`amazing_tater_gb`](../gameboy/amazing_tater_gb.py) drives the real cartridge.
This one implements the rules directly, the way [`boxxle2`](boxxle2.py) stands beside
`boxxle2_gb`. Use this one for a dependency-free benchmark; use that one when you want the
cartridge's own transition function.

## The rules, stated

A tater (a potato with legs) walks the four directions one cell at a time and has to reach
the exit flag. Some rooms hold more than one tater; SELECT hands the controls to the next one,
and the room is not finished until every one of them has reached the flag.

1. **Walls** and the area outside the room refuse a step.
2. A **pit** refuses a step. Pits are crossed by filling them, not by walking over them.
3. A step into a **block** shoves the whole block one cell, and it moves only if every square
   it would land on is clear. Blocks come in every shape the cartridge felt like drawing
   (1x1, 1x5, 4x3, L-shapes), and a shape moves as one piece. Two different blocks may sit
   flush against each other and still be two blocks: which squares belong together is
   recorded per square, not inferred from what touches what.
4. A block square that comes to rest over a pit has **settled into it**. You cannot shove
   that square: a push has to be aimed at a square of the block that is standing on floor.
   The rest of the same block can still be pushed from a square that is.
5. When *every* square of a block sits over a pit, the block **dissolves**: the pits it
   covered become floor, permanently, and the block is gone. That is the only way to cross a
   pit, and it cannot be undone.
6. A step into a **turnstile arm** turns the whole turnstile 90 degrees, in whichever
   direction carries that arm the way you pushed. Pushing an arm along its own axis does
   nothing, and neither does pushing one that is hanging over a pit. The turn needs room:
   every square an arm lands on must be clear, and so must the diagonal each arm sweeps
   through on its way there. Arms may swing over pits; taters may not walk on them.
7. Where the pusher ends up depends on whether they are shut into a compartment. If another
   arm swings into the square you pushed, you were standing between two arms and the
   turnstile carries you round with it: your position rotates about the pivot, exactly like
   a revolving door. If nothing swings in behind it, the square you pushed is now empty and
   you simply step into it.
8. The **pivot** at the centre of a turnstile is solid and never moves.

The room is solved when every tater has reached the flag. Reaching it takes a tater off the
board, and the controls pass to whoever is left.

## Where this differs from the cartridge

Nowhere that has been found, and the search was not casual: the rules above were established
by walking this module and the cartridge forward in lockstep (the same random press, then a
cell-by-cell comparison of the board) across every one of the 105 rooms below.
`tests/test_amazing_tater.py` replays the stored solutions here and, when a ROM is available,
on the cartridge too.

Four of the eight rules are worded the way they are *because* that comparison rejected a
simpler guess: rule 3's "still be two blocks", rule 4, rule 6's swept diagonal, and rule 7's
compartment. Each of those was a real disagreement with the cartridge before it was a
sentence here.

What is missing is everything outside the room: the cartridge's move counter, its timer, the
pause menu behind A with its RETRY and QUIT, and the level counter that loads the next room
over the top of a cleared one. Here a solved room is simply terminal.

## Where the levels came from

All 105 were read out of `Amazing Tater (U).gb` by booting it and dumping the board the
cartridge itself composes in work RAM (`amazing_tater_gb.AmazingTaterGBEnv.levels`) at
matching indices, so `fix_index(7)` here and there are the same room. Nothing was transcribed
by hand.

They are the cartridge's two puzzle sets: 41 rooms behind PUZZLE MODE (`A-01` to `A-41`) and
64 behind BEGINNER and ACTION MODE (`C-01` to `C-64`). The third set on the cartridge, the 96
rooms behind PRACTICE MODE, is deliberately absent: that mode is a timed climb through ten
floors, its board buffer holds the corridors of the neighbouring floors as well as the room,
and the tater starts outside the room. It is a different game, not a different level.

## The alphabet the levels are written in

One character per cell, and one character per *cell code the cartridge uses*, so a stored
level and a board dumped out of the emulator are the same string. That is why blocks are
letters rather than a single `$`: the cartridge records for every block square which of its
neighbours belong to the same block, two different blocks are flush against each other in
half of these rooms, and a `$` for all of them would quietly weld them together.

    ' '  outside the room          '#'  wall                '.'  floor
    'O'  an open pit               'E'  the exit flag       '1'-'4'  the taters
    '@'  a turnstile pivot
    '^' '>' 'v' '<'   an arm, pointing the way it sticks out from its pivot
    'U' 'R' 'D' 'L'   the same four arms, hanging over a pit
    'a'-'p'           a block square on floor, one letter per set of neighbours it is
                      joined to: `a` is joined to nothing, `p` to all four
    'A'-'V' (see `SETTLED_GLYPHS`)   the same sixteen, for a square settled into a pit

`str(state)` prints a friendlier version of the same board (`$` for every block square, `+`
for every arm, `o` for every pivot), which is what the docs show. `board(level, state)` is
the exact one, and it is what the tests compare.
"""
from collections import deque

from planiverse.environments.base import Environment

# ------------------------------------------------------------------------- the alphabet

OUTSIDE, WALL, FLOOR, PIT, EXIT, PIVOT = " ", "#", ".", "O", "E", "@"

#: The four characters, in the order the cartridge numbers them.
TATER_GLYPHS = "1234"

#: A turnstile arm, by the direction it sticks out from its pivot: up, right, down, left.
ARM_GLYPHS = "^>v<"
ARM_OVER_PIT_GLYPHS = "URDL"

#: A block square, by which of its neighbours belong to the same block. The index into these
#: is a mask with 1 right, 2 down, 4 left and 8 up: the cartridge's own encoding.
BLOCK_GLYPHS = "abcdefghijklmnop"
#: The same sixteen for a square that has settled into a pit. Not a contiguous run because
#: `E`, `O` and the four `ARM_OVER_PIT_GLYPHS` are spoken for.
SETTLED_GLYPHS = "ABCFGHIJKMNPQSTV"

UP, RIGHT, DOWN, LEFT = (-1, 0), (0, 1), (1, 0), (0, -1)

#: Row and column deltas, in the order the cartridge's own arm codes number them.
DIRECTIONS = {"up": UP, "right": RIGHT, "down": DOWN, "left": LEFT}

#: Which bit of a turnstile's arm mask means an arm in which direction.
ARM_BIT = {UP: 8, RIGHT: 4, DOWN: 2, LEFT: 1}
BIT_ARM = {bit: offset for offset, bit in ARM_BIT.items()}

#: And which bit of a block square's mask means a neighbour in which direction. A different
#: numbering from the arms above, and that is the cartridge's doing, not a slip here.
BLOCK_BIT = {RIGHT: 1, DOWN: 2, LEFT: 4, UP: 8}

#: Handing the controls to the next tater. Not a direction, and free: it moves nobody.
SWITCH = "switch"

ACTIONS = tuple(DIRECTIONS) + (SWITCH,)

#: The glyphs `str(state)` prints instead, for people rather than for tests.
FRIENDLY = {glyph: "$" for glyph in BLOCK_GLYPHS}
FRIENDLY.update({glyph: "&" for glyph in SETTLED_GLYPHS})
FRIENDLY.update({glyph: "+" for glyph in ARM_GLYPHS})
FRIENDLY.update({glyph: "*" for glyph in ARM_OVER_PIT_GLYPHS})
FRIENDLY[PIVOT] = "o"


def rotate_cw(offset):
    """A cell offset, turned 90 degrees clockwise on screen."""
    return (offset[1], -offset[0])


def rotate_ccw(offset):
    return (-offset[1], offset[0])


#: Every room on the cartridge that is a room: 41 from PUZZLE MODE, then 64 from
#: BEGINNER / ACTION MODE. Dumped off the cartridge, not typed.
LEVELS = (
    # --------------------------------------------------------------- set A
    (   # A-01   15 x 5
        "      #####",
        " ### #.....# ###",
        "#...##@>.^.##...#",
        "#.E...v^<@....1.#",
        "#...##.@>..##...#",
        " ### #.....# ###",
        "      #####",
    ),
    (   # A-02   15 x 6
        "     #######",
        " ####.......####",
        "#...#..dg...#...#",
        "#...###jmdg.#...#",
        "#.E....dgjm...1.#",
        "#...###jm...#...#",
        "#...#.......#...#",
        " ### ####### ###",
    ),
    (   # A-03   16 x 6
        " ###### ##### ###",
        "#....OO#..^..#...#",
        "#...#OO..<@>.#...#",
        "#...#.....v....1.#",
        "#.E.#.^...dg.#...#",
        "#...#<@>..jm.#...#",
        " ####.v......####",
        "     ########",
    ),
    (   # A-04   10 x 7
        "     ######",
        "    #......#",
        " ####..be..#",
        "#EOOO..dg..#",
        "#.OOO..jm.1#",
        " ####.dhg..#",
        "    #.jnm..#",
        "    #......#",
        "     ######",
    ),
    (   # A-05   12 x 3
        " ############",
        "#......^.^..1#",
        "#.a.a.<@<@O##",
        "#.......Ov.OE#",
        " ############",
    ),
    (   # A-06   16 x 8
        "       ####",
        "      #....####",
        "      #..c.....#",
        " ###   #ck.dhhg##",
        "#...####kk.lppo.1#",
        "#.E.....ki.jnnm##",
        "#...####k......#",
        " ###   #i..####",
        "      #....#",
        "       ####",
    ),
    (   # A-07   13 x 5
        "     ###",
        "    #...##",
        " ####..a..####",
        "#...##O^a.#...#",
        "#.E...L@....1.#",
        "#...##.va.#...#",
        " ###  #### ###",
    ),
    (   # A-08   16 x 6
        "     ############",
        "    #............#",
        "    #....a...#.1.#",
        " ####........#...#",
        "#...#........####",
        "#.E.###<@<@>#",
        "#....O..v.v..#",
        " ############",
    ),
    (   # A-09   11 x 8
        "     #####",
        "    #.@...##",
        "   #.^v.^...#",
        "  ##<@.<@.^.#",
        " #...v..^<@.#",
        "#.@>.<@.@>##",
        "#.v.^.v^v.#",
        " ##<@><@>#",
        "  #.v.1v.E#",
        "   #######",
    ),
    (   # A-10   10 x 6
        "  #########",
        " #........1#",
        " #.a..^...#",
        " #<@><@<@.#",
        " #.v..v...#",
        " ######<@.#",
        "#E....O.v.#",
        " #########",
    ),
    (   # A-11   9 x 5
        "  ########",
        " #.......1#",
        "  #.#O^.a.#",
        "  #.^<@>..#",
        " #..@>..a.#",
        "#EO.v..#..#",
        " ###### ##",
    ),
    (   # A-12   10 x 10
        " ##########",
        "#..^..1..^.#",
        "#.a@.<@><@>#",
        "#..v..v..v.#",
        "#....^.^...#",
        " ##.<@.@>.#",
        "   ##......#",
        "    ##O####",
        "   #.....#",
        "   #..E..#",
        "   #.....#",
        "    #####",
    ),
    (   # A-13   8 x 10
        " ########",
        "#.^...^..#",
        "#.@>a<@..#",
        "#.v...v..#",
        "#.^bfe^..#",
        "#.@>.<@>.#",
        "#.vO.....#",
        " ##.####.#",
        "#...##...#",
        "#.E.##.1.#",
        "#...##...#",
        " ###  ###",
    ),
    (   # A-14   11 x 10
        " ###########",
        "#...........#",
        "#.dhgdhhg...#",
        "#.jnmjnnmdg.#",
        " ##1#####lo.#",
        "#.dg# # #lo.#",
        "#.lo##E##jm.#",
        "#.lo##.##4##",
        "#.jmdhhgdhg.#",
        "#...jnnmjnm.#",
        "#...........#",
        " ###########",
    ),
    (   # A-15   16 x 7
        "            ##",
        "         ###..#",
        " ###    #.be..###",
        "#...####...^..a..#",
        "#.E.OOOOOOO@>.a.1#",
        "#...####...v..a..#",
        " ###    #.be..###",
        "         ###..#",
        "            ##",
    ),
    (   # A-16   10 x 9
        "  #########",
        " #.........#",
        " #.E.###.^.#",
        " #...# #.@>#",
        "  ###  ##vO#",
        " ### ##O...#",
        "#...#.Oa<@>#",
        "#.1.a.O..v.#",
        "#.2.a.O#.@>#",
        "#...#.O..v.#",
        " ### ######",
    ),
    (   # A-17   14 x 11
        "  ###      ###",
        " #...#    #...#",
        "#....# ## #....#",
        "#..a.##.1#..a..#",
        "#.....#..#.....#",
        " ##..O^..^..###",
        "   ##<@..@>##",
        " ###..v..vO..##",
        "#.....#OO#.....#",
        "#..a..#OO##.a..#",
        "#....##.E##....#",
        " #...# ## #...#",
        "  ###      ###",
    ),
    (   # A-18   15 x 6
        " ### ####### ###",
        "#...#....be.#...#",
        "#.E....bebe...1.#",
        "#...#.^be...#...#",
        " ####<@><@>.####",
        "    #....v..#",
        "    #..be...#",
        "     #######",
    ),
    (   # A-19   16 x 7
        "    ###########",
        "   #...........#",
        " ###...ccOO....##",
        "#...OOOki.Odhg...#",
        "#.E.OOOk..Olpo.1.#",
        "#...OOOkc.Ojnm...#",
        " ###...iiOO....##",
        "   #...........#",
        "    ###########",
    ),
    (   # A-20   8 x 8
        " ########",
        "#........#",
        "#.ca^.^..#",
        "#.i.@<@>.#",
        "#.@>va^..#",
        "#.v.@>@>.#",
        "#...v....#",
        " ##.bffe.#",
        "#EO.O...1#",
        " ########",
    ),
    (   # A-21   8 x 14
        "     #",
        "    #E#",
        "   #OOO#",
        "   #OOO#",
        "  ##OOO#",
        " #......#",
        "#.@>..<@1#",
        "#.v....v.#",
        "#..bffe..#",
        "#.bfebfe.#",
        " #.^.....#",
        " #<@>####",
        "#.c..#",
        "#.i..#",
        "#....#",
        " ####",
    ),
    (   # A-22   11 x 12
        "     ###",
        " ####...####",
        "#...#.E.#...#",
        "#.1.#...#.2.#",
        "#...##.##...#",
        " ##...be..##",
        "   ##be.##",
        "   ##.ac#",
        "  #.be.i#",
        " ##.###.###",
        "#...# #....#",
        "#.3.#  #.4.#",
        "#...#  #...#",
        " ###    ###",
    ),
    (   # A-23   12 x 10
        "   ########",
        "  #E.......#",
        " ##.OObeOO.##",
        "#..##O..O##..#",
        "#....O..O....#",
        "#.bffe##bffe.#",
        "#.....##..c..#",
        "#..be.....i..#",
        "#..##....##..#",
        " ##..bffe..##",
        "  #.......1#",
        "   ########",
    ),
    (   # A-24   14 x 11
        "  ############",
        " #............#",
        " #@>.c....c.<@#",
        "#.v..i....i..v.#",
        "#..bffe..bffe..#",
        "#....c<@>.c....#",
        " #@>.i....i.<@#",
        " #v..^....^..v#",
        " #...@>.1<@...#",
        " #............#",
        "  ####OOOOOOOO#",
        " #EOOOOOOOOOOO#",
        "  ############",
    ),
    (   # A-25   16 x 10
        "     ###      ###",
        "    #...### ##...#",
        "    #......#...1.#",
        "    #...be...#...#",
        "    #.bea<@..####",
        "    #O^...vc.#",
        "    #L@<@.ai.#",
        " ####Ov.v.c..#",
        "#...#O#<@>i...#",
        "#.E.OOOODO....#",
        "#...#######...#",
        " ###       ###",
    ),
    (   # A-26   16 x 13
        " ################",
        "#...dhg....be^...#",
        "#.<@jnmdhg.^.@>..#",
        "#..v...lpo<@>vdgc#",
        "#dg.<@>jnm....jmi#",
        "#jm..bfe....dg...#",
        "#..^.^.dhg.^jm...#",
        "#E.@>@>lpo.@>...1#",
        "#..v.v.jnm.vdg...#",
        "#dg..bfe....jm...#",
        "#jm.<@>dhg....dgc#",
        "#..^...lpo<@>^jmi#",
        "#.<@dhgjnm.v.@>..#",
        "#...jnm....bev...#",
        " ################",
    ),
    (   # A-27   16 x 15
        "    ##      ##",
        "   #..#    #..#",
        "  #O.a.#  #....#",
        " #Ec##..##be##OO#",
        " ##K...OOOOa....#",
        "#1.O###.OO.###be.#",
        "#be#c..#O.#c..#OO#",
        "#..ck.O.O.akcOOOO#",
        "#OOikOO.OOOkkOOO.#",
        "#.beiOO....iibfe.#",
        "#.OO.a..#.bffebe.#",
        " #.bfebffHHGCOOO#",
        " #O...O#..#.iO.O#",
        "#.#.....##.acO.#.#",
        " #.#...^.^..iO#.#",
        "#.#.##.@.@..##.#.#",
        " # #  ######  # #",
    ),
    (   # A-28   11 x 8
        "   ####",
        "  #EOOO##",
        "  #OO..a.###",
        " ##OO....^..#",
        "#1.####.<@..#",
        "#...........#",
        "#.dgbfe....#",
        "#.jmbfe.###",
        "#.....##",
        " #####",
    ),
    (   # A-29   13 x 9
        " #############",
        "#.............#",
        "#..c...c...c..#",
        "#..kbfekbfek..#",
        "#a.i.c.i.c.i.a#",
        "#Ebfekbfekbfe1#",
        "#a.c.i.c.i.c.a#",
        "#..kbfekbfek..#",
        "#..i...i...i..#",
        "#.............#",
        " #############",
    ),
    (   # A-30   11 x 11
        "   #######",
        "  #.......#",
        " ##......a##",
        "#..OOOOOOO..#",
        "#..OcabecO..#",
        "#..Ok...kO..#",
        "#..Oi.1.kO..#",
        "#..Oa...iO..#",
        "#..OaabfeO..#",
        "#..OOOOOOOa.#",
        " ##OOO....##",
        "#EOOOO####",
        " #####",
    ),
    (   # A-31   10 x 7
        "  #",
        " #1#####",
        " #..^...###",
        "#..^@>...OE#",
        "#.<@.a@>.##",
        "#...a.v..#",
        " #.<@...#",
        " #..v..#",
        "  #####",
    ),
    (   # A-32   9 x 11
        " #########",
        "#....1....#",
        "#....dg...#",
        "#....jm...#",
        " #.bffffe.#",
        "#........#",
        "#..bffffe.#",
        "#........#",
        "#..bffffe.#",
        " #........#",
        "#..OOOO...#",
        "#..OOEO...#",
        " #########",
    ),
    (   # A-33   14 x 14
        "    ###########",
        "   #O^........1#",
        "   #c@>..ac....#",
        "   #k.^^.#k.OOO#",
        "   #k<@@>#kOOOO#",
        "  ##i....#iFIO.#",
        " #O.a....#.jQ^.#",
        " #.OOO...#...@>#",
        " #...OOO..^.O..#",
        " #O.OOOOOO@>be.#",
        " #O.OO..Oa...a.#",
        " #O.cO..O......#",
        "#OO#k.....#####",
        "#O.#i.@>..#",
        "#E.#.OD...#",
        " ## ######",
    ),
    (   # A-34   18 x 16
        "###################",
        "#1...a............a#",
        "#..bfffe.bfffe.....#",
        " #####.bfffebfffe.a#",
        "#.bfffe.bfffe......#",
        "#.bfffe.bfffe.bfffe#",
        "#....bfffe..bffe...#",
        "#a..be....bfffe....#",
        "#.bfffe.bffe......c#",
        "#...bfffe.bfffebfei#",
        " #####..be...####..#",
        "#.abfffe.bffffe.a..#",
        "#a......bea.bffe...#",
        "#bfe.bfffe.bffe..be#",
        "#.bfebfe.#....#####",
        "#....bfe...bfffe...#",
        "#...be..bfffe.....E#",
        " ##################",
    ),
    (   # A-35   18 x 16
        "###################",
        "#....a....a......a1#",
        " #.#a#.#.#c#a#.#.#c#",
        "#.a..be.a.ka..bfe.k#",
        " #.#.#.#a#i#.#.#.#k#",
        "#.c...be...bfffea.k#",
        " #i#c#.#a#.#.#.#.#k#",
        "#...k.c...a.c.a.a.k#",
        " #a#k#k#a.^.i#.#.#k#",
        "#.a.k.k..<@>..c.a.k#",
        " #.#k#k#..v..#i#.#k#",
        "#a.ak.kbfe.be...a.k#",
        " #.#i#k#.#.#.#.#.#k#",
        "#a...ai.a.c.a..a..k#",
        " #a#.#.#.#k#.#a#.#i#",
        "#..be...a.i.a..a.a.#",
        "#E.#.#.#.#.#.#.#.#.#",
        " ## # # # # # # # #",
    ),
    (   # A-36   8 x 8
        " ########",
        "#1.c....4#",
        "#.cia.a..#",
        "#.ka.a.a.#",
        "#.i######",
        "#...c....#",
        "#bfei.OOO#",
        "#.bfe.##O#",
        "#2....#EO#",
        " ##### ##",
    ),
    (   # A-37   18 x 13
        "   ##############",
        "  #.^.^......^.^.#",
        "  #<@<@>^..^<@>@>#",
        "  #.v...@><@...v.#",
        "  #.^...v..v...^.#",
        "  #<@<@.^..^.@>@>#",
        " ##.v.v<@><@>v.v.##",
        "#EO...O..........a1#",
        " ##.^.^<@><@>^.^.##",
        "  #<@<@.v..v.@>@>#",
        "  #.v...^..^...v.#",
        "  #.^...@><@...^.#",
        "  #<@<@>v..v<@>@>#",
        "  #.v.v......v.v.#",
        "   ##############",
    ),
    (   # A-38   18 x 9
        "      ### ###",
        "     #...#.^.##",
        "     #.....@...#",
        " ### #.O...v<@.####",
        "#...##OO@>...v.c...#",
        "#E..OOOOO.a#be.ic.1#",
        "#...##OO@>...^..i..#",
        " ### #O....^<@.####",
        "     #.....@...#",
        "     #...#.v.##",
        "      ### ###",
    ),
    (   # A-39   17 x 9
        "  ###############",
        " #OOOOOOO.....Oc.#",
        " #OOOOOO.O.O..Ok.#",
        " #OOOOOOO..cc.Oi.#",
        " #OOOOOO#O.ii.Oc.#",
        "#EO#OOOOOOabe.Ok.1#",
        " #OOOOOO#O.cc.Oi.#",
        " #OOOOOOO..ii.Oc.#",
        " #OOOOOO.O.O..Ok.#",
        " #OOOOOOO.....Oi.#",
        "  ###############",
    ),
    (   # A-40   18 x 16
        "###################",
        "#^........^....^...#",
        "#@>##.....@>.<@@>..#",
        "#...######vbfevbfe.#",
        "#..bfffe.@>..^.....#",
        "#.^...@>.vdg.@>###.#",
        "#.@>.^v..clobe.#.@>#",
        "#...<@>..klo.^.#.v.#",
        "#.a...^..klo.@.#.#.#",
        "#.bffe@.cijm#v.#.#.#",
        "#..^cav.k@>.####.#.#",
        " #<@k.beiv.......#.#",
        "#Edgk.@>.#########.#",
        " #jmi.v..#......@>.#",
        "#.bfe^be.#.bec..v.2#",
        "#..@.@>..^<@>i@>.13#",
        "#..v.v...@>...v..4.#",
        " ##################",
    ),
    (   # A-41   18 x 16
        "###################",
        "#.^......<@.....O4.#",
        "#<@dg...^.v.dhgOO32#",
        "#.alo..<@>OOlpoOO.1#",
        "#..jQOOOOOO#lpoO.^.#",
        " #..O^....O#lpoO.@>#",
        " #dgO@>.OO.#jnmO<@.#",
        "#.loO.^.##.#.<@O.v.#",
        "#.loO.@>#E.#.^vOO..#",
        "#.loO.v.####<@OOO..#",
        "#.jmO@>.^.^..vO#.c.#",
        "#.^.Ov.<@>@>a.##.k.#",
        "#<@>O.......OO#dgk.#",
        "#.v^OOOOOOO####jmi.#",
        "#..@>.<@>^O......^.#",
        "#..v.@>v.@.@>@>a.@>#",
        "#....v...v...v.bfe.#",
        " ##################",
    ),
    # --------------------------------------------------------------- set C
    (   # C-01   9 x 6
        "  ########",
        " #.......1#",
        "  ##.^....#",
        " #..<@.dg.#",
        " #.a.v.jm.#",
        " #OO......#",
        "#EOO.#....#",
        " #### ####",
    ),
    (   # C-02   16 x 3
        " ### ######## ###",
        "#...#.^...^..#...#",
        "#.E...@>#<@>...1.#",
        "#...#.v...v..#...#",
        " ### ######## ###",
    ),
    (   # C-03   16 x 4
        " ###          ###",
        "#...##########...#",
        "#.E....^^..^...1.#",
        "#...##<@@><@.#...#",
        " ### #.....v.####",
        "      #######",
    ),
    (   # C-04   18 x 3
        " ### ########## ###",
        "#...#.O^.^.^...#...#",
        "#.E....@.@.@>a...1.#",
        "#...#.OvO......#...#",
        " ### ########## ###",
    ),
    (   # C-05   15 x 7
        " ###   ###   ###",
        "#...###...###...#",
        "#...#.......#...#",
        "#...#..^.^..#...#",
        "#.E..O.@a@....1.#",
        "#...#..v.v..#...#",
        "#...#.......#...#",
        "#...###...###...#",
        " ###   ###   ###",
    ),
    (   # C-06   16 x 3
        " ### ######## ###",
        "#...#.^..@...#...#",
        "#.E...@>.v.^...1.#",
        "#...#.v...<@.#...#",
        " ### ######## ###",
    ),
    (   # C-07   14 x 4
        " ### ##  ## ###",
        "#...#..##..#...#",
        "#.E.#..^.......#",
        "#....O.@>a.#.1.#",
        "#...#..v...#...#",
        " ### ###### ###",
    ),
    (   # C-08   14 x 5
        " ### ###### ###",
        "#...#......#...#",
        "#....O.dhg.#...#",
        "#.E.#..jnm...1.#",
        "#...##...a.#...#",
        "#...##.....#...#",
        " ###  ##### ###",
    ),
    (   # C-09   7 x 6
        " ####",
        "#....#",
        "#....###",
        "#.....a1#",
        "#bffe#..#",
        " ##.....#",
        "#EO.....#",
        " #######",
    ),
    (   # C-10   7 x 8
        " #######",
        "#......1#",
        "#.a.a...#",
        " ##.####",
        "#...#...#",
        "#.be....#",
        "#.......#",
        " ###....#",
        "#EOOOO##",
        " #####",
    ),
    (   # C-11   16 x 3
        " ### ######## ###",
        "#...#O^......#...#",
        "#.E...@>@>.a...1.#",
        "#...##..v....#...#",
        " ###  ####### ###",
    ),
    (   # C-12   7 x 7
        "  #",
        " #E#",
        "#OO###",
        "#.OO..#",
        "#.#...#",
        "#...be.#",
        " ##..a.#",
        "  #....1#",
        "   #####",
    ),
    (   # C-13   14 x 5
        " ### ###### ###",
        "#...#......#...#",
        "#...#...dhg#...#",
        "#.E..<@.lpo..1.#",
        "#...#.v.jnm#...#",
        "#...#......#...#",
        " ### ###### ###",
    ),
    (   # C-14   9 x 6
        "      ##",
        "   ###..##",
        "  #.O...a1#",
        " ##dg...##",
        "#EOjm^.#",
        " ##..@>#",
        "  #....#",
        "   ####",
    ),
    (   # C-15   11 x 6
        " #    ##",
        "#E####..####",
        "#.#.....#...#",
        "#.#be.....a1#",
        "#.O..#be#...#",
        " ##.....####",
        "   ###..#",
        "      ##",
    ),
    (   # C-16   10 x 6
        " #####   #",
        "#.....###E#",
        "#.....#..O.#",
        "#..a..#....#",
        "#.....#...1#",
        " #...bfffe.#",
        "  #.......#",
        "   #######",
    ),
    (   # C-17   14 x 6
        "      ####",
        " ### #....# ###",
        "#...##.^...#...#",
        "#...##.@>..#...#",
        "#.E..OOvbe...1.#",
        "#...##.....#...#",
        "#...# #...##...#",
        " ###   ###  ###",
    ),
    (   # C-18   8 x 5
        " ########",
        "#...O....#",
        "#.<@>O...#",
        "#bevOO.a.#",
        "#....#.1.#",
        "#EOa.#...#",
        " #### ###",
    ),
    (   # C-19   7 x 13
        " #######",
        "#.......#",
        "#...1...#",
        "#.......#",
        " ####.##",
        "#...#...#",
        "#.bfe..a#",
        "#....dg.#",
        "#.dhgjm.#",
        "#.jnm...#",
        " ##.####",
        "#.......#",
        "#...E...#",
        "#.......#",
        " #######",
    ),
    (   # C-20   7 x 7
        " #######",
        "#...1...#",
        "#@>###.#",
        "#v.#E#a.#",
        "#.dhhg..#",
        "#.jnnm..#",
        "#.#....#",
        "#...###",
        " ###",
    ),
    (   # C-21   16 x 6
        " ###  ####### ###",
        "#...##O.^....#...#",
        "#.E.....@>.....1.#",
        "#...##O@><@>.#...#",
        " ### #.v^.v..####",
        "     #..@>...#",
        "     #..v....#",
        "      #######",
    ),
    (   # C-22   12 x 6
        " ############",
        "#...OOOO...^.#",
        "#.E.#.O....@>#",
        "#...#.dgbfev.#",
        " ### #jm^.^..#",
        "  ###..<@a@>.#",
        " #1..........#",
        "  ###########",
    ),
    (   # C-23   16 x 7
        "     ########",
        "    #........#",
        " ####OOOOabe.####",
        "#...#OUO.be..#...#",
        "#.E..<@OObe....1.#",
        "#...#ODO.be..#...#",
        " ####OOOOabe.####",
        "    #........#",
        "     ########",
    ),
    (   # C-24   15 x 7
        "     #######",
        "    #.......#",
        " ####.@><@be####",
        "#....^v..vc.....#",
        "#.E..@>...ka..1.#",
        "#....v^..^i.....#",
        " ####.@><@be####",
        "    #.......#",
        "     #######",
    ),
    (   # C-25   9 x 8
        "    ###",
        "   #...###",
        "   #......#",
        "   #.@<@>.#",
        "   #.v.a^.#",
        "   #.^..@>#",
        "   #<@1.v.#",
        " ###.v<@>.#",
        "#E.O...v..#",
        " #########",
    ),
    (   # C-26   13 x 6
        "    ###    ##",
        "   #...####.O#",
        "   #.^....a.O#",
        "   #.@<@.##.O#",
        " ###.v.v..c.a#",
        "#..#<@....i...#",
        "#E.OOv..##...1#",
        " #######  ####",
    ),
    (   # C-27   12 x 7
        "   #",
        "  #E#    ##",
        " ##O#####..##",
        "#............#",
        "#a.<@>@><@..#",
        "#...v.v..va.1#",
        " ##...####.##",
        "   ###   #.#",
        "          #",
    ),
    (   # C-28   13 x 8
        "      ###",
        "     #...#",
        "    #.....#",
        "    #.^be.#",
        "    #<@.^.#",
        " ####.a<@.####",
        "#...#@>.v.#...#",
        "#.E.Ovbfe...1.#",
        "#...##...##...#",
        " ###  ###  ###",
    ),
    (   # C-29   10 x 10
        " ##########",
        "#........a1#",
        "#aa##@>#.c.#",
        "#.c#.vbfek.#",
        "#.i.....ak.#",
        "#..#.####ia#",
        "#..##......#",
        "#.bffe..a.#",
        "#OO#.c.be#",
        " #<@.i..#",
        "#E.v...#",
        " ######",
    ),
    (   # C-30   11 x 10
        "   ######",
        "  #......#",
        " #...a..c.##",
        " #.dhga.k...#",
        " #1lpobei#..#",
        "  #jnm#....#",
        "   #..#...#",
        "  #..bfe.#",
        " #OO.....#",
        "#Ea.###..#",
        " #O.#  ##",
        "  ##",
    ),
    (   # C-31   10 x 10
        " ######",
        "#1.....#",
        "#.#.@>.##",
        "#...v....#",
        " ##......#",
        "  #a##c#.##",
        "  #..#k#...#",
        "  #..#k#...#",
        "  #..#i....#",
        "   ###.###.#",
        "  #EO......#",
        "   ########",
    ),
    (   # C-32   10 x 10
        "    #####",
        "   #1....#",
        "   #.###.#",
        "  ##a###.#",
        " #..^.....#",
        " #..@.be^..#",
        " #..v...@..#",
        "  #.....v..#",
        "   #.###a##",
        " ###.###.#",
        "#EOO.....#",
        " ########",
    ),
    (   # C-33   10 x 6
        "     #####",
        "    #.....#",
        " # #..#<@..#",
        "#1##....DOO#",
        "#..^bfe.##E#",
        "#.O@R#..# #",
        "#......#",
        " ######",
    ),
    (   # C-34   12 x 7
        "    #",
        "   #.#",
        "  #..########",
        " #.^O.^..^.a.#",
        "#E<@a<@.<@..1#",
        " #.vO.v..v.a.#",
        "  #..########",
        "   #.#",
        "    #",
    ),
    (   # C-35   10 x 9
        "   ###",
        "  #...# ###",
        "  #.a###..1#",
        "  #cO..^...#",
        " #.iO#.@>##",
        "#..## #....#",
        "#..OE# #.^.#",
        " #.#####<@.#",
        " #.........#",
        " #...######",
        "  ###",
    ),
    (   # C-36   10 x 10
        "     ##",
        " ####..####",
        "#.1.a......#",
        "#...#..#...#",
        "#...#..#...#",
        " ###..^.###",
        "#..<@<@....#",
        "#...v...dg.#",
        "#O#..<@.jm.#",
        "#..#..v..#.#",
        "#.E##..## #",
        " ##  ##",
    ),
    (   # C-37   8 x 5
        " #  ###",
        "#E##...##",
        "#O..@>a..#",
        "#..^v^...#",
        "#..@>@>.#",
        "#....be.1#",
        " ########",
    ),
    (   # C-38   9 x 8
        " #########",
        "#.^..^..^.#",
        "#.@..@..@.#",
        "#.v..v..v.#",
        "#.#..bfe..#",
        "#.#..#.#.1#",
        "#.#be#.aOO#",
        "#.#.##...#",
        "#...OE###",
        " #####",
    ),
    (   # C-39   13 x 8
        " #",
        "#E#",
        "#O##",
        "#..1#####",
        "#........#",
        "#.c####...##",
        "#.i...bfffe.##",
        "#..##.##....a.#",
        "#...a....##...#",
        " ########  ###",
    ),
    (   # C-40   10 x 8
        "  #### ####",
        " #....#....#",
        " #.abfe..c.#",
        " #..#.#.#k1#",
        "#E#......i#",
        "#O#c......#",
        "#O.k#.#.#..#",
        "#..i..bfea.#",
        " #....#....#",
        "  #### ####",
    ),
    (   # C-41   8 x 10
        " ########",
        "#.......1#",
        "#........#",
        "#<@><@>#.#",
        "#...a..#.#",
        "#..be..#.#",
        "#<@><@>#.#",
        "#.v..v..O#",
        "#......##",
        " ##OO##",
        "  #.E#",
        "   ##",
    ),
    (   # C-42   10 x 9
        "   ######",
        "  #.^..^1#",
        " #.<@a.@>.#",
        "#...v..v...#",
        "#^..#..#..^#",
        "#@>##be##<@#",
        "#...#..#...#",
        "#.O.^..^...#",
        " #EL@..@>.#",
        "  #.v..v.#",
        "   ######",
    ),
    (   # C-43   7 x 13
        " # #####",
        "#.#.....#",
        "#...c...#",
        "#.c.i<@R#",
        "#.i...v.#",
        "#Oa@>O^.#",
        "#..vO<@.#",
        "#a..O^..#",
        "#..O.@>1#",
        "#..##...#",
        "#..####O#",
        " ###...O#",
        "   #.E.#",
        "   #...#",
        "    ###",
    ),
    (   # C-44   11 x 8
        "   #### ###",
        "  #.^..#...#",
        "  #<@>...^a#",
        "  #.v.<@<@.#",
        " ##..^.v.v..#",
        "#EO..@>.^..^#",
        " ##..v.<@.<@#",
        "   ###..v.^.#",
        "      ##..@1#",
        "        ####",
    ),
    (   # C-45   15 x 8
        "     #####",
        " ####.....# ####",
        "#...##...# #....#",
        "#...O.<@>.#.....#",
        "#.E.#....^..#.1.#",
        "#...#.@>.@>.#...#",
        "#...#.^..v..#...#",
        " ####.@>.^a.####",
        "    #.v.<@..#",
        "     #######",
    ),
    (   # C-46   7 x 11
        "    #",
        " ###1###",
        "#O^O.O^O#",
        "#.@>.<@.#",
        "#.vbfev.#",
        "#.cOO.c.#",
        "#.k.a.k.#",
        "#.iOO.i.#",
        "#.^bfe^.#",
        "#.@>.<@.#",
        "#..O....#",
        " ##E####",
        "   #",
    ),
    (   # C-47   10 x 7
        "    #####",
        "   #.....##",
        "   #.<@>...#",
        " ##.^.v.^..#",
        "#E#.@>a<@dg#",
        "#O..v.^.vjm#",
        " ###.<@>..1#",
        "    #....##",
        "     ####",
    ),
    (   # C-48   18 x 8
        " ###            ###",
        "#...############...#",
        "#.E..<@....^.....1.#",
        "#...#.v<@..@<@>#...#",
        " ####..^v<@v...####",
        "    #.<@..v<@..#",
        "    #.^.^..^v..#",
        "    #.@>@><@.@>#",
        "    #.v........#",
        "     ##########",
    ),
    (   # C-49   14 x 12
        "            ###",
        "           #...#",
        "    ########.1.#",
        "   #...........#",
        "   #.<@dhgc.###",
        "   #..vjnmi.#",
        "   #.c<@>^..#",
        "   #.i.v.@..#",
        "    #<@>.va.#",
        " ###.OOOObe.#",
        "#....#OOO.a.#",
        "#.E.# ######",
        "#...#",
        " ###",
    ),
    (   # C-50   15 x 6
        "       ########",
        "   ####........#",
        " ##OO..<@>be....#",
        "#..OOO..v....dg.#",
        "#E.OOO..^bffejm1#",
        " ##OO..<@>......#",
        "   ####........#",
        "       ########",
    ),
    (   # C-51   16 x 9
        " #### #######",
        "#....#...O.c.#",
        "#.beO..<@C.i.#",
        " ###....vi...#",
        "    #......#.#",
        "    #...<@>..#",
        "    #.<@>a<@>#",
        "   #....<@....#",
        "   #.O...v<@..###",
        "   #1O.....v..OOE#",
        "    #############",
    ),
    (   # C-52   18 x 5
        "      ######",
        " ##  #...^..#######",
        "#..###..<@>.be..^..#",
        "#E.OO.O.....bea<@>1#",
        "#..###..<@>.be..v..#",
        " ##  #...v..#######",
        "      ######",
    ),
    (   # C-53   16 x 7
        "        ##",
        "   #####..#####",
        "  #O.....abe.@.#",
        " ##OO....^...v..#",
        "#E.O#O##.@>..Oa.1#",
        " ##OO....v...^..#",
        "  #O.....Obe.@.#",
        "   #####..#####",
        "        ##",
    ),
    (   # C-54   8 x 8
        " ########",
        "#...^....#",
        "#.a<@>.^.#",
        "#.a..a<@.#",
        "#..^be.v.#",
        "#.<@>OO^.#",
        "#1.a.c.@>#",
        " #<@>i.v.#",
        "#EOD.....#",
        " ########",
    ),
    (   # C-55   18 x 7
        "      ###",
        "    ##...##",
        "   #..a...a#      #",
        "  #.a..dhhg.#    #E#",
        " #.##c.lppo..#  #.O#",
        " #.c#i.lppo...##.c.#",
        "#..i...jnnm.bfffei.#",
        "#1.................#",
        " ##################",
    ),
    (   # C-56   8 x 10
        " #######",
        "#1......#",
        "#.##cbe.#",
        "#.##i.a.#",
        "#.#.bea.#",
        "#.......#",
        "#...###.#",
        "#....#...#",
        " ##OO#OO.#",
        "#OOOO.OO#",
        "#E######",
        " #",
    ),
    (   # C-57   10 x 12
        " ##########",
        "#....dg....#",
        "#....jm....#",
        "#..#a##a#..#",
        "#.##.dg.##.#",
        "#.#..jm..#.#",
        "#.#..a...#.#",
        "#.##....##.#",
        "#..######..#",
        "#1........O#",
        " #########O#",
        "         #O#",
        "         #E#",
        "          #",
    ),
    (   # C-58   11 x 7
        "      #####",
        "     #c....#",
        "  # #.i..a.#",
        " #E##..bffe#",
        "#OaO#.c..a.#",
        "#...#.k#...#",
        " #....ibfe.#",
        "  #...be...1#",
        "   #########",
    ),
    (   # C-59   9 x 8
        " #########",
        "#......OOE#",
        "#..bfe.###",
        "#...dg....#",
        "#adgjmdhg1#",
        "#.jm.alpo#",
        "#.adg.jnm#",
        "#..jmdg..#",
        "#....jm..#",
        " ########",
    ),
    (   # C-60   12 x 11
        "       ######",
        "      #.^.^.1#",
        " ######.@.@>.#",
        "#..dg...v.v..#",
        "#..jmdg.dhhg.#",
        "#.^.aloajnnm.#",
        "#c@>.jm..####",
        "#iv..be.#",
        "#...^..#",
        "#..<@.#",
        "#O.dg.#",
        "#EOjm.#",
        " #####",
    ),
    (   # C-61   8 x 8
        " ########",
        "#.....c..#",
        "#c.be.kc.#",
        "#kcc..kk.#",
        "#ikka1kk.#",
        "#.kk.akkc#",
        "#.kk..iik#",
        "#.ik.be.i#",
        "#EOK.....#",
        " ########",
    ),
    (   # C-62   10 x 10
        " ###",
        "#...#",
        "#.1..####",
        "#..@>....#",
        " #.v.be..#",
        "  #.c..c.#",
        "  #.i..i.#",
        "  #..be@>O#",
        "  #....vOOO#",
        "   ####OOOO#",
        "       #OOE#",
        "        ###",
    ),
    (   # C-63   13 x 13
        " #############",
        "#.............#",
        "#...c.........#",
        "#...k.1..dg...#",
        "#...i..O.jm...#",
        "#...OOOOOOO...#",
        "#...O.OaO.O...#",
        "#..OOOOOOOOO..#",
        "#..OOOOEOOOO..#",
        "#...O..O..O...#",
        "#...OO.a.OO...#",
        "#...OOOOOOO...#",
        "#......O......#",
        "#.............#",
        " #############",
    ),
    (   # C-64   8 x 14
        "  #####",
        " #.....#",
        "#..aaa.#",
        "#..a1a.#",
        "#..aaa.##",
        " #.......#",
        " #bfffe..#",
        "#......##",
        "#......#",
        "#......#",
        " #.....#",
        "  #....#",
        "   ##O#",
        "    #O#",
        "    #E#",
        "     #",
    ),
)

LEVEL_COUNT = len(LEVELS)

#: `(letter, the menu entry that reaches the set, how many rooms)`, in index order.
LEVEL_SETS = (("A", "PUZZLE MODE", 41), ("C", "BEGINNER / ACTION MODE", 64))


def label_for(index):
    """How the cartridge's own menus number a room, as `"set-number"` counting from one."""
    start = 0
    for letter, _mode, size in LEVEL_SETS:
        if index < start + size:
            return f"{letter}-{index - start + 1:02d}"
        start += size
    raise IndexError(f"Invalid index: {index}. There are {LEVEL_COUNT} rooms.")


# --------------------------------------------------------------------------- reading a room

def parse_level(rows):
    """One room's text into everything about it, static and moving.

    The two things that are recovered rather than read are a block's extent and a
    turnstile's arms. A block's squares each say which neighbours they are joined to, so the
    squares group into blocks by following those links and never by mere adjacency. A
    turnstile's arms each say which way they stick out, so an arm names its own pivot even
    where two pivots are close enough to share a neighbour, which happens in thirty-six
    places across these rooms, and is why the arms carry a direction at all.
    """
    height, width = len(rows), max(len(row) for row in rows)
    padded = [row.ljust(width) for row in rows]
    walls, pits, taters, masks = set(), set(), {}, {}
    pivots, exit_cell = set(), None
    for row, text in enumerate(padded):
        for col, glyph in enumerate(text):
            cell = (row, col)
            if glyph in (WALL, OUTSIDE):
                walls.add(cell)
            elif glyph == FLOOR:
                pass
            elif glyph == PIT:
                pits.add(cell)
            elif glyph == EXIT:
                exit_cell = cell
            elif glyph == PIVOT:
                pivots.add(cell)
            elif glyph in TATER_GLYPHS:
                taters[TATER_GLYPHS.index(glyph)] = cell
            elif glyph in BLOCK_GLYPHS:
                masks[cell] = BLOCK_GLYPHS.index(glyph)
            elif glyph in SETTLED_GLYPHS:
                masks[cell] = SETTLED_GLYPHS.index(glyph)
                pits.add(cell)
            elif glyph in ARM_GLYPHS:
                pass
            elif glyph in ARM_OVER_PIT_GLYPHS:
                pits.add(cell)
            else:
                raise ValueError(f"unknown glyph {glyph!r} at row {row}, column {col}")
    turnstiles = {pivot: 0 for pivot in pivots}
    for row, text in enumerate(padded):
        for col, glyph in enumerate(text):
            index = (ARM_GLYPHS.index(glyph) if glyph in ARM_GLYPHS
                     else ARM_OVER_PIT_GLYPHS.index(glyph)
                     if glyph in ARM_OVER_PIT_GLYPHS else None)
            if index is None:
                continue
            offset = (UP, RIGHT, DOWN, LEFT)[index]
            pivot = (row - offset[0], col - offset[1])
            if pivot not in turnstiles:
                raise ValueError(f"the arm at row {row}, column {col} has no pivot")
            turnstiles[pivot] |= ARM_BIT[offset]
    if any(mask == 0 for mask in turnstiles.values()):
        raise ValueError("a turnstile pivot has no arms")
    return (frozenset(walls), frozenset(pits), exit_cell, group_blocks(masks),
            turnstiles, taters, (height, width))


def group_blocks(masks):
    """Block squares into blocks, following each square's own record of what it is joined to.

    `masks` is `{cell: mask}`. Adjacency is deliberately not consulted: half the rooms here
    have two different blocks sitting flush against each other, and grouping by touch would
    weld them into one piece that the cartridge would never move as one.
    """
    remaining, blocks = dict(masks), []
    while remaining:
        start = next(iter(remaining))
        stack, group = [start], set()
        while stack:
            cell = stack.pop()
            if cell in group:
                continue
            group.add(cell)
            mask = remaining.pop(cell, masks.get(cell))
            for offset, bit in BLOCK_BIT.items():
                if mask & bit:
                    stack.append((cell[0] + offset[0], cell[1] + offset[1]))
        blocks.append(frozenset(group))
    return frozenset(blocks)


def block_mask(block, cell):
    """Which neighbours of `cell` are part of the same block, as the cartridge encodes it."""
    mask = 0
    for offset, bit in BLOCK_BIT.items():
        if (cell[0] + offset[0], cell[1] + offset[1]) in block:
            mask |= bit
    return mask


def arm_cells(turnstiles):
    """Every turnstile arm on the board, as `{cell: (pivot, offset from it)}`."""
    cells = {}
    for pivot, mask in turnstiles:
        for bit, offset in BIT_ARM.items():
            if mask & bit:
                cells[(pivot[0] + offset[0], pivot[1] + offset[1])] = (pivot, offset)
    return cells


class Level:
    """The part of a room that never moves, worked out once when it is loaded.

    Only the walls and the exit are truly fixed. The pits are here too because the *set of
    squares that were ever pits* is fixed; which of them a dissolved block has since filled
    is carried by the state, not by the level.
    """

    def __init__(self, index, rows):
        self.index = index
        self.rows = tuple(rows)
        (self.walls, self.pits, self.exit, self.start_blocks,
         start_turnstiles, self.start_taters, self.shape) = parse_level(rows)
        self.start_turnstiles = tuple(sorted(start_turnstiles.items()))
        if self.exit is None:
            raise ValueError(f"level {index} has no exit flag")
        if not self.start_taters:
            raise ValueError(f"level {index} has no tater")

    @property
    def label(self):
        return label_for(self.index)

    def inside(self, cell):
        height, width = self.shape
        return 0 <= cell[0] < height and 0 <= cell[1] < width


# ------------------------------------------------------------------------------- the rules
# Free functions over `(level, state)` rather than methods, so the two halves that were hard
# to get right (`push` and `turn`) can be tested on their own.

def is_pit(level, state, cell):
    """An open pit: one nothing has filled in. Filled ones are floor and stay floor."""
    return cell in level.pits and cell not in state.filled


def clear_for_object(level, state, cell, moving_block=None, turning_pivot=None):
    """Can a block square or a turnstile arm come to rest here?

    Pits do not stop either of them: a block settles into one and an arm swings over it.
    Everything else on the board does, including the exit flag, which is scenery a block
    cannot be shoved onto.
    """
    if not level.inside(cell) or cell in level.walls or cell == level.exit:
        return False
    if cell in state.taters_by_cell:
        return False
    block = state.block_of.get(cell)
    if block is not None and block != moving_block:
        return False
    arm = state.arms.get(cell)
    if arm is not None and arm[0] != turning_pivot:
        return False
    return cell not in state.pivots


def walkable(level, state, cell):
    """Can a tater stand here? Everything `clear_for_object` wants, and floor under it."""
    return clear_for_object(level, state, cell) and not is_pit(level, state, cell)


def push(level, state, block, target, offset):
    """Shove `block` one cell along `offset`. Returns `(blocks, filled)` or None if refused.

    Two things beyond "is there room". A square that has settled into a pit cannot be
    shoved: `target` is the square the tater's hands are on, and it has to be standing on
    floor.
    And a block whose every square ends up over a pit dissolves into it, filling those pits
    for good; that is the only thing in the game that changes the terrain.
    """
    if is_pit(level, state, target):
        return None
    moved = frozenset((row + offset[0], col + offset[1]) for row, col in block)
    for cell in moved - block:
        if not clear_for_object(level, state, cell, moving_block=block):
            return None
    blocks = set(state.blocks)
    blocks.discard(block)
    filled = state.filled
    if all(is_pit(level, state, cell) for cell in moved):
        filled = filled | moved                      # it dissolves, and those pits are gone
    else:
        blocks.add(moved)
    return frozenset(blocks), filled


def turn(level, state, pivot, target, offset, mover):
    """Turn the turnstile at `pivot` because `mover` walked into its arm at `target`.

    Returns `(turnstiles, destination)` or None if refused. `destination` is where the tater
    that pushed ends up, which is the subtle half: see rules 6 and 7 in the module docstring.
    """
    if is_pit(level, state, target):
        return None                     # an arm hanging over a pit is out of reach
    arm = (target[0] - pivot[0], target[1] - pivot[1])
    if rotate_cw(arm) == offset:
        forward, backward = rotate_cw, rotate_ccw
    elif rotate_ccw(arm) == offset:
        forward, backward = rotate_ccw, rotate_cw
    else:
        return None                                  # pushed along its own axis: no turn
    mask = dict(state.turnstiles)[pivot]
    slots = [BIT_ARM[bit] for bit in (8, 4, 2, 1) if mask & bit]
    occupied = {(pivot[0] + slot[0], pivot[1] + slot[1]) for slot in slots}
    for slot in slots:
        landing = forward(slot)
        cell = (pivot[0] + landing[0], pivot[1] + landing[1])
        if cell not in occupied and not clear_for_object(
                level, state, cell, turning_pivot=pivot):
            return None
        # An arm sweeps the diagonal between where it was and where it lands, and the
        # cartridge checks that corner too. The pusher is exempt: they are standing in the
        # compartment that is about to swing round with them.
        corner = (pivot[0] + slot[0] + landing[0], pivot[1] + slot[1] + landing[1])
        if corner != mover and not clear_for_object(
                level, state, corner, turning_pivot=pivot):
            return None
    landings = [forward(slot) for slot in slots]
    turned = 0
    for landing in landings:
        turned |= ARM_BIT[landing]
    if backward(arm) in slots:
        # Another arm swings into the square that was pushed, so the pusher was shut into a
        # compartment: the turnstile carries them round instead of letting them step in.
        carried = forward((mover[0] - pivot[0], mover[1] - pivot[1]))
        destination = (pivot[0] + carried[0], pivot[1] + carried[1])
    else:
        destination = target
    if destination in {(pivot[0] + l[0], pivot[1] + l[1]) for l in landings}:
        return None
    if not walkable(level, state, destination) and destination != target:
        return None
    if destination == target and (is_pit(level, state, target)
                                  or not level.inside(target) or target in level.walls):
        return None
    turnstiles = tuple(sorted((p, turned if p == pivot else m) for p, m in state.turnstiles))
    return turnstiles, destination


# --------------------------------------------------------------------------------- state

class AmazingTaterState:
    """A position: where the taters, blocks and turnstiles are, and which pits are gone.

    The level carries the walls, the exit and the set of squares that were ever pits, so none
    of that is repeated here. What is here is everything a move can change, which is more
    than in most of the twins in this package, because a turnstile's arms and a dissolved
    block's pits both belong to the position rather than to the board.
    """

    def __init__(self, level, taters, active, blocks, turnstiles, filled, home, depth=0):
        self.level = level
        self.taters = tuple(sorted(taters))           # (character, cell), still on the board
        self.active = active                          # which character has the controls
        self.blocks = frozenset(blocks)
        self.turnstiles = tuple(sorted(turnstiles))
        self.filled = frozenset(filled)
        self.home = frozenset(home)
        self.depth = depth

        self.taters_by_cell = {cell: who for who, cell in self.taters}
        self.block_of = {cell: block for block in self.blocks for cell in block}
        self.arms = arm_cells(self.turnstiles)
        self.pivots = {pivot for pivot, _ in self.turnstiles}
        self.solved = not self.taters

        literals = [f"taters-home({len(self.home)})"]
        literals += [f"at(tater{who + 1}, {cell[0]}, {cell[1]})" for who, cell in self.taters]
        literals += [f"home(tater{who + 1})" for who in sorted(self.home)]
        literals += [f"at(block, {row}, {col})" for row, col in sorted(self.block_of)]
        literals += [f"turnstile({p[0]}, {p[1]}, {m})" for p, m in self.turnstiles]
        literals += [f"pit({row}, {col})" for row, col in sorted(level.pits - self.filled)]
        if self.active is not None:
            literals.append(f"controlled(tater{self.active + 1})")
        if self.solved:
            literals.append("goal-reached")
        self.literals = frozenset(literals)

    # The position is what is on the board and who holds the controls. Depth and history are
    # not part of it, so a position reached two ways compares equal and search can close.
    def __key__(self):
        return (self.taters, self.active, self.blocks, self.turnstiles, self.filled)

    def __eq__(self, other):
        return (isinstance(other, AmazingTaterState) and self.level is other.level
                and self.__key__() == other.__key__())

    def __hash__(self):
        return hash(self.__key__())

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        return "\n".join(friendly(board(self.level, self)))

    def __repr__(self):
        return (f"<AmazingTaterState(home={len(self.home)}/"
                f"{len(self.home) + len(self.taters)}, depth={self.depth})>")


def initial_state(level):
    taters = tuple(sorted(level.start_taters.items()))
    return AmazingTaterState(level, taters, taters[0][0], level.start_blocks,
                             level.start_turnstiles, frozenset(), frozenset())


def advance(level, state, action):
    """One action from `state`. Returns the new state, or None when the game refuses it."""
    if state.solved:
        return None
    if action == SWITCH:
        if len(state.taters) < 2:
            return None
        order = [who for who, _ in state.taters]
        following = order[(order.index(state.active) + 1) % len(order)]
        return AmazingTaterState(level, state.taters, following, state.blocks,
                                 state.turnstiles, state.filled, state.home, state.depth + 1)
    offset = DIRECTIONS[action]
    here = dict(state.taters)[state.active]
    target = (here[0] + offset[0], here[1] + offset[1])
    taters = dict(state.taters)
    blocks, turnstiles, filled, home = (state.blocks, state.turnstiles,
                                        state.filled, state.home)

    if target == level.exit:
        del taters[state.active]
        home = home | {state.active}
        following = min(taters) if taters else None
    else:
        block = state.block_of.get(target)
        if block is not None:
            outcome = push(level, state, block, target, offset)
            if outcome is None:
                return None
            blocks, filled = outcome
            taters[state.active] = target
        elif target in state.arms:
            outcome = turn(level, state, state.arms[target][0], target, offset, here)
            if outcome is None:
                return None
            turnstiles, taters[state.active] = outcome
        elif target in state.pivots:
            return None
        elif walkable(level, state, target):
            taters[state.active] = target
        else:
            return None
        following = state.active
    return AmazingTaterState(level, tuple(taters.items()), following, blocks,
                             turnstiles, filled, home, state.depth + 1)


# -------------------------------------------------------------------------------- output

def board(level, state):
    """The position in the alphabet the levels are written in: the exact one.

    A board printed by this and a board dumped out of the emulator by
    `amazing_tater_gb.read_board` are the same tuple of strings, which is what lets the two
    be compared cell by cell rather than eyeballed.
    """
    height, width = level.shape
    lines = []
    for row in range(height):
        line = []
        for col in range(width):
            cell = (row, col)
            pit = is_pit(level, state, cell)
            if cell in level.walls:
                glyph = level.rows[row][col:col + 1] or OUTSIDE
                glyph = WALL if glyph == WALL else OUTSIDE
            elif cell in state.taters_by_cell:
                glyph = TATER_GLYPHS[state.taters_by_cell[cell]]
            elif cell == level.exit:
                glyph = EXIT
            elif cell in state.pivots:
                glyph = PIVOT
            elif cell in state.arms:
                index = (UP, RIGHT, DOWN, LEFT).index(state.arms[cell][1])
                glyph = (ARM_OVER_PIT_GLYPHS if pit else ARM_GLYPHS)[index]
            elif cell in state.block_of:
                mask = block_mask(state.block_of[cell], cell)
                glyph = (SETTLED_GLYPHS if pit else BLOCK_GLYPHS)[mask]
            else:
                glyph = PIT if pit else FLOOR
            line.append(glyph)
        lines.append("".join(line).rstrip())
    return tuple(lines)


def friendly(rows):
    """The same board with every block square a `$`, every arm a `+` and every pivot an `o`.

    For people. `board` is for tests, because collapsing the block letters is exactly the
    information that tells two blocks apart.
    """
    return tuple("".join(FRIENDLY.get(glyph, glyph) for glyph in row) for row in rows)


def render(level, state):
    return "\n".join(friendly(board(level, state)))


# --------------------------------------------------------------------------- environment

class AmazingTaterAction:
    """One press: a direction, or SELECT to hand the controls to the next tater."""

    #: What each press costs. SELECT moves nobody, so it is free; otherwise the cheapest
    #: plan for a two-tater room would be measured partly in how often you swapped.
    cost_map = {"up": 1, "right": 1, "down": 1, "left": 1, SWITCH: 0}

    def __init__(self, name):
        if name not in self.cost_map:
            raise ValueError(f"Unknown action: {name}. Choose from {sorted(self.cost_map)}.")
        self.name = name

    def cost(self):
        return self.cost_map[self.name]

    def __eq__(self, other):
        return isinstance(other, AmazingTaterAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<AmazingTaterAction({self.name})>"


class AmazingTaterGame(Environment):
    """Amazing Tater, implemented rather than emulated. Needs nothing installed."""

    def __init__(self):
        super().__init__("amazing_tater")
        self.index = 0
        self.level = None
        self.state = None
        self.state_history = []

    def fix_index(self, index):
        if not 0 <= index < LEVEL_COUNT:
            raise IndexError(
                f"Invalid index: {index}. There are {LEVEL_COUNT} rooms, so the index must "
                f"be 0-{LEVEL_COUNT - 1}.")
        self.index = index

    def label_for(self, index=None):
        return label_for(self.index if index is None else index)

    def reset(self):
        self.level = Level(self.index, LEVELS[self.index])
        self.state = initial_state(self.level)
        self.state_history = [self.state]
        height, width = self.level.shape
        return self.state, {"level_index": self.index,
                            "level": self.level.label,
                            "size": (width - 2, height - 2),
                            "taters": len(self.state.taters),
                            "blocks": len(self.state.blocks),
                            "turnstiles": len(self.state.turnstiles),
                            "pits": len(self.level.pits)}

    def is_goal(self, state):
        return state.solved

    def is_terminal(self, state):
        """A position with no move left in it, and no tater home.

        Sound but weak, and deliberately so. Amazing Tater has dead ends this does not catch
        (a block settled into the one pit that had to be crossed somewhere else is gone for
        good, and so is the room), but recognising those needs reachability under moving
        turnstiles, and a wrong `is_terminal` prunes a solvable branch, which is a much worse
        failure than missing a dead one.
        """
        return not state.solved and not self.successors(state)

    def successors(self, state):
        successors = []
        if state.solved:
            return successors
        for name in ACTIONS:
            following = advance(self.level, state, name)
            if following is None or following == state:
                continue
            successors.append((AmazingTaterAction(name), following))
        return successors

    def __advance__(self, state, action):
        name = action.name if isinstance(action, AmazingTaterAction) else str(action)
        following = advance(self.level, state, name)
        return state if following is None else following

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = len(self.state.home)
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, len(self.state.home) - before

    def get_actions(self):
        return [AmazingTaterAction(name) for name in ACTIONS]

    def render(self):
        rendered = [str(state) for state in self.state_history]
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered


def solve(index, limit=2_000_000):
    """Breadth-first search for a shortest plan, as a list of action names. None if none.

    Here rather than in the tests because it is how the stored solutions were found and how a
    new one can be checked; `tests/test_amazing_tater.py` replays what it produced.
    """
    game = AmazingTaterGame()
    game.fix_index(index)
    state, _ = game.reset()
    queue, seen = deque([(state, ())]), {state}
    while queue:
        state, plan = queue.popleft()
        if game.is_goal(state):
            return list(plan)
        if len(seen) > limit:
            return None
        for action, following in game.successors(state):
            if following in seen:
                continue
            seen.add(following)
            queue.append((following, plan + (action.name,)))
    return None
