"""Boxxle II in pure Python: no ROM, no emulator, no dependencies.

The sibling [`boxxle2_gb`](../gameboy/boxxle2_gb.py) drives the real cartridge. This one
implements the rules directly, the way [`flipull`](flipull.py) stands beside `flipull_gb`.
Use this one for a dependency-free benchmark; use that one for the cartridge's actual
behaviour.

## The rules, stated

Sokoban, and nothing more. A warehouse keeper walks the four directions one cell at a time:

1. A step into a **wall** is refused, and the position does not change.
2. A step into a **box** pushes it one cell in the same direction, but only if the cell
   *behind* the box is empty floor. A box cannot be pulled, and no chain of boxes can be
   pushed at once.
3. Any other step just moves the keeper.

The level is solved when every box stands on a goal square. Every level ships with exactly as
many goals as boxes, so "every box home" and "every goal filled" are the same sentence.

Boxxle II gives the player an undo button and a restart. Neither is here, and that is the
point: a Sokoban without undo is a search problem in which a wrong push is permanent, which
is what makes deadlock detection worth anything.

## Where this differs from the cartridge

Nowhere that has been found. This is unusual among the twins in this package and it has a
simple cause: the cartridge decompresses each level into three plain byte planes in work RAM
and the rules above are all it applies to them, so there is no animation, no timer, no
scoring and no randomness to reproduce. `tests/test_boxxle2.py` replays every stored solution
here *and* on `boxxle2_gb`, and the two agree move for move.

The one thing the cartridge has that this does not is the level counter: clearing a level
there loads the next one over the top of it. Here a solved position is simply terminal.

## Where the levels came from

All 120 of them were decoded out of `Boxxle II (USA, Europe).gb` by
`boxxle2_gb.read_levels`, at matching indices: `fix_index(7)` here and on `boxxle2_gb` are
the same board. Nothing was transcribed by hand, which is why none of them carries the kind
of quiet error the hand-typed Puzznic levels did; `tests/test_boxxle2.py` re-decodes the ROM
and compares, when a ROM is available, so a level cannot drift away from the cartridge
unnoticed.
"""
from collections import deque

from planiverse.environments.base import Environment

#: The alphabet the levels are written in, shared with `boxxle2_gb` so a board printed by
#: either module reads the same.
WALL, BOX, GOAL, BOX_ON_GOAL, PLAYER, PLAYER_ON_GOAL, FLOOR = "#", "$", "o", "*", "@", "+", " "

BOX_GLYPHS = (BOX, BOX_ON_GOAL)
GOAL_GLYPHS = (GOAL, BOX_ON_GOAL, PLAYER_ON_GOAL)
PLAYER_GLYPHS = (PLAYER, PLAYER_ON_GOAL)

#: Row and column deltas, in the order the Game Boy's own move table uses them.
DIRECTIONS = {"left": (0, -1), "up": (-1, 0), "down": (1, 0), "right": (0, 1)}

#: How the cartridge groups its levels: twelve stages of ten.
LEVELS_PER_STAGE = 10

#: The cartridge's 120 boards, decoded from the ROM at $4E18 and stored at matching
#: indices. See the module docstring.
LEVELS = (
    # 1-01
    " #######\n # @ooo#\n #   ####\n###$    #\n#   #$# #\n# $ #   #\n#   #####\n#####",
    # 1-02
    "  ####\n  #  #\n###  ####\n#  o*   #\n#@$oo $ #\n### $####\n  #  #\n  ####",
    # 1-03
    "#######\n## o @#\n# $ $ #\n#o * o#\n# $ $ #\n#  o ##\n#######",
    # 1-04
    "#########\n#       #\n# $o$o$ #\n# o$o$o #\n# $o*o$ #\n# o$o$o #\n# $o$o$ #\n#      @#\n#########",
    # 1-05
    "#########\n#       #\n#$$$ $$$#\n#ooooooo#\n#ooo#ooo#\n#$$$$$$$#\n#   @   #\n#########",
    # 1-06
    "#########\n#ooo##  #\n#o*o  $@#\n#  $#  ##\n### ## ###\n#   $    #\n#  $##$# #\n#    $  o#\n##########",
    # 1-07
    "#######\n#  o @#\n#o$*$ #\n# oo  #\n#$$*$$#\n#  oo #\n# $*$o#\n#  o  #\n#######",
    # 1-08
    " ##########\n## $  *o* #\n#  $  *o* #\n#  $  *o*@#\n#  $  *o* #\n#  $  *o* #\n###########",
    # 1-09
    "######   #####\n#    ### #  o#\n#  $ $ # #ooo#\n# #  $ ###  o#\n#  $$$   $ @o#\n###  $  $#  o#\n  #  $#$ #ooo#\n  ##     #  o#\n   ###########",
    # 1-10
    "      #########\n      #       #\n      # # # # #\n      #  $ $# #\n#######   $   #\n#oo#  ## $ $# #\n#oo   ## $ $  #\n#oo#  ## ######\n#oo# # $ $ #\n#oo     $  #\n#  ### @ ###\n#### #####",
    # 2-01
    " ######\n##  o@##\n# * #  ##\n#   #   #\n# $## # #\n#  $ $*o#\n#  o#   #\n#########",
    # 2-02
    "  ######\n###   o#\n#  $o#####\n# $$o#@  #\n#   ##$$ #\n#  o$o   #\n###$ ###o#\n  #o # ###\n  ####",
    # 2-03
    "  ####\n  #  ###########\n  #    $   $ $ #\n  # $# $ #  $  #\n  #  $ $  #    #\n### $# #  #### #\n#@#$ $ $  ##   #\n#    $ #$#   # #\n##  $    $ $ $ #\n #### ##########\n #      #\n #      #\n #oooooo#\n #oooooo#\n #oooooo#\n ########",
    # 2-04
    "########\n#   #  #\n# $    ####\n## ##$ #@ ######\n #  $ $##$  #  #\n # #  $ # $ ooo#\n # $ #$    #ooo#\n #     ## $#ooo#\n ### $ #  $ ooo#\n   ### #  $#ooo#\n    #  $$  #####\n    #     ##\n    #######",
    # 2-05
    "   ##########\n####oooooo  #\n#   ooooo#  #\n#  #oooooo ##\n## ####$##$#\n#@$  $ $   ###\n# $$    ##   #\n# #  $$##  # #\n#   $  # $$  #\n#  $  $   #$ #\n####   # $ $ #\n   #   #     #\n   ###########",
    # 2-06
    "          ######\n   ########    #\n   #   $   $ $ #\n   # # # # $## #\n   #     $   $ #\n   ########$ # #\n  ##      #    #\n  #@$  $  # $  #\n###### #$ #$#$ #\n#oooo# $  #    #\n#o*oo### ## ####\n#oo*o $   #  #\n#oooo$ $     #\n#ooo##########\n#####",
    # 2-07
    "       #######\n    ####ooooo#\n#####   ooooo#\n#   #  #o ooo#\n# $@## ### ###\n#  $   $  $$ #\n#   #$   $   #\n# $ # ##$  $ #\n###    $ ##  #\n  # $ $ $   ##\n  ###    ####\n    ######",
    # 2-08
    "      #########\n      #       #\n      #$ $$$  #\n      #   # $ #\n      #   $@$ #\n    ###$ $ # ##\n    #   $#$# #\n##### #  #   #\n#ooo  # $# ###\n#ooooo   # #\n#ooooo# $# #\n########   #\n       #####",
    # 2-09
    "     #####\n   ###   #\n   #   # #\n ###  $  ######\n #  #$#  #    #\n #$ $ ##   $  #\n##  #    $$$###\n#    ###$   #\n# $#$$ooo####\n#    #oooo#\n# # @ #ooo#\n#     $ooo#\n##  #     #\n ##########",
    # 2-10
    "    #####\n#####   ######\n#     #  ##  ##\n# # $$   $    #\n#  #   ## # # #\n# $ # ##   $  #\n#    $ $   # ##\n## $ #  #$## #\n ###  # # # $#\n   # $  @ ooo#\n   # $##$#ooo#\n   ##   oooo##\n    ### ooo##\n      ######",
    # 3-01
    "         #####\n        #     #\n       #  $##  #\n      #  #   $ #\n#  #  # #  # # #\n # #  # $ $  $ #\n  ####### #$#  #\n  #  $ $      #\n  #@oo$ **o###\n   #oooooo#####\n    ############",
    # 3-02
    "  ##########\n  #    #   #\n  # $#   $ #\n  # $ $#$# #\n ###o#ooo  #####\n##  oooo#$ #   #\n# $#o##*#  $$$ #\n#   ooooo@# $  #\n#  #o#ooo  $ $ #\n# $$  $#$#  $  #\n#   #  #  #   ##\n######    #####\n     ######",
    # 3-03
    "   #####\n   # @ #\n   #$$$#\n####   #\n#   o#$##\n# $o$o o#\n#  #o#o##\n########",
    # 3-04
    "############\n#ooo #     #\n#oo  # ##  #\n#oo     #  #\n#oo  # $## #\n#ooo #$ $  #\n######  $$ #\n ##  $ $$  #\n #@ $$$  # #\n ## $ ##   #\n  #        #\n  ##########",
    # 3-05
    "######\n#    #\n# $  ####\n# $*oo* #\n# *oo*$ #\n####  $ #\n   # @  #\n   ######",
    # 3-06
    "   #####\n####o  ##\n# $o$o  #\n#@$# #$ #\n# $o o  #\n####$#$ #\n  #o o  #\n  #######",
    # 3-07
    "##########\n##       #\n#   #$#$ #\n# $$  o$o#\n# @###ooo#\n##########",
    # 3-08
    " ######\n##    #\n# * $ #\n#o$$o##\n#o @ o#\n##o$$o#\n# $ * #\n#    ##\n######",
    # 3-09
    " #####\n #   ####\n## #$   #\n# $  $$ #\n# #$#o*o#\n#   @ooo#\n#########",
    # 3-10
    "#######\n#     #####\n# $ o*o $ #\n#@$o* *o$ #\n# $ o*o $ #\n######    #\n     ######",
    # 4-01
    "####\n#  ###\n# $$o#\n#@ $o#\n# $ o#\n# # o#\n######",
    # 4-02
    " ###### #####\n##@#  ###   ##\n#    $  # $  #\n#  $  #    $ #\n### ######   #\n### ##  ##$###\n# $  #### oo#\n# $ $ $  ooo#\n#    ####ooo#\n# $$ #  #ooo#\n#  ###  #####\n####",
    # 4-03
    "    ####\n#####  #\n#  $ $ # #######\n#   $  # #*o*o*#\n## $ $ ###o*o*o#\n #$ $  #  *o*o*#\n #@$ $    o*o*##\n #$ $  #  *o*o*#\n## $ $ ###o*o*o#\n#   $  # #*o*o*#\n#  $ $ # #######\n#####  #\n    ####",
    # 4-04
    "     #####\n     #@  ####\n  #####$    #\n  #ooo#  # $##\n  #o*o#$ #$  #\n###ooo $ # # #\n#  oo##$#    #\n#  ####  # $ #\n##   $   #$$ #\n #  ## ##   ##\n #####      #\n     #  #####\n     ####",
    # 4-05
    "   ######\n   #    ##\n   # $ $ ##\n#### #$#  ####\n# $    $  #  ###\n#    $ # $ooooo#\n### # ####ooooo#\n ## #$ $  oo####\n #  #   $#  #\n #@$  $  ####\n #  ######\n ####",
    # 4-06
    "#########\n#       ##\n# #$#$#@ #####\n# $    $ #   #\n## #### ## # #\n#  #  #      #\n# $#  $  #####\n#oo$  ## #   #\n#oo#  #   $$ #\n#oo####  $   #\n#*ooo# $ $ $ #\n#oooo#       #\n##############",
    # 4-07
    "   #######\n   #     ######\n   #  $#      #\n####$  $  #$$ #\n#    $$ $   $ #\n# $#  ## ### ##\n# @##$ $  $  #\n#     $  $ $ #\n##########$$ #\n  #ooooooooo##\n  #ooooooooo#\n  ###########",
    # 4-08
    "    ######\n  ###    ###\n  #   #$   ###\n  #   $   $$ #\n  # $$ #$    #\n  ##   $   $ #\n###### #$#####\n#oo@ #$  #\n#o#oo  $##\n#oooo$# #\n#oooo   #\n#########",
    # 4-09
    "###############\n#      #      #\n# $ #$ # $##$ #\n# #  $ #      #\n#   ##$#$##$$ #\n# # # ooo #   #\n# $  o # o$ # #\n# $#@$ooo# #  #\n#    o # o  $ #\n# ##o$###$o # #\n# # $ooooo ## #\n#             #\n###############",
    # 4-10
    " #########\n #  #    #######\n #    $   #    #\n##o#### $ $@$# #\n#oooo###  ## $ #\n#oooo####$ #   #\n#oooo        $ #\n### ####$# $ $ #\n#        ##  ###\n# $$ $ #######\n#  #####\n####",
    # 5-01
    "########\n#      #\n# $$   ###\n#  $ $$$ #####\n## ## ooo    ##\n # #@#ooo###$ #\n # # $ooo     #\n## # $ooo$ # ##\n#  ##### ### #\n#      $   $ #\n###########  #\n          ####",
    # 5-02
    " #### #####\n #  ###   ##\n # $  # #  ##\n # $# # ##  ##\n #   $  # $  ##\n # $  $ #$ $  #\n### # ### # $ #\n#     #oooo#  #\n# # $$ oooo## #\n# #   #oooo   #\n#  ##$#oooo#@##\n# $ $   # ####\n##    # $ #\n ###### $ #\n      ##  #\n       ####",
    # 5-03
    "  #####\n###   ##\n#      #\n# $$ #$########\n## #$ $ $ #   #\n#oooo#   $    #\n#oooo@#  ###$##\n#oooo$#   $ $ #\n#oooo$ $#     #\n#####   # $#$##\n    #####   $ #\n        ###   #\n          #####",
    # 5-04
    " ##### ######\n #   ###    #\n## $ $ #$ #$#\n#  $ @ $  $ ##\n# #  ## #oooo#\n#  ## $ #o##o#\n##  $    oooo#\n # $$ #$#oooo#\n #   #   #$ ##\n ##### $    #\n     ####  ##\n        ####",
    # 5-05
    "  ########\n  #   #  ###\n### # $  $ ###\n#    #$$ #   #\n# ##$#  @  $ #\n#    $$###$ ##\n#   $# ##  ##\n# #$ # ooo##\n# $ #ooooo#\n##  ooooo##\n # $# ####\n ##   #\n  #####",
    # 5-06
    " #####\n##   ##\n#     #####\n# #    $  #\n#  # $ ## #\n## $$#ooo #\n ###  ooo #\n   ####oo#####\n######ooo##  #\n#  ### $@ #$ #\n# $ #   #    #\n#  $  #$  $ ##\n##    $  ####\n #####  ##\n     ####",
    # 5-07
    "#########\n#       #\n#  $ $ $#\n## #$## #\n # oo oo##\n ##oo oo #\n  # ##$# ##\n  #$ $ $  #\n  #      @#\n  #########",
    # 5-08
    "      ####\n      #  ######\n      #    #  #\n      # $$    #\n#######$#  #  #\n#  #o*oo ###$##\n#  #o#*o$     #\n#  #o#o*# #   #\n# $$oooo# #####\n# @$ # ## #\n# $$$#    #\n#    ######\n######",
    # 5-09
    "#######\n#     #\n# $ $$#\n#@ $  #\n# $ $ #####\n# $  *oooo#\n#####o###o#\n   ##o# #o#\n   # o###o#####\n   # oooo*  $ #\n   #   ## $ $ #\n   ######  $  #\n        #$$ $ #\n        #     #\n        #######",
    # 5-10
    "########\n#oooooo#\n#  $ # ##\n# $ # $ #\n##$ $ $ #\n #  @   #\n ########",
    # 6-01
    "    ######\n ####    ######\n #  # $    #  #\n # $  #$$  $$ #\n #  $#  #oo#o #\n##$  # #ooo*o##\n# $ # $ oo##o #\n# #$ $ #ooooo #\n#    #  #  ####\n# $#  #   @##\n#    $# #  #\n####  $ ####\n   #  ###\n   ####",
    # 6-02
    "     ######\n #####    #\n #  #  $ $#\n #  $ # $ ###\n##  #  #  $ ##\n# $#  @ #    #\n#    $ $# $$ #\n#  $ #$o*o ###\n##  # $ooo##\n #### #oooo#\n    #  oooo#\n    ########",
    # 6-03
    "#######\n#o o o#\n# $$$ #\n#o$@$o#\n# $$$ #\n#o o o#\n#######",
    # 6-04
    "##############\n#       ###  #\n# $$ $    $$ #\n#     $ ###  #\n##$#####  # ##\n## # ooooo#  #\n## # ooo#o#  #\n#   #ooooo#$ #\n#   #o*ooo$@##\n# $ #  ### ###\n# $ ### $ $  #\n# $$$     $$ #\n#   ###      #\n##############",
    # 6-05
    "   ######\n####o  @#\n#  $$$  #\n#o##o##o#\n#   $   #\n#  $o# ##\n####   #\n   #####",
    # 6-06
    "  ##########\n###   o    #\n#   ##$##  #\n# @$o o o$##\n## $##$## #\n #    o   #\n ##########",
    # 6-07
    " ######\n #o oo#\n #o $o#\n###  $##\n# $  $ #\n# #$## #\n#   @  #\n########",
    # 6-08
    "   ########\n####    o #\n#  $ $ $o #\n#  o####o##\n# $o$ $ @#\n#  o  ####\n#######",
    # 6-09
    "  #########\n###   #   #\n#  $o * $ #\n# #o#o#o# #\n# $ $@$ $ #\n# #o#o#o# #\n# $ * o$  #\n#   #######\n#####",
    # 6-10
    "   ######\n#####   #\n# $  o  #\n# $ oo###\n#@$ $$###\n### o$o ###\n ##$ooo$  #\n # $o##   #\n #   ######\n #####",
    # 7-01
    "      #\n      #\n     ###\n   ##   ##\n  #o$     #\n #o $$$$$  #\n##  #   #  ##\n#  #*#@#*#  #\n#   #   #   #\n## o$$$$$o ##\n ##ooooooo##\n  #########",
    # 7-02
    "        ####\n  #######  ##\n  #    ##   #\n  # #   $o# ##\n  #   $ #o @ #\n  ##  ##ooo  #\n   ##$ $ooo ##\n  ##  ##o*o##\n### $    ###\n# $$ #$#$##\n# #     $ #\n#    ##   #\n###  ######\n  ####",
    # 7-03
    " #######\n##  *  ##\n# o o o #\n# $ *   #\n#*$$*$$*#\n#   * $ #\n# o o@o #\n##  *  ##\n #######",
    # 7-04
    "#######\n# @   #\n#  o$ #\n# o$o #\n#*$o$*#\n# o$o #\n# $o$ #\n#     #\n#######",
    # 7-05
    " ######\n##    #\n#  ##$#\n#  oo ##\n# $**$ #\n## oo  #\n #$##  #\n # @  ##\n ######",
    # 7-06
    "###############\n#             #\n# $o$o$o$o$o# #\n# o$o$o$o$o$  #\n# $o$o$o$o$o#@#\n# o$o$o$o$o$  #\n# $o$o$o$o$o# #\n#             #\n###############",
    # 7-07
    "#######\n#  * @#\n# $*$ #\n#  o  #\n# $*$ #\n#$o*o #\n#o o$o#\n# $o  #\n#######",
    # 7-08
    " #########\n #       ##\n #$ ####  #\n## # o* # #\n#   $*o # #\n# #$@o* # #\n#    $    #\n####$##$ ##\n   # ooo #\n   #######",
    # 7-09
    "#######\n# $o @#\n#  o  #\n#$$*$ #\n#oo*oo#\n# $*$ #\n# $o$ #\n#  o  #\n#######",
    # 7-10
    "     #####\n######ooo#\n#   $@ $##\n#  $ # $ #\n#o $ #$  #\n# #### #o#\n#o     $o#\n##########",
    # 8-01
    "    ######\n#####    ######\n#   #   @  #  #\n#  $o******o$ #\n####    #  #  #\n   ##   #    ##\n    ##########",
    # 8-02
    "       #\n      ###\n     # @##\n    ## $  #\n   #  $o$  #\n  #  $o*o$ ##\n ## $o*o*o$  #\n## $o*o*o*o$ ##\n #  $o*o*o$ ##\n  ## $o*o$  #\n   #  $o$  #\n    #  $ ##\n     ##  #\n      ###\n       #",
    # 8-03
    "#######\n#  o @#\n# $o$ #\n# $o$ #\n#*ooo*#\n# $*$ #\n#  *  #\n#  *  #\n#######",
    # 8-04
    "#########\n##     ##\n# #o$ $ #\n# o*o$  #\n# $o#o$ #\n#  $o#o #\n# $ $o# #\n##  @  ##\n#########",
    # 8-05
    "#######\n#  o @#\n#$$*$ #\n#o$o$o#\n# ooo #\n# $o$ #\n# $*$ #\n#  o  #\n#######",
    # 8-06
    "###############\n#  ooooooooo  #\n# $$$$$$$$$o$ #\n# $ooooooo$o$ #\n# $o$$$$$o$o$ #\n# $o$ooo$o$o$ #\n# $o$o$*$o$o$@#\n# $o$o$ooo$o$ #\n# $o$o$$$$$o$ #\n# $o$ooooooo$ #\n# $o$$$$$$$$$ #\n#  ooooooooo  #\n###############",
    # 8-07
    "  #####\n###   ####\n#  * * o #\n# * * *  #\n#  * *   #\n#  ###$ ##\n##  @#  #\n ########",
    # 8-08
    "#######\n#  o @#\n# $o$ #\n# *o$ #\n#o$*$o#\n# $o* #\n# $o$ #\n#  o  #\n#######",
    # 8-09
    "##########\n#        ##\n# $#$#$#$@#\n#o o$ooo$o#\n#o o$ooo$o#\n# $#$#$#$##\n#        #\n##########",
    # 8-10
    "##########\n#     #  #\n#  $ $#  ##\n#  ##o#   #\n##$##o##$ #\n# $ooooo  #\n#@$##o##$##\n## ##o#   #\n#   $ $   #\n#   #   ###\n#########",
    # 9-01
    " #######\n #     #   #####\n #  ## #####   #\n #$ $ $  #   # #\n # $ $ #       #\n##  $$ ###$### #\n# $    #oooo#  #\n#@#$$ ##oooo#  #\n#  $   #oooo$  #\n##  ## #oooo# ##\n #$$#  ## ###  #\n #      $      #\n ############  #\n            ####",
    # 9-02
    "        #######\n    #####     #\n #### $ $ $ $ #\n #  #  $ ###  #\n #    @    #$ #\n # ### ###    #\n # #ooooo# $$ #\n # #ooooo#  $##\n##$#ooooo$   #\n#    ## #  ###\n# $ $   $$ #\n##  #####  #\n ####   ####",
    # 9-03
    "   ####\n####  ######\n#     ##   #\n# $# $    $#\n## ### $#  #\n # $ $  ## #\n #   $ ##  ##\n #$ #ooo* $ #\n # @# oooooo#\n ############",
    # 9-04
    "###########\n#   ##oooo#####\n#  $  ooo##   #\n###$$ oo### # #\n  # ##ooo##$  #\n  # #   ###  ##\n ## # $  $ $ #\n #  # # #    #\n #    # $ #$ #\n # $$ # $#  ##\n ### ##  # ##\n   #@      #\n   #########",
    # 9-05
    "  #########\n  #    #  #\n### #$  $ ####\n#  $  ##oo#  ###\n# #  $ #oo$ $  #\n# $$  $#oo  #  #\n##  #  ooo#$   #\n #$ @ #ooo# $ ##\n #    #ooo$  ##\n # ##$ ###   #\n #   $     ###\n #  ########\n ####",
    # 9-06
    "     #####\n #####   #\n # oo $# #\n # #o*   #\n## *o#$ ##\n# $  $  #\n#   ## @#\n#########",
    # 9-07
    "##### #######\n#   ###  #  #\n# $     $ @ #\n## #$##o##  #\n #  ooo*o $ #\n # $# #o# # #\n ##    $    #\n  #  ########\n  ####",
    # 9-08
    " ############\n #    o #   #\n #   $* # $ #\n #   #o # ###\n #  ##o#   #\n##  ##o# $ ##\n#   # o#$#  #\n# @ # o   $ #\n#  ## *$$#  #\n#   # o    ##\n#   ########\n#####",
    # 9-09
    "    #####\n#####   #\n#   $ @ #\n#  $ #o#####\n##$ ##o##  ####\n #  ooooo $#  #\n # $##o##  #$ #\n #   #o##     #\n ### $ ##### ##\n   # #$     $ #\n   #    ###   #\n   ###### #####",
    # 9-10
    "####\n#  #\n#  ##########\n#    ##     #\n#oo#    $$# #\n#oo  ##   $ ###\n#oo#  ##$# $  #\n#oo   # @$ $  #\n#oo#  # $ $   #\n# o   # $ $ ###\n#  #  #   ###\n#  #    ###\n#########",
    # 10-01
    "############\n#    ooo $ #\n# $$$*** $@#\n#    ooo $ #\n############",
    # 10-02
    "#####\n#ooo# #####\n#ooo###   #\n#oooo   $$#####\n#oooo  #  #   ##\n#oo#$#### #$#  #\n## $  #     $$ #\n#  $# @ $ $$#  #\n# $ $ $ #   $ ##\n#   #  $ ##   #\n######   ######\n     #####",
    # 10-03
    "  #####\n  #   #\n###$o$#####\n#   o $   #\n# ##$## @ #\n#   o #####\n### o #\n  #   #\n  #####",
    # 10-04
    "     #######\n     #  #  #\n     #  $$ #\n###### $#  #\n#ooo### #  ##\n#o  #  $ #  #\n#o    $ $ $ #\n#o  #  $ #  #\n#ooo### #  ##\n###### $   #\n     #@ #  #\n     #######",
    # 10-05
    "    #####\n    #   #\n #### # #####\n##   $ $    #\n# $# #ooo#$ #\n#@$  ooo#  ##\n# $# oo# $  #\n##   o#     #\n ### $ $ #  #\n   ###   ####\n     #####",
    # 10-06
    "  #########\n  #       #\n  #$ $ $ $#\n #  $ $ $  #\n# $  # #  $ #\n#   #ooo#   #\n # #ooooo# #\n # #ooooo# #\n #  #o#o#  #\n #$ $ $$$ $#\n #  #   #  #\n  ### @####\n    ####",
    # 10-07
    "             #\n            ##\n           ###\n          #   #\n   ######## # #\n  # $ $ $ $   #\n ## #o#o#o#@$#\n###ooooooo   ##\n ## # # # #$##\n  # $ $ $ $   #\n   ######## # #\n          #   #\n           ###\n            ##\n             #",
    # 10-08
    "       #\n     #####\n   ### @ ###\n   #  $ $  #\n   # *o*o* #\n  ## o$ $o ##\n ### *o*o* ###\n####  $ $  ####\n   ######  #\n    ##   ##",
    # 10-09
    " ##############\n #@  * * * #  ##\n #$#  * *  #   #\n # # * * *     #\n # #  * *  ## ##\n # # * * * ## #\n # #  * *  ## #\n # # * * * ## #\n # #  * *  ## #\n # # * o * ## ##\n## ##########  #\n#              #\n#   #########  #\n#####       ####",
    # 10-10
    "#######\n# @#  #####\n# $$  $   #\n#  #o##$# #\n##$#ooo   #\n## ooo##$##\n#  ##o##  #\n#  $  $   #\n#  #   #  #\n###########",
    # 11-01
    " #############\n #    #  ##  #\n #$$$ # $$  $##\n # $  #  oooo #\n #  $  #$o##o #\n #  # $# oooo##\n##$ $  #$o##o #\n# $  $ @$oooo #\n#   ###  ######\n##### ####",
    # 11-02
    "     #########\n     #       #\n     #  $#$# #\n ######  # $ #\n #   # $  $  #\n## $     ### #\n#   #$####   #\n#    $ ### ###\n#####oo @# ##\n   #ooo$ $$ #\n   #ooo#    #\n   #ooo######\n   #####",
    # 11-03
    "    ####\n ####  ####\n #  #   $ ####\n #ooo $#$ #  #\n##o#o  #   $ #\n#ooooo ###$  #\n#oooo#$#  $ ##\n###$## $$   #\n #   #    $ #\n # $  @$  ###\n ####$# ###\n    #   #\n    #####",
    # 11-04
    "####\n#  ####  #####\n# $   ####   #\n# $    ##    #\n# ###$ $  $  ##\n# # $  $ ##   #\n#  $ #$$#   # #\n##   #  #$    #\n ##  ##  $ # ##\n####o###  $# #\n#  #o#oo #   #\n#  #oooo @ ###\n# $ oooo# ##\n#   ##oo  #\n###########",
    # 11-05
    "      #####\n#######   #\n#   ##o   #\n#  $#oo  ###\n##  ooo#$$ #####\n # $o#o$       #\n # $###$## # $ #\n #   #     $$# #\n ##$$# ##$#$   #\n #ooo $@ $   ###\n #ooo#$#   ###\n #ooo  #####\n #######",
    # 11-06
    " ####  ######\n #  ####    #\n##*   * **  #\n# $ *    *# #\n# o   ###   #\n######   #@##\n# * o *  ** #\n#   #   #   #\n##*   * #$# #\n #  #####   #\n ####   #####",
    # 11-07
    "   #######\n   #     #\n   # $ $ ##\n   #####oo#####\n######oo*o  $ #\n#  $@$oooo#$$ #\n#   $ #$###   #\n#####       ###\n    ###  ####\n      #  #\n      ####",
    # 11-08
    "         #####\n         #   #\n########## * ###\n#          o   #\n# $$$$****$ooo@#\n#          o   #\n########## * ###\n         #   #\n         #####",
    # 11-09
    "#######\n#  o$ ###\n# o$o$  #\n#*$o$o@ #\n# o$o$ ##\n#  o$  #\n########",
    # 11-10
    "   #####\n   # @ #\n   # $ #\n   #$o$#\n  ##o$o##\n###o$o$o###\n#  $o$o$  #\n#    o    #\n###########",
    # 12-01
    "     ####\n######  #####\n#@$    $  $ #\n#$### $ # # #\n#  #  # $   #\n# $#    # ###\n#  $ #$#   #\n#ooooooooo #\n########   #\n       #####",
    # 12-02
    " ####\n##  ############\n# $ $ $ $ $ $ @#\n# #         $  #\n# #  ####*###$##\n# #  #ooooo#  #\n# #$ #o***o*  #\n# #  *ooooo#  #\n# # $#######$$#\n#             #\n########### # #\n          #   #\n          #####",
    # 12-03
    "###########\n# $o $ o  #\n#  #  $o  #\n#oo*oo#*#o#\n#$$#$$$o $#\n#  o$@$o  #\n#$ o$$$#$$#\n#o#*#oo*oo#\n#  o$  #  #\n#  o$ $o  #\n###########",
    # 12-04
    "   #####\n   #   #\n   # $$#\n#### o ####\n# $ *o*   #\n# $ooooo$ #\n#   *o* $ #\n#### o ####\n   #$$$#\n   # @ #\n   #####",
    # 12-05
    "#######\n#  o  #\n#  o$ #\n# $*o #\n#* o$*#\n# **$ #\n#  o$ #\n# $o @#\n#######",
    # 12-06
    "#######\n#@ *  #\n# $o $#\n#$ *  #\n#o*o*o#\n#  o$ #\n# $o$ #\n#  *  #\n#######",
    # 12-07
    "  #####\n  #   ######\n###$#o     #\n# $ ooo# $ #\n#@ $o#*$   #\n####    ####\n   ######",
    # 12-08
    "###########\n#    #    #\n# $@$$$$$ #\n#         #\n##### #####\n   #o  #\n   #o  #\n   #ooo#\n   #o  #\n   #####",
    # 12-09
    "  #####\n  # @ #\n  # $ #\n### o ###\n#   *   #\n# ***** #\n#   *   #\n###$*$###\n  # o #\n  # * #\n  # o #\n  #####",
    # 12-10
    "##############\n#o           #\n#o$ $ $ $ $  #\n#o#########  #\n#o#o* $ oo$*##\n#o# $ $ *o$@#\n#o#o  $ oo$$#\n#o#########o#\n#o          #\n#o#$#$#$#$#$#\n#o          #\n#############",
)


def parse_level(text):
    """An ASCII level into `(walls, goals, boxes, player, shape)`.

    The static half of a level (walls, goals and the board's shape) is separated from the
    part that moves, because it never changes and copying it into every state would make
    every state a hundred times bigger than the position it describes.
    """
    rows = text.split("\n")
    height = len(rows)
    width = max(len(row) for row in rows)
    walls, goals, boxes, player = set(), set(), set(), None
    for row, line in enumerate(rows):
        for col, glyph in enumerate(line.ljust(width)):
            if glyph == WALL:
                walls.add((row, col))
            if glyph in GOAL_GLYPHS:
                goals.add((row, col))
            if glyph in BOX_GLYPHS:
                boxes.add((row, col))
            if glyph in PLAYER_GLYPHS:
                player = (row, col)
    if player is None:
        raise ValueError("level has no player")
    return frozenset(walls), frozenset(goals), frozenset(boxes), player, (height, width)


def push(walls, boxes, player, direction):
    """One step. Returns `(player, boxes)`, or None when the step is refused.

    None rather than an unchanged position, so that a caller can tell "the keeper walked
    somewhere" from "the keeper walked into a wall" without comparing states, which is what
    `successors` needs to drop the actions that do nothing.
    """
    row_step, col_step = DIRECTIONS[direction]
    target = (player[0] + row_step, player[1] + col_step)
    if target in walls:
        return None
    if target not in boxes:
        return target, boxes
    behind = (target[0] + row_step, target[1] + col_step)
    if behind in walls or behind in boxes:
        return None                        # a box against a wall, or a chain of two
    return target, frozenset(boxes - {target} | {behind})


def dead_squares(walls, goals, shape):
    """Cells a box can never leave: the corners of walls that are not goals.

    Sokoban's cheapest and safest deadlock test. Sound (a box here really is stuck for good,
    because both of the axes it could be pushed along are blocked) and deliberately not
    complete: a wrong "dead end" prunes a solvable branch, which is a far worse failure than
    letting a doomed one run. Frozen pairs, and rows that hug a wall with no goal on them, are
    dead too and are not claimed here.
    """
    height, width = shape
    outside = lambda cell: not (0 <= cell[0] < height and 0 <= cell[1] < width)
    blocked = lambda cell: cell in walls or outside(cell)
    dead = set()
    for row in range(height):
        for col in range(width):
            if (row, col) in walls or (row, col) in goals:
                continue
            vertical = blocked((row - 1, col)) or blocked((row + 1, col))
            horizontal = blocked((row, col - 1)) or blocked((row, col + 1))
            if vertical and horizontal:
                dead.add((row, col))
    return frozenset(dead)


def reachable(walls, boxes, player):
    """Every cell the keeper can walk to without pushing anything."""
    seen, frontier = {player}, deque([player])
    while frontier:
        cell = frontier.popleft()
        for row_step, col_step in DIRECTIONS.values():
            neighbour = (cell[0] + row_step, cell[1] + col_step)
            if neighbour in seen or neighbour in walls or neighbour in boxes:
                continue
            seen.add(neighbour)
            frontier.append(neighbour)
    return seen


def render(walls, goals, boxes, player, shape):
    """The position as ASCII, in the same alphabet the levels are written in."""
    height, width = shape
    lines = []
    for row in range(height):
        line = []
        for col in range(width):
            cell = (row, col)
            if cell in walls:
                glyph = WALL
            elif cell == player:
                glyph = PLAYER_ON_GOAL if cell in goals else PLAYER
            elif cell in boxes:
                glyph = BOX_ON_GOAL if cell in goals else BOX
            elif cell in goals:
                glyph = GOAL
            else:
                glyph = FLOOR
            line.append(glyph)
        lines.append("".join(line).rstrip())
    return "\n".join(lines)


class Boxxle2Action:
    """`left`, `up`, `down` or `right`."""

    def __init__(self, name):
        if name not in DIRECTIONS:
            raise ValueError(f"unknown action: {name!r}")
        self.name = name

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, Boxxle2Action) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Boxxle2State:
    """Where the keeper is and where the boxes are. The walls and goals live on the level."""

    def __init__(self, level, boxes, player, depth=0):
        self.level = level
        self.boxes = frozenset(boxes)
        self.player = player
        self.depth = depth
        self.boxes_home = len(self.boxes & level.goals)
        self.solved = self.boxes <= level.goals

        literals = [f"at(player, {player[0]}, {player[1]})",
                    f"boxes-home({self.boxes_home})"]
        literals += [f"at(box, {row}, {col})" for row, col in sorted(self.boxes)]
        literals += [f"goal(cell, {row}, {col})" for row, col in sorted(level.goals)]
        if self.solved:
            literals.append("goal-reached")
        if self.stuck():
            literals.append("terminal-state")
        self.literals = frozenset(literals)

    def stuck(self):
        """Is a box standing on a square it can never be pushed off?"""
        return bool(self.boxes & self.level.dead)

    def __eq__(self, other):
        return (isinstance(other, Boxxle2State) and self.boxes == other.boxes
                and self.player == other.player and self.level is other.level)

    def __hash__(self):
        return hash((self.boxes, self.player))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        return render(self.level.walls, self.level.goals, self.boxes, self.player,
                      self.level.shape)

    def __repr__(self):
        return (f"<Boxxle2State(home={self.boxes_home}/{len(self.boxes)}, "
                f"player={self.player})>")


class Level:
    """The part of a level that never moves, worked out once when it is loaded."""

    def __init__(self, index, text):
        self.index = index
        self.text = text
        walls, goals, boxes, player, shape = parse_level(text)
        self.walls, self.goals, self.shape = walls, goals, shape
        self.start_boxes, self.start_player = boxes, player
        self.dead = dead_squares(walls, goals, shape)

    @property
    def label(self):
        """How the cartridge numbers this level, as `"stage-level"` counting from one."""
        return f"{self.index // LEVELS_PER_STAGE + 1}-{self.index % LEVELS_PER_STAGE + 1:02d}"


class Boxxle2Game(Environment):
    """Boxxle II, implemented rather than emulated. Needs nothing installed."""

    def __init__(self):
        super().__init__("boxxle2")
        self.index = 0
        self.level = None
        self.state = None
        self.state_history = []

    def fix_index(self, index):
        if not 0 <= index < len(LEVELS):
            raise IndexError(
                f"Invalid index: {index}. There are {len(LEVELS)} levels, so the index must "
                f"be 0-{len(LEVELS) - 1}.")
        self.index = index

    def reset(self):
        self.level = Level(self.index, LEVELS[self.index])
        self.state = Boxxle2State(self.level, self.level.start_boxes, self.level.start_player)
        self.state_history = [self.state]
        height, width = self.level.shape
        return self.state, {"level_index": self.index,
                            "level": self.level.label,
                            "size": (width, height),
                            "boxes": len(self.state.boxes),
                            "goals": len(self.level.goals)}

    def is_goal(self, state):
        return state.solved

    def is_terminal(self, state):
        """A box on a square it can never be pushed off again.

        Sound, not complete: see `dead_squares`. The Game Boy sibling computes exactly the
        same test from exactly the same information: this cartridge keeps its walls in work
        RAM, so for once the twin has no analytical advantage over the emulator.
        """
        return not state.solved and state.stuck()

    def successors(self, state):
        successors = []
        if self.is_goal(state) or self.is_terminal(state):
            return successors
        for name in DIRECTIONS:
            outcome = push(self.level.walls, state.boxes, state.player, name)
            if outcome is None:
                continue                       # refused: a wall, or a box that cannot move
            player, boxes = outcome
            successors.append((Boxxle2Action(name),
                               Boxxle2State(self.level, boxes, player, state.depth + 1)))
        return successors

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        name = action.name if isinstance(action, Boxxle2Action) else str(action)
        outcome = push(self.level.walls, state.boxes, state.player, name)
        if outcome is None:
            return state
        player, boxes = outcome
        return Boxxle2State(self.level, boxes, player, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.boxes_home
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.state.boxes_home - before

    def get_actions(self):
        return [Boxxle2Action(name) for name in DIRECTIONS]

    def render(self):
        rendered = [str(state) for state in self.state_history]
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered
