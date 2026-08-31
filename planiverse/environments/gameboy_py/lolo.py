"""Adventures of Lolo in pure Python: no ROM, no emulator, no dependencies.

The sibling [`lolo_gb`](../gameboy/lolo_gb.py) drives the real cartridge. This one implements
the rules directly, the way [`puzznic`](puzznic.py) stands beside `puzznic_gb`. Use this one
for a dependency-free benchmark; use that one for the cartridge's actual behaviour.

## The rules, stated

Lolo walks the four directions, one cell at a time, on an 8x8 board. Every rule below was
measured on the cartridge (the probes are listed in
`docs/environments/lolo-gb-memory-map.md` §9) rather than taken from a manual.

1. A step into a **rock**, a **tree**, a **river** or the edge is refused.
2. A step into a **one-way pass** is refused if it goes against the arrow, and allowed from
   the other three sides.
3. A step into an **Emerald Framer** or an **egg** pushes it one cell the same way, but only
   if the cell behind is empty walkable ground. Nothing can be pulled, and no chain of two
   can be pushed at once.
4. A step onto a **heart framer** collects it. A *magic* heart framer (the cartridge stores
   two heart codes and this is the second) also gives Lolo **two magic shots**; a plain one
   gives none.
5. A step into an **enemy** is refused. Enemies never move on their own: left alone, the
   board is completely still.
6. **Magic shot**: fires one cell in the direction Lolo last tried to move, whether or not
   that move succeeded. An enemy there becomes an **egg**, which can then be pushed; an egg
   there is blasted out of the room. Each shot costs one of the two a magic heart gave.
7. A **Medusa** kills Lolo when he stands anywhere in its row or column, unless a tree, an
   Emerald Framer, a heart framer, an enemy or an egg stands between them. Rocks, rivers,
   bridges, deserts, flower beds, one-way passes and break tiles do *not* block a Medusa.
8. The **door** opens once every heart framer is collected. Standing on the open door clears
   the room.

## Where this differs from the cartridge

In one place, and it is worth stating plainly rather than burying.

**Six of the eight enemies are frozen here.** Snakey and Medusa never move on the cartridge
either, so rooms holding only those two are modelled exactly; `EXACT_ROOMS` lists the 26 of
them. The other six (Leeper, Rocky, Alma, Gol, Skull and Don Medusa) do move on the
cartridge, in lock-step with Lolo, and this module leaves them where they started. For the
137 rooms that contain one, a plan found here is a plan against a *strictly easier* puzzle,
and it may well die on the cartridge. That is the wrong direction for an approximation to
err in, so it is flagged rather than smoothed over: `Room.exact` says which rooms the model
is faithful for, and `lolo_gb` is the authority for the rest.

**Rafts are refused.** On the cartridge an egg shoved into a river floats, and Lolo can step
onto it and ride across; that is how int 1-3 is cleared. Five of the six river codes accept
one; two of them then *carry* the raft, one cell every few frames, in a direction the code
picks. Modelling a moving raft means modelling time, which nothing else here needs, so this
module refuses to push an egg into a river at all. Rooms that need a raft cannot be cleared
here; `lolo_gb` clears them.

Two smaller divergences, both in the safe direction. Medusa's shot is modelled as instant,
where the cartridge gives one move of grace, but that move cannot be used to escape, so no
plan is lost. And an Emerald Framer is not pushed onto a heart framer, a door or a marker
here, which was never tested on the cartridge and is refused rather than guessed.

**The hammer is not modelled.** A few rooms start with one in the status bar's PWR meter
(int 1-5 does, and cannot be cleared without it), and what it breaks was not established.

## Where the rooms came from

All 163 of them were decoded out of `Adventures of Lolo (U) [S][!].gb` by
`lolo_gb.read_rooms`, at matching indices: `fix_index(38)` here and on `lolo_gb` are the
same room. Nothing was transcribed by hand. `tests/test_lolo.py` re-decodes the ROM and
compares, when a ROM is available, so a room cannot drift away from the cartridge unnoticed.

The 163 slots hold 144 distinct puzzles: the tutorial's 19 are each stored twice, once as the
demonstration the game plays for you and once as the room to try, and the two halves of a
pair are near-identical but not equal.
"""
from planiverse.environments.base import Environment

# ------------------------------------------------------------------------- the alphabet
# Declared here rather than imported from `lolo_gb`, which needs PyBoy: this module promises
# to need nothing, and a twin that drags in an emulator to spell "#" would not be one. The two
# copies are pinned against each other by `tests/test_lolo.py`, which fails if they drift.
# The names are the cartridge's own: the object list at `$2CA9` is plain ASCII and reads
# "EMERALD FRAMERS / TREES / ROCKS / DESERTS / ENEMY HOLES / RIVERS / BREAK TILE /
# FLOWER BEDS / AND JEWEL BOXES", then "BRIDGE / ONE-WAY PASS / HAMMER".

ROCK, TREE, RIVER, FLOOR = "#", "T", "~", "."
BRIDGE, FRAMER, HEART, MAGIC_HEART, DOOR = "=", "O", "H", "h", "D"
DESERT, BREAK_TILE, FLOWER_BED, MARKER = ",", "x", "*", "o"
ONE_WAY = {"v": (1, 0), "<": (0, -1), "^": (-1, 0), ">": (0, 1)}
LOLO = "@"

#: The eight characters, in the order the cartridge lists them at `$2CA9`.
SNAKEY, LEEPER, GOL, ROCKY, ALMA, SKULL, MEDUSA, DON_MEDUSA = "S", "L", "G", "R", "A", "K", "M", "N"
ENEMY_GLYPHS = frozenset({SNAKEY, LEEPER, GOL, ROCKY, ALMA, SKULL, MEDUSA, DON_MEDUSA})
HEART_GLYPHS = frozenset({HEART, MAGIC_HEART})

#: How the 163 slots are grouped. `$26AA` computes `floor = (room - 38) // 14` from the room
#: number, which is what fixes 38 as the end of the tutorial.
TUTORIAL_END = 38                # 19 puzzles, each stored twice
INTERMEDIATE_START, INTERMEDIATE_PER_FLOOR = 38, 14
ADVANCED_START, ADVANCED_PER_FLOOR = 108, 5
PRO_START = 158


def room_label(index):
    """How the game itself numbers a room.

    The tutorial stores each of its 19 puzzles twice (the demonstration the game plays for
    you, then the same room to try), so its labels carry which half of the pair a slot is.
    """
    if index < TUTORIAL_END:
        pair, half = divmod(index, 2)
        return f"tutorial {pair + 1}{'a' if half == 0 else 'b'}"
    if index < ADVANCED_START:
        floor, level = divmod(index - INTERMEDIATE_START, INTERMEDIATE_PER_FLOOR)
        return f"int {floor + 1}-{level + 1}"
    if index < PRO_START:
        floor, level = divmod(index - ADVANCED_START, ADVANCED_PER_FLOOR)
        return f"adv {floor + 1}-{level + 1}"
    return f"pro {index - PRO_START + 1}"


#: An enemy that has been shot. Not a cell code (the cartridge draws it as a sprite), but it
#: needs a glyph to render and a name to reason about.
EGG = "e"

#: Ground Lolo may stand on, once whatever was on it is gone. The door is walkable at any
#: time: standing on it before the last heart is collected is allowed and simply does nothing.
WALKABLE = frozenset({FLOOR, BRIDGE, DESERT, BREAK_TILE, FLOWER_BED, MARKER, DOOR})

#: Where an Emerald Framer or an egg may be pushed to. Measured one destination at a time: a
#: Framer goes onto floor, desert, a flower bed and a one-way pass (the arrow does not stop a
#: push, only a walk), and is refused by a river, a heart framer and the door.
PUSHABLE_ONTO = frozenset({FLOOR, BRIDGE, DESERT, BREAK_TILE, FLOWER_BED, MARKER}) \
    | frozenset(ONE_WAY)

#: What stops a Medusa seeing through a cell. Measured one glyph at a time; the surprise is
#: that rocks and rivers are *not* on this list. An egg that has sunk into a river is *not*
#: counted, which was not measured either way and is the choice that cannot invent a shield
#: the cartridge does not have.
MEDUSA_SHIELDS = frozenset({TREE, FRAMER, EGG}) | HEART_GLYPHS | ENEMY_GLYPHS

#: Magic shots one magic heart framer is worth. Measured: a third shot after one magic heart
#: does nothing at all.
SHOTS_PER_MAGIC_HEART = 2

#: Row and column deltas, in the order `lolo_gb` names its buttons.
DIRECTIONS = {"left": (0, -1), "up": (-1, 0), "down": (1, 0), "right": (0, 1)}
SHOOT = "shoot"

SIZE = 8

#: The cartridge's 163 rooms, decoded from bank 13 and stored at matching indices, rows
#: separated by `|`. See the module docstring.
ROOMS = (
    #   0  tutorial 1a
    "##D#####|#......#|#....TTT|#....SH#|#....###|#......#|#.h..@.#|########",
    #   1  tutorial 1b
    "##D#####|#...h#.#|#...TT##|#..TTSH#|#..#.STT|#....#.#|#h..@..#|########",
    #   2  tutorial 2a
    "H.#.D~.L|#.#.@~..|#S...~~H|##h.....|.....###|TTT....H|..T...#.|L.H..H.L",
    #   3  tutorial 2b
    "H.#.D~.L|#.#.@~..|#S...~~H|##h.....|.....###|TTT...#H|..T...H.|L.H..H.L",
    #   4  tutorial 3a
    "#D#@##H#|#......#|G....G.#|#......G|G......#|#......G|H......h|########",
    #   5  tutorial 3b
    "#D#@##H#|#......#|GH...G.#|#......G|G......#|#......G|H......h|########",
    #   6  tutorial 4a
    "H#H#h#D#|.*.*R*S.|.H.*.H..|.*.*.*..|.*.*.*..|.*.H.*..|.*R*.*..|H.H.#H#@",
    #   7  tutorial 4b
    "H#H#h#D#|R*.*R*S.|.H.*.H..|.*.*.*..|.*.*.*..|.*.H.*..|.*R*.*..|H.H.#H#@",
    #   8  tutorial 5a
    "#T#TT..h|*****##*|HA.....H|*##*****|H.....AH|*****##*|#..#*...|.#TG.D@G",
    #   9  tutorial 5b
    "#T#TT..h|*****##*|HA.....H|.##****.|H.....AH|*****##*|#..#*...|.#TG.D@G",
    #  10  tutorial 6a
    "T~~~~#.H|TKK..#..|Th...*..|TKKH.#K.|###*##..|...K...D|......@.|TT...TTT",
    #  11  tutorial 6b
    "T~~~~#H.|TKK..#..|Th...#..|TKK..#..|###H##..|.....S.D|...K..@.|TT...TTT",
    #  12  tutorial 7a
    "..D.H...|#.......|####..M.|...S....|........|.M..#...|..G.#...|....h@..",
    #  13  tutorial 7b
    "D...h.H.|#..S.S..|#.##..M.|...S....|........|.M..#...|..G.#...|....h@..",
    #  14  tutorial 8a
    "N......H|TT..TTT~|N......~|..SH...~|..Sh@D.~|.#.....~|.#.....~|.#~~~.~~",
    #  15  tutorial 8b
    "N......H|TT..TTT~|N......~|..SH...~|..Sh@..~|.#...D.~|.H.#...~|.#~~~.~~",
    #  16  tutorial 9a
    "N..D...M|..G.....|...#...#|...#...#|...#..G#|...Oh@.H|...#...#|.###...#",
    #  17  tutorial 9b
    "N..D...M|...#....|...#...#|...#...#|...G...#|..OOh@.H|...#...#|.###...#",
    #  18  tutorial 10a
    "....G#.D|N......H|.O###TST|........|#KTTTT#.|.......N|T###T#S.|@.....h.",
    #  19  tutorial 10b
    "....G#.D|N......H|.O##.TST|........|#K.T.T#.|.......N|T###T#S.|@.....h.",
    #  20  tutorial 11a
    "NTDT~~GH|.T.T....|.T.T....|.T.T....|.T...TTT|TT@.OT..|.....T..|H......M",
    #  21  tutorial 11b
    "NTDT~~GH|.T.T....|.T.T....|.T.h....|.T..GTTT|TT@..~..|....K~..|H......M",
    #  22  tutorial 12a
    "~~~~~...|#.hhT..@|#...T.K.|#.....K.|#...T.KH|#TTTT,,,|#...K,,,|#....,,D",
    #  23  tutorial 12b
    "~~~~~..H|#.hhTO.@|#...T.K,|#.....K,|#...T.K,|#TTTT,,,|#...K,,,|#....,,D",
    #  24  tutorial 13a
    "M.....oD|.######.|.#......|o~......|.~......|h#.#####|.O...SO.|.....h.@",
    #  25  tutorial 13b
    "M...Hh.D|.##.##.K|.#...o..|o~......|.~......|h#.#####|.O...SO.|.....h.@",
    #  26  tutorial 14a
    "MT.####D|.O......|H..T....|~~~~####|.S.~#..@|.H.~....|...~.Sh.|...~....",
    #  27  tutorial 14b
    "MT.####D|.K.S~~~.|H..T...T|~~~~####|.S.~#..@|.h.~....|...~.Sh.|...~....",
    #  28  tutorial 15a
    "D##H~###|~.#.=..h|~.##~...|~...~...|~~~~~~~x|#..S..@.|..ST....|...T....",
    #  29  tutorial 15b
    "D##H~###|~.#.=..H|~.##~...|~..hx...|~~~~~~~x|#..S..@.|..ST....|...T....",
    #  30  tutorial 16a
    "M..T..*H|.**DT*..|~...**A*|~T***T..|~T...T*L|~T...T..|~TO..*H*|H...@*.H",
    #  31  tutorial 16b
    "M.HG..*H|.**DT*..|~...**A*|~T***T..|~TTK.T*L|~T...T..|~TTO.*h*|H...@*.H",
    #  32  tutorial 17a
    "~~~~*.#h|~**~**#.|~SH~.L*.|~##~~~~~|~......H|~..D....|~~~~..@H|H..x....",
    #  33  tutorial 17b
    "~~~~.*#h|~**~.*#.|~SH~.L*.|~##~~~~#|~......H|~..D....|~~~~..@H|H..x....",
    #  34  tutorial 18a
    ".T.....T|.D###v#.|T.#<>H<.|T.#>#^#T|..#v#.#.|.#><#H#.|vH<.#^.v|T^TTH>H@",
    #  35  tutorial 18b
    ".T.....T|.D###v#.|T.#<>H<.|T.#>#^#T|.v#v#.#.|>H><#H#.|vH<.#^.v|T^TTH>H@",
    #  36  tutorial 19a
    "h..~~D.H|....#HK.|#***~<..|#...#**H|#...##T#|#..*..^.|#.K#@.#.|H..####M",
    #  37  tutorial 19b
    "GhS~~D.H|....#HK.|#***~<..|#...#*.h|#...##T#|#..*..x.|#.K#@.#.|H..####M",
    #  38  int 1-1
    "##.#####|#D..H.H#|#......#|#......#|#...H.H#|#......#|#@..H.H#|########",
    #  39  int 1-2
    "#####.##|#...D..#|#M.....#|#.O.H..#|#.....M#|#....O.#|#..@...#|########",
    #  40  int 1-3
    "###D####|#......#|#......#|#......#|~~~~~~~~|#....S.#|#......#|###@.h##",
    #  41  int 1-4
    "#SSS####|#SDS...#|#.S....#|#......#|........|.....@..|.S....O.|M.H##h.M",
    #  42  int 1-5
    "####D###|M......#|#......M|#OH..HO#|#......#|#..T...#|TTTSTTTT|H....@.H",
    #  43  int 1-6
    "#H..@.H#|#.O..O.#|...D....|H.#..#.H|#NO..ON#|#.#..#.#|#.O..O.#|#H....H#",
    #  44  int 1-7
    "HTHTHTHD|...T...#|R#.*.TR#|.#.#.#.#|.#.T@#.T|*..T.#.T|.#....*.|H#HTHTHT",
    #  45  int 1-8
    ".......D|.KKOKK..|.KOHOK..|.OH.HO..|.OH@HO..|.KOHOK..|.KKOKK..|........",
    #  46  int 1-9
    "H..~D...|...==.SH|...~~...|~~=.M=~~|#....HOh|#.O..O..|##....S@|####....",
    #  47  int 1-10
    "**...LAh|...A.**.|H******.|....D**L|.***@**A|.*.***..|.A...L.M|.M.H....",
    #  48  int 1-11
    "H.######|#.#D,.##|#Hv#,,##|vv*vO,vT|.T>Tv.@.|*.O<O...|.*.*.*..|H#HKKKK#",
    #  49  int 1-12
    ".....H..|.#~~~~#.|G*.Kh~~.|.~~KD~H.|.H~K@~H.|.H~~~~~.|.#~HH~#.|.......M",
    #  50  int 1-13
    "#T###.D~|~O.....~|~TTT~~.~|~TT.H~.~|~T~~~~.~|.S@h...~|..THHT.#|H##..N..",
    #  51  int 1-14
    "..@.....|.**O.O..|#OO...M.|...M...#|.......#|.TM...RH|..T..O..|#....D.#",
    #  52  int 2-1
    "h...M..D|#...#...|....<...|S...#.#.|#.S##.GS|#.S.#...|..hh*...|#@..#.HH",
    #  53  int 2-2
    "M*HHHH..|.*..@..#|.*..S.##|*****.#*|#AHhO*#.|**#*K#*M|..D.KK*.|H.....*M",
    #  54  int 2-3
    ".><^v.M.|<<>.vv<D|.^v<v^.v|>>O<<S>H|<v^.v.<.|>.<v.SOv|v^>.<.^.|@^<^.S.M",
    #  55  int 2-4
    "T..h@.TT|T..S..#T|T#vDv###|H#..*..H|...*L*..|....*...|...N...T|M...H.TT",
    #  56  int 2-5
    ".......N|##T**.*H|**x*OH**|*G#HD..*|*T~~~.*#|*G#=#T**|**~#G*hR|T....@.H",
    #  57  int 2-6
    "D.~.~..H|.S=.=...|..~=~~=~|.~.O.~H.|.~.H==.M|.=O.~~h.|@~~=~~=~|........",
    #  58  int 2-7
    "M.H.~..M|L.*.~..L|..*.~...|***.~~~~|HH.....h|...@O***|....*...|.D..*..M",
    #  59  int 2-8
    "M#.Dh..M|#.O..Oh#|#.HHHS.#|#.H#H..#|.@HhH.N#|#.H#HT.T|..HHH.SH|H...##TT",
    #  60  int 2-9
    "M,T,hDTT|THS,,@TT|K.T.G.MT|..K...MT|..T.H.KT|..K.S.TT|hTTOG.TT|...O...H",
    #  61  int 2-10
    "....*KKK|..@*...O|...T....|.*TDTKKK|.OT,TTKK|..T,,,KK|.H.TTTKK|##......",
    #  62  int 2-11
    "~~#.#~~~|M....OH.|......O.|~T...~~D|H..ShGGT|H#...TTT|~T#.HOHT|~~~@.~~~",
    #  63  int 2-12
    "M#..h.#M|*.h...H*|....@...|h.SD.S.h|.S..S...|...h..h.|.......#|N....H*M",
    #  64  int 2-13
    "HH...H~H|.O.DHS.#|..*...*#|.S#G..H#|h@#GNN.#|...***##|#..**O*#|##.*HHHH",
    #  65  int 2-14
    "########|#.hS~...|h@..~.h.|S~~~~..R|~~HG~~~~|~~.T~..#|~S..~.H#|##D.####",
    #  66  int 3-1
    "h#***###|H...**RH|#.#.*T#@|**OD~G#.|***H~=~.|~#*GTG~.|#~*HHH*.|#~~~####",
    #  67  int 3-2
    "D**...@<|*TTTTT#v|*h#G#..H|S**#..#.|*.,>*T..|..#,H.O.|AH#H.N..|M#T.#...",
    #  68  int 3-3
    "@.*A..~H|T*#H.**#|..*.*.H#|.*L...##|h#*..T*#|T#H..##*|##..*~D#|#H.NhT..",
    #  69  int 3-4
    "hh~..H*M|..D...*.|.~.@~O*.|~..~.~*.|#.~.R~*.|H.O~*K*.|....****|M#.....H",
    #  70  int 3-5
    "M....HTM|........|H.O~HS@.|#.~O..##|..~.h.O.|..~~O.~.|.~#~h.~D|MT##~~~.",
    #  71  int 3-6
    "##D#.G##|#~~..#..|~~S@*...|~#**ShH#|~~~~~<~M|#..^O..*|#.S...A*|###....h",
    #  72  int 3-7
    "M...D..M|**.@H.**|........|#*O**O*#|H*....*H|L.*..*.L|**....**|M.A..A.M",
    #  73  int 3-8
    "##.~H~.K|~D.>x..H|~@.>xS~~|~...<~#~|~.O.xH~~|H=Kh.xxH|#*~=~*~#|A..xH..A",
    #  74  int 3-9
    "M.HKKG,D|KK...,,,|..hK.#O,|.K####,,|.O.HK,,,|.O@O.H.G|.#HOKK.H|H.....HH",
    #  75  int 3-10
    "~...###M|~.@xh*.L|~.##.*hH|~=~~.*.A|...~S##~|..O~S...|.#.~.,..|DGG~.H,.",
    #  76  int 3-11
    "~M~~~###|~.*....H|H.*h....|..##.@h.|***#.R..|#^K..*.*|#H.#***M|#~~~~D~~",
    #  77  int 3-12
    "H~~~~~~D|.~~TT.~~|Th......|.......T|....M.ST|..OO.T~H|hS~OH#SH|@...TThH",
    #  78  int 3-13
    "#M#MM#.H|........|.O...O..|@HOO..O.|.H.MKMT.|.OH...O.|.O..,,,,|KKMKD#,H",
    #  79  int 3-14
    "#*T*h.H.|TT*H@..#|..S.....|~~~=~~TM|...#.OA*|T.h~.*HD|.TT~G*##|#.H~H**T",
    #  80  int 4-1
    "M...D..H|.TT...TT|..T..TT.|...HHTS.|M..H@...|...h....|h.S...TT|TT.h...M",
    #  81  int 4-2
    "H##,,,,D|H..K,,,,|@G...***|h#.K#GGG|h.K.**##|#.K.O*##|#...**##|########",
    #  82  int 4-3
    "..h.@<HD|..SS.^.#|M.>^.<.#|~~~~S~~~|T.O~.~^.|...~S...|hT<T....|h<.MTH.M",
    #  83  int 4-4
    "M.H###.M|~.H@....|~*H..OD.|~*~~~~G.|~******.|~LHhHH*.|~*****..|H......H",
    #  84  int 4-5
    "~..>..#M|~H.~<.#.|#Kx~#h#.|#~@~~...|G~h<S...|D~R<....|.#TTHH.*|.#....Nh",
    #  85  int 4-6
    "..~~~M.h|..*.=T**|h.##=..#|^##@~~=~|.x.H#~=G|.#H.v~=~|R~~~..**|~~~DG.KG",
    #  86  int 4-7
    "M~~~=~.H|~D.KO~..|~..#h~TM|~=~~x~.T|..S.....|.....OO@|........|H....N##",
    #  87  int 4-8
    "K~~~..D.|H.A~@~~K|K~x#h..#|~..~xK..|~.KK~##.|K^~.x.~.|~.~xKxH.|K...x...",
    #  88  int 4-9
    "G.*HRH**|#.TTTTT*|#h...@*H|#***#.*.|#x##G.H*|#xG*x*.#|#TT#,.#G|##H#DG##",
    #  89  int 4-10
    "~~###~##|##K#K#~#|#..KhK#~|#...vS*~|#~~~~~~M|H,D.S**G|....*LH.|~~#@h*.#",
    #  90  int 4-11
    ",,,..H.A|,,,...#.|,,,....H|...OO...|...ODO..|H..@.,,,|.#...,,,|A.H..,,,",
    #  91  int 4-12
    "###~~~M#|hh.....#|#...##H~|**H@#..~|*S*.KD.M|**hK#~.#|*S*..~.#|#*...~.#",
    #  92  int 4-13
    "R>*M.HTD|.>*T...~|.O*...@.|M.*....h|.#v~~~#^|.#...H~.|.R###.~.|H.......",
    #  93  int 4-14
    ".G.....M|.vG.#...|D.#~~#O#|H@~~H~S.|..T#~~.h|.h~T=T..|..H..h..|.......N",
    #  94  int 5-1
    "~~~~~H~#|S#D=v~o*|H#SvO~v.|^..oO*^^|h####.*~|..h@*..~|#..*LS.~|##..~~~#",
    #  95  int 5-2
    "H.>....H|.GT##~~=|HGGG#@.<|.##~~H.#|*.HL*..~|.ODH*~~.|...~#H..|#H.Nx>.L",
    #  96  int 5-3
    "M.#*...#|TO..#DvM|H.~>@#OG|.##h<#..|~>O>##<T|.S>##>v.|..#.#.#^|H^xx.H<H",
    #  97  int 5-4
    "D>...H*#|~T~~^TLT|....<H*.|vST.Tv.#|~..Th*@T|##..*#*.|>....*.*|MGG~#.*H",
    #  98  int 5-5
    "#~~~.h..|...O.S.#|ho.~.@~~|#O.~.~~h|#..#~#~~|#..~~~o#|#..~oOD.|#~~#.###",
    #  99  int 5-6
    "Dh..>.#M|~x~~#.H~|...A#...|..S.~#..|.#H#.~#.|.#..@.~.|H~~#~**.|........",
    # 100  int 5-7
    "..M.D.oM|###v.*v#|@..*.TO.|.Tv..o*.|h*T....H|AO<vS~~#|.**.~~##|....####",
    # 101  int 5-8
    "##H***#D|#.......|h.M..#HH|....G...|O.H..G.H|......*.|.######.|@h.....A",
    # 102  int 5-9
    "~A##H..M|~v~~hG#.|#~A*.**M|G~~~#...|h@<#*..D|.S#H***h|..<*....|H.###..M",
    # 103  int 5-10
    "@.....>~|..O<hS#~|.S~^.~#~|.~~=~~H~|h~hGM~S~|.#G.S~~~|~~~~~~=.|~~~~~~~D",
    # 104  int 5-11
    "hG.>@*.K|..T#.*,D|..MTh.*~|^.MT~~v~|..T##..~|...^.S.~|...#.<..|h~~~~.#M",
    # 105  int 5-12
    "~~~~~~~D|..#####T|=~##.##M|hG@....T|h,>#^Gv#|*O>~.O.~|.T>~~~GS|NT....H.",
    # 106  int 5-13
    "#H.~h..#|##.~.o.G|M.T~...D|#..O#~~~|#...#H~=|#.S.##~.|...hS...|h@......",
    # 107  int 5-14
    "#.h=<..H|.@.~##..|DS<~H#..|G~=~~~=~|#~v#o~^#|#....~<H|#.S..~..|#h.~~~#M",
    # 108  adv 1-1
    "N...T~h~|.T.h<~~~|.^>O..o@|...ST..#|...~<..*|...T^.O.|~~~~T~v.|HD~TH~o<",
    # 109  adv 1-2
    "Gh~~~~~#|~S~~~GM~|Hh~~hh~~|..~~~~~~|.@~.S..~|oS~.###~|~.~=~~~M|D.....M.",
    # 110  adv 1-3
    "Moh<vD.H|#..T~.TH|.#H.~TT.|.ooO~T.T|....~..T|#...~.O@|T#.#=.Sh|T~~~~...",
    # 111  adv 1-4
    "#hh...*H|##..##^#|#..#@S.~|#..v....|#~~^O^.#|~~~~~G^H|~#~#~.oT|#~~D#H.M",
    # 112  adv 1-5
    "H..#~~~M|..HT~.~.|LvvO~A~.|#.H*~~~T|##,..G<h|.@..*~~~|D..G~A.~|...h~~~~",
    # 113  adv 2-1
    "H....M..|N...H#.D|#..O#oGH|...S@#H.|......H.|....hO..|#S..OH.N|.h.#..M#",
    # 114  adv 2-2
    "@...D###|.^H...^.|<T...NT>|.N####.H|..#MM#..|H.####N.|<TN...T>|.v...Hv.",
    # 115  adv 2-3
    "~~~~~~~~|~~G~~~#~|H###Hhh~|G~G~@~~#|~~~~ShG~|#~DG~~~~|hS~~#~~~|~~~~~GhS",
    # 116  adv 2-4
    "D~o#MMM.|,~v#...@|,,*..~~.|*v.*HHH.|.#SH....|.#A<*..O|.~~~...*|A...h..*",
    # 117  adv 2-5
    "H@.....#|oT#v^~~#|M.#Gh...|.....T#.|H.#.S.h.|...S.O..|o......#|~D.#~..M",
    # 118  adv 3-1
    "GDG##.@#|~K~KHKOh|~H~h#..#|~#~#.#..|~o#.#...|#.##~K..|H.<vG.<#|#.H.##.h",
    # 119  adv 3-2
    ".M.M~~.H|H.#.~T..|...T~.*~|..#v.h@~|..G>...T|..#H^.v.|.D~~K#S.|NTh..*..",
    # 120  adv 3-3
    ".=..NT~~|.~...TO.|.~..o>.S|.=.h.>..|.~.D.Th.|N~.#.T@.|H=S#~~~=|#~......",
    # 121  adv 3-4
    "H#D#.H=M|~.<>*..~|v<.S*>*~|~T*h#@~.|~#AT#*#.|H#*#*.o.|~#..h..#|~T.<..*H",
    # 122  adv 3-5
    ".....x..|.#~^~#Sv|.~~D~~Sv|.~~####*|.~.@#L*.|.=**O<o*|T*O*h<HH|H#.hN...",
    # 123  adv 4-1
    "D#~~~~#h|.#~h~S~~|.#~O.O~S|~~~N.S~h|.S~~o~~.|.~~~#=O.|~~.S^~S<|##.H.~.@",
    # 124  adv 4-2
    "###HD<H~|.@h^o#^~|.O#N...~|..H...H.|.S..M.Hv|..H..N..|~O....#~|~==~~~~~",
    # 125  adv 4-3
    "#HTH<D<H|Tv#<#~~~|#.#Sv~h~|^.#~~~~~|T^##S#Tx|...@hT#x|#~~.Sxxo|~H~.xx..",
    # 126  adv 4-4
    "D.#.#H#H|.O>#.v>x|.>..<x#<|S#>v#^.H|..>o<OH#|v<#^#>v<|~#v#<oOv|HOh<>Oo@",
    # 127  adv 4-5
    "M~~##o.H|##~~~~~D|~~~~#.#~|~~H~~~~G|o~h~~~hH|~SS~#~OS|~@hO~~~~|~~~~~~##",
    # 128  adv 5-1
    "M.~.DH.M|.O.~#<S~|.^#S#@h~|..H.##.~|..##.h~~|*O#.^*v~|.....*.~|H.#..*LM",
    # 129  adv 5-2
    "...#.#~H|...##.o#|.Sh*..*D|HhS@*.#.|**HHO##*|HAA*o<..|~HhH*~~~|H~~H**TH",
    # 130  adv 5-3
    "T~~~.H.h|.D.H#.S#|~~~#.@.~|~L..*#.~|#~~*##.~|#~~.o..~|...H.hS~|H#~###MT",
    # 131  adv 5-4
    "DH~L*#H.|###TH#o.|..~~.>>.|.K#vT#..|.><hhH*.|~G*@.H*N|.*....*.|*..####.",
    # 132  adv 5-5
    "TD<HH.#M|~v~<.o..|~OS~#T#.|T~~#~HT.|...hh~T.|..o..~#.|.Oo..=..|SOoT@~#.",
    # 133  adv 6-1
    "Thh.##M#|T......T|#...#vH~|~#H#T,,~|LS<.GD*M|T*hG#~.T|*....~.#|.T@..~h#",
    # 134  adv 6-2
    "#H###.<h|HAH.>O~<|HAH....~|HAH..#<.|hAh.@...|Hhh..***|...S*Ko.|###..*KD",
    # 135  adv 6-3
    "~~##...*|G~~ov##T|D.*~~h@.|.~###.#.|.~*h....|MS...*.#|.h.G*L*#|TT>.**T#",
    # 136  adv 6-4
    "M.~~~~.D|........|.KKOKK.~|.KO@OK.~|.O.H.O.~|.KO.OK.~|.KKOKK..|.......M",
    # 137  adv 6-5
    "M.h~oM.h|..A.~TO.|*H~.*~.~|~~~*.<~.|h~..~.O.|......S.|~~.TTT.h|D.~#G>.@",
    # 138  adv 7-1
    "H.##.MoD|~S...#..|...v.<..|.ThS~~=~|^#hO.~..|.*A*@#..|..*<*T..|#~#*.#.T",
    # 139  adv 7-2
    "#H....T.|M~~S....|M~TvH...|o~.@H...|H~..H...|##HOHhSv|D..ohh.N|SO..TNh.",
    # 140  adv 7-3
    "h~~~~#*M|.~....o*|h~h~=~~D|@~GG..~~|.~~*..v#|.O*.**.#|#.*.*L*.|H^*..*.M",
    # 141  adv 7-4
    ".#~~h<H#|D#h#~S..|~#~Sv~~~|~#~~~h#~|~~Ho>~#~|~GS#~#.^|G#~=S..S|#~#~#.h@",
    # 142  adv 7-5
    "M.~###.H|#*D...*.|#.#T.~.N|#.#Kh..T|.*h.xS.@|^^~.^^Gh|~oh.h~G*|H....N*.",
    # 143  adv 8-1
    "....##DH|.ST#H#H.|~.##~#..|.S@h.<..|..T.....|..THHT..|.>.^^.#.|H>h...NN",
    # 144  adv 8-2
    "******~h|~v*h@<~#|~SS<.<~#|.#~*#h~.|.>>R..~.|.DT*.o~M|..#THH#T|......NH",
    # 145  adv 8-3
    "H.h.M#.D|.K^.#M.H|M...*...|.K..*.*.|~...K.*.|~v^.ML..|.^*O#hhh|H.^h*..@",
    # 146  adv 8-4
    "M*.H.MM.|.AH*<<.o|x.*H*hh*|**#~~*TT|A>>*.^>.|o*h~..S@|~~##OTTh|D.#.....",
    # 147  adv 8-5
    "T~M#A~A*|hO.T.>.H|**h~~~~~|.AH~~##~|~#~#~~#~|~S*#~A..|h#*~~.TD|.^>H~h@T",
    # 148  adv 9-1
    "M##oh.~M|~.......|..S~Sh@.|h.~..<##|..~.Sh..|.S#...~.|..T...~.|M~##H#~D",
    # 149  adv 9-2
    "~~Sxh..@|.Oo.*SH#|M.~~=~Gh|#.~.A..h|#.~##D~#|..~~~~~#|..o..>O.|M##.#..H",
    # 150  adv 9-3
    "#.Oo...D|h.#..o.o|#.#..MMo|NT......|......oO|O.Oh.OO.|.#......|G.@oG.##",
    # 151  adv 9-4
    "D*.***..|.*.*H..*|.*.**M..|.*.M....|.*.T.hAh|M*A**.M.|.OATh..h|h*.M.LO@",
    # 152  adv 9-5
    "D.~h..**|..~..#*A|.O~~~~~~|..~.T>h~|..~MTS<~|.S~..><~|@h#H.<S~|..>....M",
    # 153  adv 10-1
    "T.##..*.|h*...*A*|*K#^.*.H|T*H*##**|vv#*T#*.|*^**#*O#|*****@H#|D*#####M",
    # 154  adv 10-2
    "##M~A..*|HS.~**.h|**H~G#~~|.AH~#G~A|~~##~~~*|~S~#~A..|H.~~~.TD|.^#h~h@~",
    # 155  adv 10-3
    "MT#h#**M|o#h~*H<*|D.^#o*#~|##^T*AHH|...T>hGv|>###>*.#|H~~<>S.#|H..>..@H",
    # 156  adv 10-4
    "D...~h~h|....S*~,|HN...*~#|N~.H.###|~~*h....|~N*.STh.|~.*..*A*|~h.~~~*@",
    # 157  adv 10-5
    "#.....#h|#^^xxx..|#..##H.#|#.K.<S#o|#~~~#~#M|#D^*L*#.|##hh*G..|#H@.....",
    # 158  pro 1
    "#A.....H|#.*####.|..*...#.|H#.@DH*H|.#..#.HA|.#.##.HA|A#..#A#.|A.H.HA#.",
    # 159  pro 2
    ",,,...HA|,,,...#,|,,,,,,.H|,,,OO@..|...ODO,,|H..,,,,,|,#...,,,|AH...,,,",
    # 160  pro 3
    "@..##GG#|H..##..#|...#G.D.|...##..#|...##..#|.......#|.......#|GGG....G",
    # 161  pro 4
    "##~S...@|.H~...O.|..~.hhh.|..~.hAh.|..~.HAHG|D.=.HAH.|..~##AH#|GG~AAAH#",
    # 162  pro 5
    "DH.NN.H.|.O....O.|.OLAALO.|..HHHH..|..HHHH..|.OLAALO.|.O....O.|@H.NN.H.",
)


def parse_room(text):
    """One room's text into `(terrain, hearts, framers, enemies, lolo, door)`.

    The static half (terrain and the door) is separated from the parts that move, because
    the terrain never changes and copying it into every state would make every state far
    bigger than the position it describes. Objects are lifted out of the grid and the ground
    under them left as floor, so `terrain[r][c]` always answers "what would Lolo be standing
    on here".
    """
    rows = text.split("|")
    terrain, hearts, framers, enemies, lolo, door = [], {}, set(), {}, None, None
    for row, line in enumerate(rows):
        ground = []
        for col, glyph in enumerate(line):
            cell = (row, col)
            if glyph in HEART_GLYPHS:
                hearts[cell] = glyph == MAGIC_HEART
                glyph = FLOOR
            elif glyph == FRAMER:
                framers.add(cell)
                glyph = FLOOR
            elif glyph in ENEMY_GLYPHS:
                enemies[cell] = glyph
                glyph = FLOOR
            elif glyph == LOLO:
                lolo = cell
                glyph = FLOOR
            elif glyph == DOOR:
                door = cell
            ground.append(glyph)
        terrain.append(tuple(ground))
    if lolo is None or door is None:
        raise ValueError("room has no Lolo, or no door")
    return tuple(terrain), hearts, frozenset(framers), enemies, lolo, door


def inside(cell):
    return 0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE


def one_way_allows(glyph, direction):
    """May a one-way pass be entered by something travelling `direction`?

    Only the arrow's own reverse is refused: a pass pointing left blocks anything walking
    right through it and lets the other three sides by. Measured from all four sides on all
    four arrows.
    """
    return glyph not in ONE_WAY or ONE_WAY[glyph] != (-direction[0], -direction[1])


def occupied(state, cell):
    """Is something standing on `cell` that Lolo cannot walk through?"""
    return (cell in state.framers or cell in state.eggs
            or cell in state.alive or cell in state.hearts)


class Room:
    """The part of a room that never moves, worked out once when it is loaded."""

    def __init__(self, index, text):
        self.index = index
        self.text = text
        terrain, hearts, framers, enemies, lolo, door = parse_room(text)
        self.terrain, self.door, self.start = terrain, door, lolo
        self.start_hearts, self.start_framers, self.enemies = hearts, framers, enemies
        self.medusas = frozenset(cell for cell, glyph in enemies.items() if glyph == MEDUSA)
        self.enemy_cells = frozenset(enemies)
        #: Enemy kinds this room holds that this module does not move. Empty means the model
        #: is faithful; see the module docstring.
        self.unmodelled = frozenset(set(enemies.values()) - {SNAKEY, MEDUSA})

    @property
    def exact(self):
        """Does this module model this room the way the cartridge plays it?"""
        return not self.unmodelled

    @property
    def label(self):
        """How the game itself numbers this room."""
        return room_label(self.index)


#: The rooms whose only enemies are the two that never move on the cartridge either, so the
#: model here is faithful rather than an approximation. Computed rather than typed, so it
#: cannot fall out of step with `ROOMS`.
EXACT_ROOMS = tuple(index for index, text in enumerate(ROOMS) if Room(index, text).exact)


def blocked_by_medusa(room, framers, eggs, hearts, alive, cell):
    """Is `cell` in a Medusa's clear line?

    A Medusa sees along its own row and column, for the whole width of the board, and is
    stopped only by a tree, an Emerald Framer, a heart framer, another enemy or an egg. Rocks
    and rivers do not stop it, which is the single most surprising thing measured on this
    cartridge and the one most likely to be read as a bug here.

    A Medusa that has been shot into an egg is not in `alive` and does not fire.
    """
    for medusa in room.medusas & alive:
        if medusa[0] == cell[0]:
            fixed, axis, low, high = medusa[0], 1, *sorted((medusa[1], cell[1]))
        elif medusa[1] == cell[1]:
            fixed, axis, low, high = medusa[1], 0, *sorted((medusa[0], cell[0]))
        else:
            continue
        between = [(fixed, step) if axis else (step, fixed) for step in range(low + 1, high)]
        if any(room.terrain[row][col] in MEDUSA_SHIELDS or (row, col) in framers
               or (row, col) in eggs or (row, col) in hearts or (row, col) in alive
               for row, col in between):
            continue
        return True
    return False


class LoloAction:
    """`left`, `up`, `down`, `right` or `shoot`."""

    def __init__(self, name):
        if name not in DIRECTIONS and name != SHOOT:
            raise ValueError(f"unknown action: {name!r}")
        self.name = name

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, LoloAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class LoloState:
    """Where Lolo is, what he is carrying, and where everything movable stands."""

    def __init__(self, room, lolo, hearts, framers, eggs, alive, sunk, shots, facing, depth=0,
                 dead=False):
        self.room = room
        self.lolo = lolo
        self.hearts = dict(hearts)
        self.framers = frozenset(framers)
        self.eggs = frozenset(eggs)
        #: Enemy cells still holding an enemy. Shooting one moves it into `eggs`; shooting the
        #: egg drops it from both, and the cell is walkable from then on.
        self.alive = frozenset(alive)
        #: River cells holding an egg that has been pushed in and sunk. Lolo may walk on one;
        #: it floats away the moment he steps off, so it is a bridge that works once.
        self.sunk = frozenset(sunk)
        self.shots = shots
        self.facing = facing
        self.depth = depth
        self.dead = dead
        self.hearts_left = len(self.hearts)
        self.door_open = self.hearts_left == 0
        self.solved = self.door_open and lolo == room.door and not dead

        literals = [f"at(lolo, {lolo[0]}, {lolo[1]})",
                    f"hearts-left({self.hearts_left})",
                    f"shots({self.shots})"]
        literals += [f"at(heart, {row}, {col})" for row, col in sorted(self.hearts)]
        literals += [f"at(framer, {row}, {col})" for row, col in sorted(self.framers)]
        literals += [f"at(egg, {row}, {col})" for row, col in sorted(self.eggs)]
        literals += [f"at(sunken-egg, {row}, {col})" for row, col in sorted(self.sunk)]
        literals += [f"at(enemy, {row}, {col})" for row, col in sorted(self.alive)]
        # `facing` is part of `__eq__` for the reason given there -- it decides where the next
        # shot goes -- so it has to be here too. A planner reasons over these predicates and
        # nothing else: leave a field out of them and two positions that really are different
        # become one, and a search can run out of frontier and call that a proof.
        if self.facing is not None:
            literals.append(f"facing({self.facing})")
        if self.door_open:
            literals.append("door-open")
        if self.solved:
            literals.append("goal-reached")
        if self.dead:
            literals.append("terminal-state")
        self.literals = frozenset(literals)

    # Enemies never move here, but they can stop being enemies, so `alive` and `eggs` are both
    # part of the identity. Depth is not. `facing` is, and earns its place: it decides where
    # the next shot goes, so two positions that differ only in it really are different.
    def __eq__(self, other):
        return (isinstance(other, LoloState) and self.room is other.room
                and self.lolo == other.lolo and self.hearts.keys() == other.hearts.keys()
                and self.framers == other.framers and self.eggs == other.eggs
                and self.alive == other.alive and self.sunk == other.sunk
                and self.shots == other.shots and self.facing == other.facing
                and self.dead == other.dead)

    def __hash__(self):
        return hash((self.lolo, frozenset(self.hearts), self.framers, self.eggs, self.alive,
                     self.sunk, self.shots, self.facing, self.dead))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        return render(self)

    def __repr__(self):
        return (f"<LoloState(depth={self.depth}, hearts_left={self.hearts_left}, "
                f"shots={self.shots}, lolo={self.lolo}, dead={self.dead})>")


def render(state):
    """A position as ASCII, in the same alphabet `lolo_gb` prints with."""
    rows = [list(row) for row in state.room.terrain]
    for row, col in state.alive:
        rows[row][col] = state.room.enemies[(row, col)]
    for row, col in state.eggs:
        rows[row][col] = EGG
    for row, col in state.framers:
        rows[row][col] = FRAMER
    for (row, col), magic in state.hearts.items():
        rows[row][col] = MAGIC_HEART if magic else HEART
    rows[state.lolo[0]][state.lolo[1]] = LOLO
    return "\n".join("".join(row) for row in rows)


def move(state, direction):
    """One step. Returns the successor, or None when the step changes nothing.

    None rather than an unchanged position, so a caller can tell "Lolo walked somewhere" from
    "Lolo walked into a rock" without comparing states, which is what `successors` needs to
    drop the actions that do nothing. A bump does turn him, though, and that is a real change
    when the next action is a shot, so a refused *move* is still a successor when the facing
    it leaves behind is new.
    """
    room, step = state.room, DIRECTIONS[direction]
    target = (state.lolo[0] + step[0], state.lolo[1] + step[1])
    facing = direction

    def turned():
        """The step was refused; the only thing that happened is that Lolo turned."""
        if state.facing == direction:
            return None
        return _settle(state, state.lolo, state.hearts, state.framers, state.eggs,
                       state.alive, state.sunk, state.shots, facing)

    if not inside(target):
        return turned()
    ground = room.terrain[target[0]][target[1]]
    on_sunken_egg = target in state.sunk
    if not on_sunken_egg and ground in (ROCK, TREE, RIVER):
        return turned()
    if not one_way_allows(ground, step):
        return turned()
    if target in state.alive:
        return turned()

    framers, eggs, sunk = state.framers, state.eggs, state.sunk
    if target in framers or target in eggs:
        behind = (target[0] + step[0], target[1] + step[1])
        if not inside(behind):
            return turned()
        beyond = room.terrain[behind[0]][behind[1]]
        if beyond not in PUSHABLE_ONTO:
            return turned()
        if occupied(state, behind) or behind in state.sunk:
            return turned()
        if target in framers:
            framers = frozenset(framers - {target} | {behind})
        else:
            eggs = frozenset(eggs - {target} | {behind})

    hearts, shots = dict(state.hearts), state.shots
    if target in hearts:
        if hearts.pop(target):
            shots += SHOTS_PER_MAGIC_HEART
    return _settle(state, target, hearts, framers, eggs, state.alive, sunk, shots, facing)


def shoot(state):
    """Fire the magic shot one cell ahead. Returns the successor, or None if nothing happens.

    Nothing happens when Lolo has no shot left, or when the cell he faces holds neither an
    enemy nor an egg. A shot at an enemy turns it into an egg; a shot at an egg blasts it out
    of the room, which is how a room with a Snakey in a corridor is opened up.
    """
    if state.shots <= 0 or state.facing is None:
        return None
    step = DIRECTIONS[state.facing]
    target = (state.lolo[0] + step[0], state.lolo[1] + step[1])
    if not inside(target):
        return None
    alive = state.alive
    if target in state.eggs:
        eggs = frozenset(state.eggs - {target})
    elif target in alive:
        eggs, alive = frozenset(state.eggs | {target}), frozenset(alive - {target})
    else:
        return None
    return _settle(state, state.lolo, state.hearts, state.framers, eggs, alive, state.sunk,
                   state.shots - 1, state.facing)


def _settle(state, lolo, hearts, framers, eggs, alive, sunk, shots, facing):
    """Build the successor, and decide whether Lolo survived arriving in it.

    A Medusa's line is checked here rather than inside `move`, because a push can open one:
    shoving the Emerald Framer that was shielding him out of the way kills Lolo just as surely
    as walking into the line himself.
    """
    # A sunken egg is a bridge that works once: it stays until Lolo has crossed it, and floats
    # away the moment he steps off.
    if lolo != state.lolo and state.lolo in sunk:
        sunk = frozenset(sunk - {state.lolo})
    dead = blocked_by_medusa(state.room, framers, eggs, hearts, alive, lolo)
    return LoloState(state.room, lolo, hearts, framers, eggs, alive, sunk, shots, facing,
                     state.depth + 1, dead)


class LoloGame(Environment):
    """Adventures of Lolo, implemented rather than emulated. Needs nothing installed."""

    def __init__(self, magic_shots=0):
        super().__init__("lolo")
        #: Magic shots Lolo starts a room with. Zero, like the cartridge on a cold boot: the
        #: meter belongs to the player, not to the room, and on a real playthrough whatever
        #: was left over from the room before comes with you. A few rooms (int 1-5 is one)
        #: need a shot they cannot earn in-room and can only be cleared with this set.
        #: `lolo_gb.LoloGBEnv` takes the same argument and means the same thing by it.
        self.magic_shots = magic_shots
        self.index = 0
        self.room = None
        self.state = None
        self.state_history = []
        #: Rooms are immutable and states compare their room by identity, so a room reached by
        #: two different `reset` calls has to be the same object or nothing found in one run
        #: would ever equal anything found in another.
        self._rooms = {}

    def fix_index(self, index):
        if not 0 <= index < len(ROOMS):
            raise IndexError(
                f"Invalid index: {index}. There are {len(ROOMS)} rooms, so the index must be "
                f"0-{len(ROOMS) - 1}.")
        self.index = index

    def reset(self):
        self.room = self._rooms.setdefault(self.index, Room(self.index, ROOMS[self.index]))
        # Lolo starts facing nowhere: the cartridge will not fire a shot before the first
        # move, and neither will this. He can, however, start dead: a room that puts him in a
        # Medusa's line kills him before he has pressed anything, which the cartridge does too.
        start_alive = frozenset(self.room.enemies)
        self.state = LoloState(
            self.room, self.room.start, self.room.start_hearts, self.room.start_framers,
            frozenset(), start_alive, frozenset(), self.magic_shots, None,
            dead=blocked_by_medusa(self.room, self.room.start_framers, frozenset(),
                                   self.room.start_hearts, start_alive, self.room.start))
        self.state_history = [self.state]
        return self.state, {"room_index": self.index,
                            "room": self.room.label,
                            "hearts": self.state.hearts_left,
                            "shots": self.state.shots,
                            "door": self.room.door,
                            "start": self.room.start,
                            "exact": self.room.exact,
                            "unmodelled_enemies": tuple(sorted(self.room.unmodelled))}

    def is_goal(self, state):
        return state.solved

    def is_terminal(self, state):
        """Lolo walked into a Medusa's line. There is nothing left to plan for."""
        return state.dead

    def successors(self, state):
        successors = []
        if self.is_goal(state) or self.is_terminal(state):
            return successors
        for name in DIRECTIONS:
            successor = move(state, name)
            if successor is not None:
                successors.append((LoloAction(name), successor))
        successor = shoot(state)
        if successor is not None:
            successors.append((LoloAction(SHOOT), successor))
        return successors

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        name = action.name if isinstance(action, LoloAction) else str(action)
        successor = shoot(state) if name == SHOOT else move(state, name)
        return state if successor is None else successor

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.hearts_left
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.hearts_left

    def get_actions(self):
        return [LoloAction(name) for name in DIRECTIONS] + [LoloAction(SHOOT)]

    def render(self):
        rendered = [str(state) for state in self.state_history]
        for step_index, text in enumerate(rendered):
            print(f"Step: {step_index}")
            print(text)
            print("--------------")
        return rendered


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print an Adventures of Lolo room.")
    parser.add_argument("--room", type=int, default=None,
                        help=f"room index, 0-{len(ROOMS) - 1}")
    args = parser.parse_args()
    indices = range(len(ROOMS)) if args.room is None else [args.room]
    for index in indices:
        game = LoloGame()
        game.fix_index(index)
        state, info = game.reset()
        flag = "" if info["exact"] else f"  (unmodelled: {' '.join(info['unmodelled_enemies'])})"
        print(f"--- {index:3d} {info['room']}  {info['hearts']} hearts{flag}")
        for line in str(state).split("\n"):
            print(f"  |{line}|")
