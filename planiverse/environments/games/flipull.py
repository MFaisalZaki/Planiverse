"""Flipull in pure Python: no ROM, no emulator, no dependencies.

The rules are implemented directly, the way [`puzznic`](puzznic.py) is, so this is a
dependency-free benchmark rather than a prediction of the original game.

## The rules, stated

The player stands to the right of a wall of blocks holding one of them, and moves up and down
the rows or throws. A throw sends the held block leftward along the player's row:

1. It **destroys** each block of its own type it meets, and keeps going.
2. The first block of a **different** type takes the thrown block's place, and comes back into
   the player's hand: a swap.
3. If the *very first* block it meets is a different type, **nothing happens at all**: the
   block flies out and comes back, and the position is unchanged.
4. A **wall turns the block downward**: it slides down the face of the wall it met, and rules
   1 to 3 apply to what it meets on the way down. The floor bounces it back into the hand,
   with whatever it destroyed on the way staying destroyed. So a throw from above the wall of
   blocks reaches the top of the far column, and a throw that empties its row carries on down
   the far side.
5. Every destroyed cell **collapses its column**: everything above it falls one row.

A stage is cleared when few enough blocks are left.

Rule 3 is the one that makes this a puzzle rather than a shuffling exercise. Without it every
throw would be legal and the board would be a permutation group; with it, most rows are
refused most of the time and choosing which row to stand on is the whole game.

## How faithful is this to the cartridge?

Rules 1 to 3 and 5 were derived by driving Flipull and predicting what it would do, and they
reproduce it exactly (field and hand, cell for cell) for throws taken level with the wall in
the positions checked. Rule 4 is the original's, documented for the arcade *Plotting* the
cartridge ports: a block that reaches the back wall slides straight down it, as it does off
the ceiling, and one that reaches the floor bounces back to the player
(https://gamefaqs.gamespot.com/arcade/584111-plotting/faqs/41573;
https://en.wikipedia.org/wiki/Plotting_(video_game)). It is what an earlier automated
comparison was missing: the throws it disagreed on were those whose row was empty or emptied,
which this module used to refuse or stop and the cartridge carries down the far side. The
cartridge's own stages also carry a staircase of fixed bricks and, later, pipes; a `#` inside
the board deflects a block here the way the back wall does, which is what the staircase does,
but the bundled stages carry none and pipes are not modelled.

## What a stated rule set buys

Because the rules are known here, `is_terminal` is **exact**: a position is a dead end when no
throw from any row would connect. An emulator cannot compute that (it does not know what a
throw hits) and can only tell you the clock ran out. Dead-end detection is most of what makes
a puzzle searchable, so this is not a small difference.

## Where the stages came from

The first 32 stages replicate the cartridge's own stage table: stage for stage, the same
board size and the same CLEAR target as Flipull. The arrangements are generated rather
than copied, because the cartridge has no canonical arrangements to copy: it draws each
stage's block layout from an RNG seeded by boot timing, and its ROM stores only the block
total and the CLEAR target per stage. (Under the earlier rule set, without rule 4, most
arrangements the cartridge happened to draw could not reach their targets at all, 26 of 32
in one draw, which is what showed a throw mechanic was missing.) So each board here is
`generate_instance(seed)` for the seed beside it: drawn at random and kept when the
cartridge's target can be reached, with the plan that reached it stored beside the stage.
`tests/test_flipull.py` re-derives a solution for every one of them, so a stage
whose goal drifts out of reach fails the suite rather than quietly wasting a planner's
budget.

## Generating stages

`generate_instance(seed, ...)` draws stages the same way the bundled ones were made: a
random wall of blocks with the size and CLEAR target of one of the cartridge's stages, kept
when the target can be reached, which is what the cartridge's own stages are. Given a size
of the caller's own and no target, the wall is explored exhaustively and the fewest blocks
it can be reduced to becomes the target, so a generated stage is always clearable.
"""
from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, from_profile, rng, solvable_draw

#: `1`-`4` are block types, `#` is wall, and a space is empty. A wall inside the board turns
#: a thrown block downward as the back wall does, which is what the cartridge's staircase of
#: fixed bricks does; the bundled stages carry none.
WALL, EMPTY = "#", " "
BLOCK_TYPES = ("1", "2", "3", "4")

#: `(stage, clear_target)`, matching the cartridge's own 32-entry stage table: the same
#: board size (25, 30 or 36 blocks) and the same CLEAR target (9 down to 6) as each stage
#: of Flipull. The arrangements are this module's own, because the cartridge has
#: none to copy: it draws each stage's arrangement from an RNG seeded by boot timing, so
#: there is no canonical layout per stage, only a contract. Each board here is
#: `generate_instance(seed)` for the seed in its comment: drawn at random and explored
#: exhaustively until the target is reached, and kept when it is, which is what the
#: cartridge's own stages are: a random layout against a fixed target.
#:
#: The player starts on the bottom row of the wall, as on the cartridge: the position where
#: `down` does nothing.
STAGES = (
    # seed 5040; 363 positions explored, 35-move plan
    ("#######\n#     #\n#41321#\n#12423#\n#33431#\n#24231#\n#42334#\n#######", 9),
    # seed 5071; 302 positions explored, 42-move plan
    ("#######\n#     #\n#41332#\n#21242#\n#23142#\n#33343#\n#31144#\n#######", 9),
    # seed 5036; 243 positions explored, 30-move plan
    ("#######\n#     #\n#24434#\n#41431#\n#13141#\n#21432#\n#34432#\n#######", 8),
    # seed 5008; 6855 positions explored, 42-move plan
    ("#######\n#     #\n#11314#\n#21333#\n#41222#\n#13124#\n#34112#\n#22443#\n#######", 8),
    # seed 5012; 3666 positions explored, 46-move plan
    ("#######\n#     #\n#13112#\n#22212#\n#31443#\n#34111#\n#24421#\n#42224#\n#######", 8),
    # seed 5002; 3092 positions explored, 52-move plan
    ("#######\n#     #\n#11443#\n#12323#\n#23133#\n#11411#\n#11442#\n#32443#\n#######", 7),
    # seed 5010; 1529 positions explored, 57-move plan
    ("#######\n#     #\n#14223#\n#41344#\n#23234#\n#23121#\n#14331#\n#43232#\n#######", 7),
    # seed 5000; 2893 positions explored, 54-move plan
    ("########\n#      #\n#344141#\n#343114#\n#412334#\n#211232#\n#221411#\n#323444#\n########", 7),
    # seed 5003; 644 positions explored, 56-move plan
    ("########\n#      #\n#434444#\n#223442#\n#132423#\n#412222#\n#341212#\n#213143#\n########", 8),
    # seed 5006; 13435 positions explored, 67-move plan
    ("########\n#      #\n#212232#\n#131341#\n#432131#\n#241241#\n#324432#\n#234424#\n########", 8),
    # seed 5007; 3251 positions explored, 64-move plan
    ("########\n#      #\n#334123#\n#323123#\n#242234#\n#331243#\n#142412#\n#332324#\n########", 8),
    # seed 5011; 3302 positions explored, 36-move plan
    ("#######\n#     #\n#21142#\n#13311#\n#41311#\n#11144#\n#41441#\n#44321#\n#######", 7),
    # seed 5013; 656 positions explored, 54-move plan
    ("#######\n#     #\n#44434#\n#41241#\n#12131#\n#31111#\n#41233#\n#43221#\n#######", 7),
    # seed 5005; 6008 positions explored, 67-move plan
    ("########\n#      #\n#212443#\n#321211#\n#211224#\n#232243#\n#333321#\n#214231#\n########", 7),
    # seed 5018; 17319 positions explored, 53-move plan
    ("########\n#      #\n#231212#\n#413224#\n#432242#\n#113111#\n#324113#\n#114124#\n########", 7),
    # seed 5026; 3219 positions explored, 45-move plan
    ("#######\n#     #\n#34314#\n#31241#\n#32312#\n#34413#\n#12443#\n#34121#\n#######", 7),
    # seed 5029; 6889 positions explored, 50-move plan
    ("#######\n#     #\n#44342#\n#23441#\n#13133#\n#44341#\n#32323#\n#12112#\n#######", 7),
    # seed 5030; 8031 positions explored, 42-move plan
    ("#######\n#     #\n#24141#\n#31232#\n#44224#\n#14243#\n#34322#\n#43134#\n#######", 7),
    # seed 5035; 4849 positions explored, 40-move plan
    ("#######\n#     #\n#31423#\n#44334#\n#34143#\n#22111#\n#23222#\n#12343#\n#######", 7),
    # seed 5037; 9141 positions explored, 48-move plan
    ("#######\n#     #\n#22323#\n#44232#\n#34114#\n#24331#\n#33224#\n#13132#\n#######", 7),
    # seed 5038; 10489 positions explored, 41-move plan
    ("#######\n#     #\n#44234#\n#24322#\n#12322#\n#24134#\n#23343#\n#42433#\n#######", 7),
    # seed 5004; 8585 positions explored, 48-move plan
    ("#######\n#     #\n#33243#\n#12121#\n#12134#\n#23421#\n#13312#\n#12411#\n#######", 6),
    # seed 5017; 7895 positions explored, 48-move plan
    ("#######\n#     #\n#21431#\n#21324#\n#41132#\n#32114#\n#31132#\n#42342#\n#######", 6),
    # seed 5001; 11221 positions explored, 52-move plan
    ("########\n#      #\n#134334#\n#212334#\n#341432#\n#414333#\n#433244#\n#234132#\n########", 6),
    # seed 5015; 4269 positions explored, 69-move plan
    ("########\n#      #\n#414111#\n#431124#\n#242223#\n#212214#\n#312324#\n#123324#\n########", 6),
    # seed 5020; 9748 positions explored, 61-move plan
    ("########\n#      #\n#311232#\n#442324#\n#441112#\n#223232#\n#432413#\n#443234#\n########", 6),
    # seed 5022; 26101 positions explored, 61-move plan
    ("########\n#      #\n#333221#\n#424242#\n#331114#\n#312434#\n#114213#\n#321234#\n########", 6),
    # seed 5031; 12054 positions explored, 58-move plan
    ("########\n#      #\n#134321#\n#312413#\n#422422#\n#434322#\n#441334#\n#234411#\n########", 6),
    # seed 5048; 23116 positions explored, 63-move plan
    ("########\n#      #\n#134123#\n#142324#\n#332113#\n#221233#\n#322223#\n#324114#\n########", 6),
    # seed 5024; 1334 positions explored, 39-move plan
    ("#######\n#     #\n#44113#\n#24234#\n#31432#\n#21332#\n#42241#\n#######", 6),
    # seed 5034; 568 positions explored, 40-move plan
    ("#######\n#     #\n#14331#\n#21244#\n#31322#\n#34131#\n#24441#\n#######", 6),
    # seed 5056; 290 positions explored, 45-move plan
    ("#######\n#     #\n#34122#\n#44232#\n#41414#\n#14124#\n#12211#\n#######", 6),
)


#: Stages the generator drew, kept after the cartridge's so that `set_index` offers them
#: too: `(stage, clear_target)` pairs like `STAGES`, each `generate_instance(seed)` for the
#: seed in its comment, with the size and target of one of the cartridge's stages.
GENERATED_STAGES = (
    # seed 5009; 20001 positions explored, 52-move plan
    ("########\n#      #\n#331412#\n#343114#\n#341121#\n#131413#\n#114231#\n#242312#\n########", 8),
    # seed 5014; 1823 positions explored, 57-move plan
    ("########\n#      #\n#243243#\n#114432#\n#314244#\n#214413#\n#344334#\n#113213#\n########", 8),
    # seed 5016; 9973 positions explored, 57-move plan
    ("########\n#      #\n#334113#\n#131223#\n#313344#\n#234314#\n#224213#\n#142231#\n########", 8),
    # seed 5019; 6056 positions explored, 53-move plan
    ("########\n#      #\n#313331#\n#331434#\n#234313#\n#123123#\n#143314#\n#342214#\n########", 7),
    # seed 5021; 14233 positions explored, 49-move plan
    ("#######\n#     #\n#12323#\n#42141#\n#33413#\n#12324#\n#41342#\n#31143#\n#######", 6),
    # seed 5023; 2123 positions explored, 38-move plan
    ("#######\n#     #\n#42433#\n#43221#\n#14322#\n#21341#\n#22221#\n#13344#\n#######", 8),
    # seed 5025; 8094 positions explored, 51-move plan
    ("#######\n#     #\n#14213#\n#31133#\n#33434#\n#23224#\n#31243#\n#41224#\n#######", 6),
    # seed 5027; 8087 positions explored, 53-move plan
    ("########\n#      #\n#331244#\n#121312#\n#314142#\n#422224#\n#412243#\n#232332#\n########", 8),
    # seed 5028; 625 positions explored, 39-move plan
    ("#######\n#     #\n#32122#\n#44143#\n#42213#\n#32214#\n#33442#\n#23133#\n#######", 8),
    # seed 5032; 28861 positions explored, 56-move plan
    ("########\n#      #\n#431112#\n#144321#\n#413443#\n#431233#\n#213141#\n#234132#\n########", 7),
    # seed 5033; 4642 positions explored, 39-move plan
    ("#######\n#     #\n#24431#\n#22114#\n#14441#\n#14334#\n#11441#\n#33324#\n#######", 8),
    # seed 5039; 5276 positions explored, 57-move plan
    ("#######\n#     #\n#23431#\n#44341#\n#41113#\n#21243#\n#21234#\n#42421#\n#######", 7),
    # seed 5041; 17740 positions explored, 43-move plan
    ("#######\n#     #\n#43232#\n#44243#\n#34423#\n#31134#\n#22123#\n#21324#\n#######", 7),
    # seed 5042; 11650 positions explored, 66-move plan
    ("########\n#      #\n#121413#\n#322132#\n#432221#\n#422333#\n#344341#\n#224332#\n########", 7),
    # seed 5043; 12814 positions explored, 52-move plan
    ("#######\n#     #\n#41214#\n#13433#\n#43341#\n#12422#\n#22432#\n#43134#\n#######", 6),
    # seed 5044; 507 positions explored, 57-move plan
    ("#######\n#     #\n#43221#\n#32132#\n#43112#\n#21123#\n#11314#\n#23234#\n#######", 7),
    # seed 5045; 95844 positions explored, 57-move plan
    ("########\n#      #\n#124142#\n#211313#\n#313231#\n#431333#\n#434243#\n#441113#\n########", 7),
    # seed 5046; 841 positions explored, 32-move plan
    ("#######\n#     #\n#24241#\n#42441#\n#34112#\n#11323#\n#13312#\n#######", 8),
    # seed 5047; 13098 positions explored, 53-move plan
    ("########\n#      #\n#333412#\n#244214#\n#233334#\n#132133#\n#222131#\n#421412#\n########", 7),
    # seed 5049; 1403 positions explored, 46-move plan
    ("#######\n#     #\n#34221#\n#21434#\n#12321#\n#13114#\n#11311#\n#43322#\n#######", 8),
    # seed 5050; 4419 positions explored, 42-move plan
    ("#######\n#     #\n#14423#\n#33133#\n#33432#\n#32323#\n#33112#\n#34231#\n#######", 8),
    # seed 5051; 3163 positions explored, 54-move plan
    ("########\n#      #\n#342411#\n#421122#\n#412424#\n#333131#\n#143431#\n#421442#\n########", 8),
    # seed 5052; 19086 positions explored, 60-move plan
    ("########\n#      #\n#424332#\n#122331#\n#322234#\n#212412#\n#423434#\n#131343#\n########", 6),
    # seed 5053; 8329 positions explored, 51-move plan
    ("########\n#      #\n#412411#\n#421413#\n#211422#\n#112142#\n#313341#\n#231142#\n########", 8),
    # seed 5054; 3870 positions explored, 54-move plan
    ("########\n#      #\n#433412#\n#232412#\n#414233#\n#333331#\n#133442#\n#421431#\n########", 7),
    # seed 5055; 2241 positions explored, 35-move plan
    ("#######\n#     #\n#14413#\n#22134#\n#23443#\n#41132#\n#33332#\n#13221#\n#######", 7),
    # seed 5057; 5147 positions explored, 36-move plan
    ("#######\n#     #\n#21342#\n#33121#\n#43341#\n#21344#\n#14122#\n#14112#\n#######", 7),
    # seed 5058; 1492 positions explored, 53-move plan
    ("#######\n#     #\n#22323#\n#44214#\n#43411#\n#41241#\n#34113#\n#31424#\n#######", 7),
    # seed 5059; 1596 positions explored, 46-move plan
    ("########\n#      #\n#432414#\n#133334#\n#432421#\n#121414#\n#242431#\n#433341#\n########", 7),
    # seed 5060; 2505 positions explored, 46-move plan
    ("########\n#      #\n#343113#\n#313441#\n#324214#\n#214344#\n#412444#\n#243333#\n########", 7),
    # seed 5061; 3255 positions explored, 57-move plan
    ("########\n#      #\n#233341#\n#323124#\n#333432#\n#214423#\n#112341#\n#214443#\n########", 8),
    # seed 5062; 10713 positions explored, 57-move plan
    ("########\n#      #\n#114444#\n#141121#\n#343113#\n#421131#\n#144222#\n#343243#\n########", 6),
    # seed 5063; 3445 positions explored, 46-move plan
    ("#######\n#     #\n#43131#\n#32213#\n#23134#\n#22131#\n#34322#\n#42124#\n#######", 7),
    # seed 5064; 17303 positions explored, 55-move plan
    ("########\n#      #\n#311124#\n#223442#\n#232211#\n#411422#\n#144344#\n#133224#\n########", 7),
    # seed 5065; 4491 positions explored, 54-move plan
    ("#######\n#     #\n#44241#\n#33213#\n#32232#\n#11311#\n#43141#\n#12433#\n#######", 7),
    # seed 5066; 5349 positions explored, 39-move plan
    ("#######\n#     #\n#23443#\n#21132#\n#41444#\n#42433#\n#14233#\n#32134#\n#######", 7),
    # seed 5067; 13245 positions explored, 50-move plan
    ("########\n#      #\n#433242#\n#124211#\n#122444#\n#311123#\n#221424#\n#232134#\n########", 7),
    # seed 5068; 6231 positions explored, 51-move plan
    ("#######\n#     #\n#32341#\n#41411#\n#21344#\n#43442#\n#13411#\n#14123#\n#######", 6),
    # seed 5069; 7207 positions explored, 51-move plan
    ("########\n#      #\n#331211#\n#333123#\n#424433#\n#211132#\n#222343#\n#241322#\n########", 7),
    # seed 5070; 6779 positions explored, 63-move plan
    ("########\n#      #\n#233113#\n#142432#\n#233143#\n#144331#\n#243342#\n#312432#\n########", 6),
    # seed 5072; 27840 positions explored, 54-move plan
    ("########\n#      #\n#243334#\n#123232#\n#341221#\n#444214#\n#122411#\n#314342#\n########", 7),
    # seed 5073; 1838 positions explored, 46-move plan
    ("#######\n#     #\n#34143#\n#12441#\n#41342#\n#13212#\n#24223#\n#22223#\n#######", 8),
    # seed 5074; 348 positions explored, 48-move plan
    ("#######\n#     #\n#41412#\n#12114#\n#34343#\n#21124#\n#21221#\n#######", 6),
    # seed 5075; 24621 positions explored, 57-move plan
    ("########\n#      #\n#432114#\n#241242#\n#232433#\n#242122#\n#221242#\n#322113#\n########", 8),
    # seed 5076; 1790 positions explored, 47-move plan
    ("#######\n#     #\n#41141#\n#13313#\n#42423#\n#13223#\n#42234#\n#24324#\n#######", 7),
    # seed 5077; 4276 positions explored, 38-move plan
    ("#######\n#     #\n#11314#\n#31333#\n#12414#\n#44232#\n#22414#\n#23411#\n#######", 7),
    # seed 5078; 3580 positions explored, 66-move plan
    ("########\n#      #\n#322111#\n#432222#\n#221444#\n#213232#\n#422431#\n#413423#\n########", 6),
    # seed 5079; 201 positions explored, 34-move plan
    ("#######\n#     #\n#24124#\n#14142#\n#41412#\n#31333#\n#42411#\n#######", 8),
    # seed 5080; 964 positions explored, 30-move plan
    ("#######\n#     #\n#42131#\n#11131#\n#32233#\n#32141#\n#42331#\n#######", 9),
    # seed 5081; 18620 positions explored, 71-move plan
    ("########\n#      #\n#444142#\n#323312#\n#231423#\n#222221#\n#433413#\n#222124#\n########", 6),
    # seed 5082; 813 positions explored, 41-move plan
    ("#######\n#     #\n#44114#\n#44131#\n#42413#\n#32242#\n#24243#\n#######", 6),
    # seed 5083; 4008 positions explored, 62-move plan
    ("########\n#      #\n#134433#\n#134324#\n#341313#\n#213141#\n#434123#\n#322333#\n########", 7),
    # seed 5084; 38963 positions explored, 58-move plan
    ("########\n#      #\n#434124#\n#211343#\n#221121#\n#333144#\n#211441#\n#142314#\n########", 6),
    # seed 5085; 32656 positions explored, 59-move plan
    ("########\n#      #\n#441321#\n#421342#\n#414224#\n#242211#\n#213313#\n#134421#\n########", 6),
    # seed 5086; 7375 positions explored, 46-move plan
    ("#######\n#     #\n#22143#\n#44431#\n#12133#\n#22444#\n#44432#\n#34312#\n#######", 6),
    # seed 5087; 1598 positions explored, 39-move plan
    ("#######\n#     #\n#33422#\n#21424#\n#32233#\n#41141#\n#43232#\n#######", 6),
    # seed 5088; 28182 positions explored, 57-move plan
    ("########\n#      #\n#134231#\n#234134#\n#234434#\n#334243#\n#232332#\n#441411#\n########", 6),
    # seed 5089; 17009 positions explored, 56-move plan
    ("########\n#      #\n#324342#\n#422232#\n#322422#\n#241444#\n#111223#\n#121211#\n########", 8),
    # seed 5090; 17884 positions explored, 69-move plan
    ("########\n#      #\n#411314#\n#324111#\n#412424#\n#414144#\n#242331#\n#434243#\n########", 6),
    # seed 5091; 1818 positions explored, 45-move plan
    ("#######\n#     #\n#22211#\n#23412#\n#14331#\n#43221#\n#32444#\n#22441#\n#######", 8),
    # seed 5092; 2078 positions explored, 40-move plan
    ("#######\n#     #\n#13241#\n#24244#\n#23343#\n#43323#\n#33222#\n#32334#\n#######", 7),
    # seed 5093; 2694 positions explored, 46-move plan
    ("#######\n#     #\n#13244#\n#41333#\n#13144#\n#13123#\n#44412#\n#14114#\n#######", 8),
    # seed 5094; 5698 positions explored, 46-move plan
    ("#######\n#     #\n#11122#\n#22333#\n#34213#\n#21122#\n#22142#\n#21313#\n#######", 7),
    # seed 5095; 30706 positions explored, 56-move plan
    ("########\n#      #\n#142314#\n#112422#\n#321443#\n#111441#\n#224412#\n#431212#\n########", 7),
    # seed 5096; 17280 positions explored, 53-move plan
    ("########\n#      #\n#443422#\n#313114#\n#443113#\n#131243#\n#144231#\n#321211#\n########", 6),
    # seed 5097; 608 positions explored, 33-move plan
    ("#######\n#     #\n#32114#\n#32332#\n#43342#\n#11422#\n#22123#\n#######", 6),
    # seed 5098; 1198 positions explored, 46-move plan
    ("#######\n#     #\n#13114#\n#43423#\n#13412#\n#21331#\n#24441#\n#######", 6),
    # seed 5099; 48662 positions explored, 56-move plan
    ("########\n#      #\n#242432#\n#131424#\n#244231#\n#321212#\n#322141#\n#234221#\n########", 6),
)

#: Everything `set_index` selects between: the cartridge's stages, then the generated ones.
INSTANCES = STAGES + GENERATED_STAGES


def profile(stage):
    """The contract of a stage, as the options `generate_instance` takes to draw one like
    it: the wall's width and height in blocks, and its CLEAR target."""
    text, target = stage
    grid = parse_stage(text)
    rows = block_rows(grid)
    width = sum(1 for cell in grid[rows[0]] if cell in BLOCK_TYPES)
    return {"width": width, "height": len(rows), "clear_target": target}


def parse_stage(text):
    """An ASCII stage into a grid of single characters, padded to a rectangle."""
    rows = text.split("\n")
    width = max(len(row) for row in rows)
    return [list(row.ljust(width)) for row in rows]


def generate_stage(random_, width=5, height=5, types=4):
    """A random wall of `height` rows of `width` blocks, in the stage alphabet.

    The same shape as the bundled stages: a ring of walls, one empty row above the wall for
    the player to stand on, and every cell of the wall itself a block of one of `types`
    types. Whether it is worth playing is a separate question, answered by exploring it; see
    `fewest_blocks_reachable`.
    """
    kinds = BLOCK_TYPES[:types]
    rows = [WALL * (width + 2), WALL + EMPTY * width + WALL]
    rows += [WALL + "".join(random_.choice(kinds) for _ in range(width)) + WALL
             for _ in range(height)]
    rows.append(WALL * (width + 2))
    return "\n".join(rows)


def fewest_blocks_reachable(text, limit=200_000):
    """Explore every position reachable from a stage's opening, and report the fewest blocks
    any of them has left. Returns `(fewest, exhausted, plan, explored)`: `plan` reaches a
    position with that many, and `explored` is how many positions were seen doing it.

    Exhaustive rather than goal-directed, because the question is not "can the target be
    reached" but "what is the lowest target this board could honestly be given". `exhausted`
    is False when `limit` positions were seen first, in which case `fewest` is only a bound.
    """
    game = FlipullGame()
    game.set_instance((text, 0))          # a target of 0 means no position counts as cleared
    start, _ = game.reset()
    parents, frontier, best = {start: None}, [start], start
    while frontier:
        if len(parents) >= limit:
            return best.blocks_remaining, False, _plan_to(parents, best), len(parents)
        state = frontier.pop()
        for action, successor in game.successors(state):
            if successor in parents:
                continue
            parents[successor] = (state, action)
            if successor.blocks_remaining < best.blocks_remaining:
                best = successor
            frontier.append(successor)
    return best.blocks_remaining, True, _plan_to(parents, best), len(parents)


def _plan_to(parents, state):
    """Walk the parent links back from `state` to the opening position."""
    plan = []
    while parents[state] is not None:
        state, action = parents[state]
        plan.append(action)
    return plan[::-1]


def block_rows(grid):
    """Row indices that hold at least one block."""
    return [row for row, cells in enumerate(grid) if any(c in BLOCK_TYPES for c in cells)]


def playable_rows(grid):
    """Rows the player may stand on: everything inside the border."""
    return list(range(1, len(grid) - 1))


def count_blocks(grid):
    return sum(1 for row in grid for cell in row if cell in BLOCK_TYPES)


def throw(grid, row, held):
    """Apply a throw from `row` holding `held`. Returns `(grid, held)` or `None` for a no-op.

    The whole rule set, in one place, so that it can be read and argued with. The block
    flies leftward along the player's row; a wall turns it downward and it slides down the
    face of that wall until it meets a block or the floor.
    """
    if row is None or not 0 <= row < len(grid) or held not in BLOCK_TYPES:
        return None

    grid = [cells[:] for cells in grid]
    destroyed, new_held = [], held
    r, c, falling = row, len(grid[row]) - 2, False       # inside the player's wall, flying left
    while 0 <= r < len(grid) and 0 <= c < len(grid[r]):
        cell = grid[r][c]
        if cell == WALL:
            if falling:
                break                          # the floor: the block bounces back into the hand
            falling = True                     # a wall turns the block downward, from the cell
            r, c = r + 1, c + 1                # it just left
            continue
        if cell in BLOCK_TYPES and cell != held:
            if not destroyed:
                return None                    # a different type first: the throw is refused
            grid[r][c] = held                  # swap ours in and take theirs
            new_held = cell
            break
        if cell == held:
            destroyed.append((r, c))           # our own type: destroyed, and the block flies on
        r, c = (r + 1, c) if falling else (r, c - 1)

    if not destroyed:
        return None                            # it met nothing of its own type anywhere

    for r, c in destroyed:                     # in the order they were met, so a run down one
        collapse(grid, r, c)                   # column drops the stack above it by the run
    return grid, new_held


def collapse(grid, row, col):
    """Drop the blocks stacked above `(row, col)` by one, in place.

    Only *blocks* fall. The run stops at the first cell above that is not one (empty air or
    the border), so a block with a gap under it stays where it is and the walls stay where
    they are. Getting this wrong is quiet and ugly: an earlier version shifted whatever was
    above, which walked the top border down into the play area one throw at a time, and the
    board still looked plausible while it happened.
    """
    top = row
    while top - 1 >= 0 and grid[top - 1][col] in BLOCK_TYPES:
        grid[top][col] = grid[top - 1][col]
        top -= 1
    grid[top][col] = EMPTY



#: The contract of each cartridge stage: what `generate_instance` draws a stage's size and
#: target from when the caller leaves them unset.
PROFILES = tuple(profile(stage) for stage in STAGES)

class FlipullAction:
    """`up`, `down`, or `throw`."""

    def __init__(self, name):
        if name not in ("up", "down", "throw"):
            raise ValueError(f"unknown action: {name!r}")
        self.name = name

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, FlipullAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class FlipullState:
    """A board, a row the player stands on, and a block in hand."""

    def __init__(self, grid, row, held, clear_target, depth=0):
        self.grid = tuple(tuple(cells) for cells in grid)
        self.row = row
        self.held = held
        self.clear_target = clear_target
        self.depth = depth
        self.blocks_remaining = count_blocks(self.grid)

        literals = [f"at(block-{cell}, {r}, {c})"
                    for r, cells in enumerate(self.grid)
                    for c, cell in enumerate(cells) if cell in BLOCK_TYPES]
        literals.append(f"at(player, {row})")
        literals.append(f"holding(block-{held})")
        literals.append(f"remaining({self.blocks_remaining})")
        self.literals = frozenset(literals)

    def any_throw_connects(self):
        """Is there a row this player could stand on and throw from?

        What makes `is_terminal` exact here. The cartridge cannot answer this.
        """
        return any(throw([list(cells) for cells in self.grid], row, self.held) is not None
                   for row in playable_rows(self.grid))

    def __eq__(self, other):
        return (isinstance(other, FlipullState) and self.grid == other.grid
                and self.row == other.row and self.held == other.held)

    def __hash__(self):
        return hash((self.grid, self.row, self.held))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        lines = []
        for index, cells in enumerate(self.grid):
            marker = "<" if index == self.row else " "
            lines.append("".join(cells) + marker)
        lines.append(f"held: {self.held}   blocks: {self.blocks_remaining}"
                     f"/{self.clear_target}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"<FlipullState(row={self.row}, held={self.held}, "
                f"blocks={self.blocks_remaining}/{self.clear_target})>")


class FlipullGame(Environment):
    """Flipull, implemented rather than emulated. Needs nothing installed."""

    def __init__(self):
        super().__init__("flipull")
        self.index = 0
        #: The `(stage, clear_target)` pair `reset` builds: a bundled one after `set_index`,
        #: or whatever `set_instance` was given.
        self.instance = STAGES[0]
        #: The plan that reached the fewest blocks when `generate_instance` explored the
        #: current stage, and how many positions that exploration saw; None for a bundled or
        #: hand-made one.
        self.witness = None
        self.witness_expansions = None
        self.state = None
        self.state_history = []

    def set_index(self, index):
        if not 0 <= index < len(INSTANCES):
            raise IndexError(
                f"Invalid index: {index}. There are {len(INSTANCES)} stages, so the index "
                f"must be 0-{len(INSTANCES) - 1}.")
        self.index = index
        self.instance = INSTANCES[index]
        self.witness = self.witness_expansions = None

    def set_instance(self, instance):
        """Select a `(stage_text, clear_target)` pair, in the shape of the entries of `STAGES`."""
        text, target = instance
        if not block_rows(parse_stage(text)):
            raise ValueError("a stage needs at least one block")
        self.instance = (str(text), int(target))
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, width=None, height=None, types=4,
                          clear_target=None, max_target_fraction=0.4, search_limit=200_000,
                          attempts=100):
        """Draw a fresh stage, select it, and return it as a `[stage_text, clear_target]` pair.

        Each draw is a random `width` by `height` wall of `types` block types. With nothing
        set, a draw takes the size and the CLEAR target of one of the cartridge's 32 stages
        at random and is kept when that target can be reached, which is how the bundled
        stages were made and what the cartridge's own stages are: a random layout against a
        fixed target. With a `clear_target` of the caller's own, the same. With a `width`
        and `height` and no target, the draw is explored exhaustively (up to `search_limit`
        positions) and the fewest blocks it can be reduced to becomes the target, provided
        that is no more than `max_target_fraction` of the wall. The plan the draw was
        accepted on is left in `self.witness`.
        """
        random_, _ = rng(seed)

        def options():
            if width is None and height is None:
                return from_profile(random_, PROFILES, clear_target=clear_target)
            return {"width": width or 5, "height": height or 5, "clear_target": clear_target}

        if clear_target is None and not (width is None and height is None):
            found = {}

            def draw(attempt):
                chosen = options()
                found["blocks"] = chosen["width"] * chosen["height"]
                return generate_stage(random_, chosen["width"], chosen["height"], types)

            def accept(text):
                fewest, exhausted, plan, explored = fewest_blocks_reachable(text, search_limit)
                if not exhausted or fewest > max_target_fraction * found["blocks"]:
                    return False          # undecided within the budget, or not worth playing for
                found.update(target=int(fewest), plan=plan, explored=explored)
                return True

            text = draw_until(draw, accept, attempts, "Flipull stage")
            instance = [text, found["target"]]
            self.set_instance(instance)
            self.witness, self.witness_expansions = found["plan"], found["explored"]
            return instance

        def draw(attempt):
            chosen = options()
            return [generate_stage(random_, chosen["width"], chosen["height"], types),
                    int(chosen["clear_target"])]
        return solvable_draw(self, draw, attempts, search_limit, what="Flipull stage")

    def reset(self):
        text, target = self.instance
        grid = parse_stage(text)
        rows = block_rows(grid)
        row = rows[-1] if rows else len(grid) - 2
        # The opening hand is the type of the block the player would meet first from the
        # bottom row, so the first throw always connects and the stage opens with a move.
        held = next((grid[row][col] for col in range(len(grid[row]) - 1, -1, -1)
                     if grid[row][col] in BLOCK_TYPES), BLOCK_TYPES[0])
        self.state = FlipullState(grid, row, held, target)
        self.state_history = [self.state]
        return self.state, {"stage": self.index,
                            "generated": self.index is None,
                            "blocks": self.state.blocks_remaining,
                            "clear_target": target,
                            "rows": len(playable_rows(grid))}

    def is_goal(self, state):
        return state.blocks_remaining <= state.clear_target

    def is_terminal(self, state):
        """No throw from any row would connect, so the board can never change again.

        Exact, because the rules are known here. The Game Boy sibling cannot compute this
        (it does not know what a throw hits) and can only report that the clock ran out.
        """
        return not self.is_goal(state) and not state.any_throw_connects()

    def successors(self, state):
        successors = []
        if self.is_goal(state) or self.is_terminal(state):
            return successors
        for action in ("up", "down", "throw"):
            successor = self.__advance__(state, FlipullAction(action))
            if successor == state:
                continue
            successors.append((FlipullAction(action), successor))
        return successors

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        name = action.name if isinstance(action, FlipullAction) else str(action)

        if name in ("up", "down"):
            row = state.row + (-1 if name == "up" else 1)
            if row not in playable_rows(state.grid):
                return state                     # into the ceiling or the floor
            return FlipullState(state.grid, row, state.held, state.clear_target,
                                state.depth + 1)

        outcome = throw([list(cells) for cells in state.grid], state.row, state.held)
        if outcome is None:
            return state                         # the throw was refused
        grid, held = outcome
        return FlipullState(grid, state.row, held, state.clear_target, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.blocks_remaining
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.blocks_remaining

    def get_actions(self):
        return [FlipullAction(name) for name in ("up", "down", "throw")]

    def render(self):
        rendered = [str(state) for state in self.state_history]
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered
