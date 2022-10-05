import matplotlib.pyplot as plt

cmap10 = plt.get_cmap('tab10')
cmap20 = plt.get_cmap('tab20')

def color_fade(x, factor=0.5):
    if len(x) == 3:
        r,g,b = x
        a = 1.0
    else:
        r,g,b,a = x
    r = (1*factor+(1-factor)*r)
    g = (1*factor+(1-factor)*g)
    b = (1*factor+(1-factor)*b)
    return (r, g, b, a)

GAME_GENRE = {
    'alien': "Maze",
    'amidar': "Maze",
    'assault': "Fixed shooter",
    'asterix': "Action",
    'asteroids': "Multidirectional shooter",
    'atlantis': "Fixed shooter",
    'bankheist': "Maze",
    'battlezone': "First-person shooter",
    'beamrider': "Fixed shooter",
    'berzerk': "Multidirectional shooter",
    'bowling': "Sports",
    'boxing': "Sports",
    'breakout': "Action",
    'centipede': "Fixed shooter",
    'choppercommand': "Scrolling shooter",
    'crazyclimber': "Climbing",
    'defender': "Scrolling shooter",
    'demonattack': "Fixed shooter",
    'doubledunk': "Sports",
    'enduro': "Sports", # racing
    'fishingderby': "Sports",
    'freeway': "Action",
    'frostbite': "Action",
    'gopher': "Action", # genre is hard to clasifiy
    'gravitar': "Multidirectional shooter",
    'hero': "Action",
    'icehockey': "Sports",
    'jamesbond': "Scrolling shooter",
    'kangaroo': "Platform",
    'krull': "Action",
    'kungfumaster': "Beat 'em up",
    'montezumarevenge': "Platform",
    'mspacman': "Maze",
    'namethisgame': "Action",
    'phoenix': "Fixed shooter",
    'pitfall': "Platform",
    'pong': "Sports",
    'privateeye': "Action",
    'qbert': "Action",
    'riverraid': "Scrolling shooter",
    'roadrunner': "Racing",
    'robotank': "First-person shooter", # Wiki says Shoot 'em up, but it's clearly the same genre as battlezone.
    'seaquest': "Shoot 'em up",
    'skiing': "Sports",
    'solaris': "Space combat simulator",
    'spaceinvaders': "Shoot 'em up",
    'stargunner': "Scrolling shooter",
    'surround': "Action",
    'tennis': "Sports",
    'timepilot': "Multidirectional shooter",
    'tutankham': "Maze", # action-adventure / maze-shooter
    'upndown': "Racing",
    'venture': "Action",
    'videopinball': "Pinball",
    'wizardofwor': "Maze",
    'yarsrevenge': "Multidirectional shooter",
    'zaxxon': "Scrolling shooter",
}

GENRE_COLOR = {
    'Platform': cmap20(0),
    'Maze': cmap20(1),

    'Racing': cmap20(2),
    'Sports': cmap20(3),

    'Pinball': cmap20(4),
    'Climbing': cmap20(5),

    "Shoot 'em up": cmap20(6),
    "Beat 'em up": cmap20(7),

    'Fixed shooter': cmap20(8),
    'Scrolling shooter': cmap20(9),
    'First-person shooter': cmap20(10),
    'Multidirectional shooter': cmap20(11),

    'Space combat simulator': cmap20(12),
    'Action': cmap20(13),
}

GENRE_TO_CATEGORY = {
    'Platform': 'Maze',
    'Maze': 'Maze',

    'Racing': 'Sports',
    'Sports': 'Sports',

    'Pinball': 'Other',
    'Climbing': 'Other',

    "Shoot 'em up": 'Combat',
    "Beat 'em up": 'Combat',

    'Fixed shooter': 'Combat',
    'Scrolling shooter': 'Combat',
    'First-person shooter': 'Combat',
    'Multidirectional shooter': 'Combat',
    'Space combat simulator': 'Combat',

    'Action': 'Action',
}

CATEGORY_COLOR = {
    'Combat': cmap10(1),
    'Action': cmap10(0),
    'Sports': cmap10(2),
    'Maze': cmap10(3),
    'Other': cmap10(7),  # gray
}

CATEGORY_HATCH = {
    'Combat': "+++",
    'Action': "|||",
    'Sports': "---",
    'Maze': "...",
    'Other': "xxx",
}

# these are the names of the games in the standard 57-game ALE benchmark.
canonical_57 = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "Bank Heist",
    "Battle Zone",
    "Beam Rider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Centipede",
    "Chopper Command",
    "Crazy Climber",
    "Defender",
    "Demon Attack",
    "Double Dunk",
    "Enduro",
    "Fishing Derby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "Ice Hockey",
    "James Bond",
    "Kangaroo",
    "Krull",
    "Kung Fu Master",
    "Montezuma Revenge",
    "Ms Pacman",
    "Name This Game",
    "Phoenix",
    "Pitfall",
    "Pong",
    "Private Eye",
    "QBert",
    "Riverraid",
    "Road Runner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "Space Invaders",
    "Star Gunner",
    "Surround",
    "Tennis",
    "Time Pilot",
    "Tutankham",
    "Up n Down",
    "Venture",
    "Video Pinball",
    "Wizard of Wor",
    "Yars Revenge",
    "Zaxxon"
]
printable_57 = canonical_57.copy()
canonical_57 = ["".join(x.split(" ")).lower() for x in canonical_57]

for k, v in CATEGORY_COLOR.items():
    CATEGORY_COLOR[k] = color_fade(v, 0.33)


def clean_name(game):
    """
    Converts from a print name to a lowercase no spaces name. E.g. "Space Invaders" -> "spaceinvaders"
    """
    return "".join(c for c in game.lower() if c in "abcdefghijklmnopqrstuvwxyz")


def print_name(game):
    """
    Converts from a clean name to a print. E.g. "spaceinvaders" -> "Space Invaders"
    """
    game = clean_name(game)  # standarsize input
    for canonical in printable_57:
        if clean_name(canonical) == game:
            return canonical
    return "Unknown"


def get_game_genre_color(game):
    # look up the genre
    genre = GAME_GENRE.get(clean_name(game), "none")
    if genre in GENRE_COLOR:
        return GENRE_COLOR[genre]
    else:
        print(f"no color for game {game} genre {genre}")
        return (0.8, 0.8, 0.8)


def get_game_category_color(game):
    # look up the genre
    genre = GAME_GENRE.get(clean_name(game), "none")
    category = GENRE_TO_CATEGORY[genre]
    return CATEGORY_COLOR[category]

def get_game_category_hatch(game):
    # look up the genre
    genre = GAME_GENRE.get(clean_name(game), "none")
    category = GENRE_TO_CATEGORY[genre]
    return CATEGORY_HATCH[category]
