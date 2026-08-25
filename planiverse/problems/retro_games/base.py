from planiverse.problems.base import Environment


class RetroGame(Environment):
    def __init__(self, name, year):
        self.name = name
        self.year = year
