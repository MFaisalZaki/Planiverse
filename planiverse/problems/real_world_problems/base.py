from planiverse.problems.base import Environment


class RealWorldProblem(Environment):
    def __init__(self, problem_name):
        self.name = problem_name
        self.state = None
