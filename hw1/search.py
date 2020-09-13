# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def search_with_structure(problem, data):
    """
    runs search algorithm(dfs or bfs) according to given data structure.
    continues searching in while loop until goal state is not reached.
    in while loop, for current position, adds successor states in provided data structure.
    """
    visited = set()
    data.push((problem.getStartState(), []))
    while not data.isEmpty():
        (cur_state, past_actions) = data.pop()
        if cur_state not in visited:
            visited.add(cur_state)
            if problem.isGoalState(cur_state):
                return past_actions
            for (next_state, action, cost) in problem.getSuccessors(cur_state):
                data.push((next_state, past_actions + [action]))

    return []

def depthFirstSearch(problem):
    """
    Searching path from start to the goal state using depth first search algorithm.
    to return: list of actions of legal moves which gets pacman from the start to the end state.
    about the algorithm: on every step checks if it's a goal state, if not continues searching in depth,
    meaning that pacman continues the route, if at some point there no more way to continue and the goal state
    is not yet found, pacman continues searching on the other path that is different at the deepest point from already
    checked way, meaning that first pacman checks most similar paths from start and then checks other ways changing
    from the start soon.
    for this problem stack is appropriate data structure according to depth first search algorithm, it's
    similar to recursion idea.
    """

    stack = util.Stack()

    return search_with_structure(problem, stack)


def breadthFirstSearch(problem):
    """
    Searching path from start to the goal state using breadth first search algorithm.
    to return: list of actions of legal moves which gets pacman from the start to the end state.
    about the algorithm: on every step checks if it's a goal state, if not continues searching on the same depth,
    meaning that pacman checks every direction the same depth, if at some point there no more way to continue and the goal state
    is not yet found, pacman continues searching on the next depth and check every node for that depth.
    for this problem queue is appropriate data structure according to breadth first search algorithm, because what first
    is added, it is checked first.
    """

    queue = util.Queue()
    return search_with_structure(problem, queue)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def pqueue_search(problem, data, heuristic = nullHeuristic):
    """
        runs search algorithm(UCS or A*) according to heuristic.
        given data structure: priority queue
        continues searching in while loop until goal state is not reached.
        in while loop, for current position, adds successor states in provided data structure
        with priority of total cost or total cost + heuristic(if it is given).
        """
    visited = set()
    data.push((problem.getStartState(), [], heuristic(problem.getStartState(), problem)),
              heuristic(problem.getStartState(), problem))
    while not data.isEmpty():
        (cur_state, past_actions, past_cost) = data.pop()
        if cur_state not in visited:
            visited.add(cur_state)
            if problem.isGoalState(cur_state):
                return past_actions
            for (next_state, action, cur_cost) in problem.getSuccessors(cur_state):
                data.push((next_state, past_actions + [action], past_cost + cur_cost),
                          past_cost + cur_cost + heuristic(next_state, problem))

    return []

def uniformCostSearch(problem):
    """
        Search way from the start to the goal state
        returns list of actions of legal moves which gets pacman from the start to the end state.
        priority queue is chosen for data structure, according to uniform cost search:
        costs is known(priorities for our data structure), first is checked the node with the lowest cost, priority for
        the aforementioned data structure.
        It is guaranteed that the path with the lowest cost will be found.
    """
    data = util.PriorityQueue()
    return pqueue_search(problem, data)

def aStarSearch(problem, heuristic=nullHeuristic):
    """
        Search the node that has the lowest combined cost and heuristic first.
        returns list of actions of legal moves which gets pacman from the start to the end state.
        A* search is similar to uniform cost search, but this time heuristics is added: at the start we, problem stater
        describes idea how close or far nodes can be from the goal, in other words we know estimated cost from every
        node till the goal. that's how we skip some paths, in many situations our solution will be close or as good as
        the best solution with the lowest cost with uniform cost search, but sometimes A* can find pretty bad solutions,
        but in general A* will find acceptable solution in less time.
    """
    data = util.PriorityQueue()
    return pqueue_search(problem, data, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
