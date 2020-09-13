# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        newGhostPos = successorGameState.getGhostPositions()

        curFoodPos = currentGameState.getFood()
        # get higher score if pacman eats food, also prioritise according to closest food distance
        if len(curFoodPos.asList()) > len(newFood.asList()):
            score = 10000 + 1000/min( [10000] + [util.manhattanDistance(newPos, food) for food in newFood.asList()])
        else: # if pacman does not eat food, prioritise according to closest food distance
            score = 1000 / min( [10000] + [util.manhattanDistance(newPos, food) for food in newFood.asList()])

        ghostScore = 0
        # distance to the closest ghost
        closestGhost = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos])
        # pacman cares about ghost if it is dangerously close, otherwise, it keeps tracking according food
        if closestGhost < 2: ghostScore = 20000

        # return score according to components above, plus difference between game state scores
        return successorGameState.getScore() - currentGameState.getScore() + score - ghostScore


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # return max value for pacman according to minimax algorithm
        return self.maxi(gameState, self.depth, 0)

    # minimax algorithm max function, maximizes score for pacman, in the situation
    # when ghosts try to minimize our score.
    def maxi(self, state, depth, agentIndex):
        depth = depth - 1
        if depth < 0 or state.isWin() or state.isLose():
            # stop evaluating
            return self.evaluationFunction(state)
        maxValue = float("-inf")
        bestAction = Directions.STOP
        # try every action for pacman and check what ghosts do and what happens in
        # next depths. return best among them.
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            curValue = self.mini(successor, depth, agentIndex + 1)
            #take best action and value
            if curValue > maxValue:
                maxValue = curValue
                bestAction = action
        if depth + 1 != self.depth:
            # minimax needs value of action. in this section of code,
            # maxi is called recursively by mini
            return maxValue
        else:
            # it is our destination, return best action for given depth
            return bestAction


    # minimax algorithm min function, ghosts minimize pacman score.
    def mini(self, state, depth, agentIndex):
        # stop evaluating when game is finished
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        minValue = float("inf")
        # if it is last ghost, next move is taken by pacman
        if agentIndex >= state.getNumAgents() - 1:
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                # all the ghosts have already done action and continues minimax in next depth
                curValue = self.maxi(successor, depth, 0)
                if curValue < minValue:
                    minValue = curValue
            return minValue
        else:
            # the next player is also ghost
            # ghost tries every possible action and returns lowest score among them
            # according to minimax algorithm, that every player plays optimally for him.
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                curValue = self.mini(successor, depth, agentIndex + 1)
                #take best value for ghost (lowest)
                if curValue < minValue:
                    minValue = curValue
            return minValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxiPrun(gameState, self.depth, 0, float("-inf"), float("inf"))

    # alpha-beta prunning max function, maximizes score for pacman, in the situation
    # when ghosts try to minimize our score.
    def maxiPrun(self, state, depth, agentIndex, alpha, beta):
        depth = depth - 1
        if depth < 0 or state.isWin() or state.isLose():
            # stop evaluating
            return self.evaluationFunction(state)
        maxValue = float("-inf")
        bestAction = Directions.STOP
        # try every action for pacman and check what ghosts do and what happens in
        # next depths. return best among them.
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            curValue = self.miniPrun(successor, depth, agentIndex + 1, alpha, beta)
            #take best action and value
            if curValue > maxValue:
                maxValue = curValue
                bestAction = action
            # stops exploring when there is no sense doing it
            if maxValue > beta:
                return maxValue
            alpha = max(alpha, maxValue)
        if depth + 1 != self.depth:
            # alpha-beta prunning needs value of action. in this section of code,
            # maxi is called recursively by mini
            return maxValue
        else:
            # it is our destination, return best action for given depth
            return bestAction


    # alpha-beta prunning algorithm min function, ghosts minimize pacman score.
    def miniPrun(self, state, depth, agentIndex, alpha, beta):
        # stop evaluating when game is finished
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        minValue = float("inf")
        # if it is last ghost, next move is taken by pacman
        if agentIndex >= state.getNumAgents() - 1:
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                # all the ghosts have already done action and continues minimax in next depth
                curValue = self.maxiPrun(successor, depth, 0, alpha, beta)
                if curValue < minValue:
                    minValue = curValue
                # stops exploring when there is no sense doing it
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue)
            return minValue
        else:
            # the next player is also ghost
            # ghost tries every possible action and returns lowest score among them
            # according to minimax algorithm, that every player plays optimally for him.
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                curValue = self.miniPrun(successor, depth, agentIndex + 1, alpha, beta)
                #take best value for ghost (lowest)
                if curValue < minValue:
                    minValue = curValue
                # stops exploring when there is no sense doing it
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue)
            return minValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxi(gameState, self.depth, 0)

    def maxi(self, state, depth, agentIndex):
        depth = depth - 1
        if depth < 0 or state.isWin() or state.isLose():
            # stop evaluating
            return self.evaluationFunction(state)
        maxValue = float("-inf")
        bestAction = Directions.STOP
        # try every action for pacman and check what ghosts do and what happens in
        # next depths. return best among them.
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            curValue = self.expected(successor, depth, agentIndex + 1)
            #take best action and value
            if curValue > maxValue:
                maxValue = curValue
                bestAction = action
        if depth + 1 != self.depth:
            # maxi is called recursively by expected
            return maxValue
        else:
            # it is our destination, return best action for given depth
            return bestAction


    # expected value of action by given agent
    def expected(self, state, depth, agentIndex):
        # stop evaluating when game is finished
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalActions = state.getLegalActions(agentIndex)
        probability = 1.0/len(legalActions)
        expectation = 0
        # take score for every action and find expectation among them
        if agentIndex >= state.getNumAgents() - 1:
            # if it is last ghost, next move is taken by pacman
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                # all the ghosts have already done action and continues expectimax in next depth
                curValue = self.maxi(successor, depth, 0)
                # take score by
                expectation+=(probability*curValue)
            return expectation
        else:
            # the next player is also ghost
            # ghost tries every possible action and returns expectation score among them
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                curValue = self.expected(successor, depth, agentIndex + 1)
                expectation += (probability * curValue)
            return expectation

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I take score variable, every component is added later
                    Food number is checked and lower number has higher priority.
                    manhattan distance to closest food is also priority.
                    if ghost is dangerously close to pacman score drops to very low value.
                    if it is not dangerously close, being far from ghost is better
                    but doesn't have very high priority, because food is more important
                    I also take capsules into account: lower manhattan distance to closest capsule is better
    """

    curGhostStates = currentGameState.getGhostStates()
    curFoodPos = currentGameState.getFood()
    curPacPos = currentGameState.getPacmanPosition()
    # get higher score if pacman eats food, also prioritise according to closest food distance
    score = 400 if len(curFoodPos.asList()) == 0 else 200/len(curFoodPos.asList())
    score += 10.0 / min([10000] + [util.manhattanDistance(curPacPos, food) for food in curFoodPos.asList()])

    # distance to the closest ghost

    for ghost in curGhostStates:
        curGhostManh = util.manhattanDistance(ghost.getPosition(), curPacPos)
        if(ghost.scaredTimer > 1):
            score += 100.0/curGhostManh
        else:
            if curGhostManh < 2 :
                score -= 1000000
            else:
                score -= 5.0/curGhostManh

    curCapsules = currentGameState.getCapsules()
    closestCapsule = float(min([len(curCapsules)] + [util.manhattanDistance(powerPos, curPacPos) for powerPos in curCapsules]))
    if(closestCapsule != 0):
        score += 10/closestCapsule
    else: score += 10

    return currentGameState.getScore() + score


# Abbreviation
better = betterEvaluationFunction

