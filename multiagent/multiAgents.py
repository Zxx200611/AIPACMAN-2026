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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls=currentGameState.getWalls()
        newFoods=newFood.asList()

        que, vis, dis = util.Queue(), set(), {}
        que.push(newPos); vis.add(newPos); dis[newPos] = 0
        while not que.isEmpty():
            u = que.pop()
            for dx, dy in [(0,1),(0,-1),(-1,0),(1,0)]:
                tx, ty = u[0] + dx, u[1] + dy
                if ((tx,ty) not in vis) and (not walls[tx][ty]):
                    dis[(tx,ty)] = dis[u] + 1
                    vis.add((tx,ty))
                    que.push((tx,ty))

        minGhostDist=1000000000
        for gs in newGhostStates:
            gx, gy = int(gs.getPosition()[0]), int(gs.getPosition()[1])
            minGhostDist=min(minGhostDist,dis[(gx,gy)])

        scaredSum=0
        for x in newScaredTimes:
            scaredSum+=x

        minFoodDist=1000000000
        if not newFoods:
            minFoodDist=-minFoodDist
        for f in newFoods:
            minFoodDist=min(minFoodDist,dis[f])

        
        return successorGameState.getScore()+((-200/(minGhostDist+1) if minGhostDist<=7 else 0)-2*minFoodDist-30*len(newFoods)+2*scaredSum)

def scoreEvaluationFunction(currentGameState: GameState):
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
    def miniMaxSearch(self, gameState: GameState, d: int, m: int):
        if d == self.depth * m or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        uid = d % m
        las = gameState.getLegalActions(uid)
        if(uid == 0):
            res = -1000000000
            for action in las:
                vgs = gameState.generateSuccessor(uid, action)
                res = max(res, self.miniMaxSearch(vgs, d + 1, m))
        else:
            res = +1000000000
            for action in las:
                vgs = gameState.generateSuccessor(uid, action)
                res = min(res, self.miniMaxSearch(vgs, d + 1, m))
        return res
 
    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        las, m = gameState.getLegalActions(0), gameState.getNumAgents()
        mxp, act = -1000000000, las[0]
        for action in las:
            vgs = gameState.generateSuccessor(0, action)
            res = self.miniMaxSearch(vgs, 1, m)
            if res > mxp:
                mxp, act = res, action
        return act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBetaSearch(self, gameState: GameState, d: int, alpha: int, beta: int, m: int):
        # print("DFSing at ",gameState,", depth ",d,", alpha,beta = ",alpha,beta)
        if d == self.depth * m or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        uid = d % m
        las = gameState.getLegalActions(uid)
        if(uid == 0):
            res = -1000000000
            for action in las:
                vgs = gameState.generateSuccessor(uid, action)
                res = max(res, self.alphaBetaSearch(vgs, d + 1, alpha, beta, m))
                alpha = max(alpha, res)
                if alpha > beta: break
        else:
            res = +1000000000
            for action in las:
                vgs = gameState.generateSuccessor(uid, action)
                res = min(res, self.alphaBetaSearch(vgs, d + 1, alpha, beta, m))
                beta = min(beta, res)
                if alpha > beta: break
        return res

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        las, m = gameState.getLegalActions(0), gameState.getNumAgents()
        mxp, act, alpha = -1000000000, las[0], -1000000000
        for action in las:
            vgs = gameState.generateSuccessor(0, action)
            res = self.alphaBetaSearch(vgs, 1, alpha, +1000000000, m)
            if res > mxp:
                mxp, alpha, act = res, res, action
        return act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimaxSearch(self, gameState: GameState, d: int, m: int):
        if d == self.depth * m or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        uid = d % m
        las = gameState.getLegalActions(uid)
        if(uid == 0):
            res = -1000000000
            for action in las:
                vgs = gameState.generateSuccessor(uid, action)
                res = max(res, self.expectimaxSearch(vgs, d + 1, m))
        else:
            res = 0.0
            for action in las:
                vgs = gameState.generateSuccessor(uid, action)
                res += self.expectimaxSearch(vgs, d + 1, m)
            res /= len(las)
        return res
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        las, m = gameState.getLegalActions(0), gameState.getNumAgents()
        mxp, act = -1000000000, las[0]
        for action in las:
            vgs = gameState.generateSuccessor(0, action)
            res = self.expectimaxSearch(vgs, 1, m)
            if res > mxp:
                mxp, act = res, action
        return act

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    walls = currentGameState.getWalls()
    foods=food.asList()

    minGhostDist=1000000000
    for i in range(0,len(ghostStates)):
        gs=ghostStates[i]
        if not scaredTimes[i]:
            minGhostDist=min(minGhostDist,manhattanDistance(pos,gs.getPosition()))

    scaredSum=0
    for x in scaredTimes:
        scaredSum+=x

    minFoodDist=1000000000
    if not foods:
        minFoodDist=-minFoodDist
    for f in foods:
        minFoodDist=min(minFoodDist,manhattanDistance(pos,f))

    return currentGameState.getScore()+((-300/(minGhostDist+1) if minGhostDist<=4 else 0)-3*minFoodDist-400*len(foods)+4*scaredSum)


# Abbreviation
better = betterEvaluationFunction
