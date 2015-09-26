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
import random, util, sys

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores);
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
        score = 0
        newPos = successorGameState.getPacmanPosition()
        food = successorGameState.getFood()
        foodNum = successorGameState.getNumFood()
        #want to reduce the distance to the closest food
        score = score - 10*self.nearestFoodDistance(food,newPos)
        #want to reduce the number of food(to prevent pacman from staying close to food but not eat it)
        score = score - 1000*foodNum;
        newGhostStates = successorGameState.getGhostStates()
        for ghostState in newGhostStates:
            ghostPosition = ghostState.getPosition()
            distance = manhattanDistance(newPos,ghostPosition)
            #do not want to stay close to ghosts; if close, run away
            if distance < 3:
                score = score - 5000;
                score = score+distance*50
        return score

    def nearestFoodDistance(self,food,position):
        height, width = food.height, food.width
        minDistance = height+width
        minPosition = None
        for y in range(0,height):
            for x in range(0,width):
                if food[x][y] == True:
                    distance = manhattanDistance((x,y),position)
                    if distance < minDistance:
                        minPosition = (x,y)
                        minDistance = distance
        return minDistance


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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        (max_value, max_action) = self.evaluate(gameState,0,1);
        return max_action

    def evaluate(self,gameState,agentIndex,depth):
        #if the game ends
        if gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState),None)
        #if reach the bottom of depths
        if depth > self.depth:
            return (self.evaluationFunction(gameState),None)
        #find the next agent
        if agentIndex == gameState.getNumAgents() - 1:
            nextAgentIndex = 0;
        else:
            nextAgentIndex = agentIndex+1;
        #get all successor game states
        legal_actions = gameState.getLegalActions(agentIndex)
        #if pacman, find the max successor
        if agentIndex==0:
            max_value, max_action = -1000000, None
            for action in legal_actions:
                successor_state = gameState.generateSuccessor(agentIndex,action)
                (value,a) = self.evaluate(successor_state,nextAgentIndex,depth)
                if value > max_value:
                    max_value, max_action = value, action
            return (max_value, max_action)
        #if ghost, find the min successor
        else:
            min_value, min_action = 1000000, None
            #increase depth by 1 if a pile is finished
            if agentIndex == gameState.getNumAgents() - 1:
                depth += 1
            for action in legal_actions:
                successor_state = gameState.generateSuccessor(agentIndex,action)
                (value,a) = self.evaluate(successor_state,nextAgentIndex,depth)
                if value < min_value:
                    min_value, min_action = value, action
            return (min_value, min_action)


        



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = [-sys.maxint-1]
        beta = [sys.maxint]
        (max_value, max_action) = self.evaluate(gameState,0,1,alpha,beta);
        return max_action
       
    def evaluate(self,gameState,agentIndex,depth,alpha,beta):
        #if the game ends
        if gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState),None)
        #if reach the bottom of depths
        if depth > self.depth:
            return (self.evaluationFunction(gameState),None)
        #find the next agent
        if agentIndex == gameState.getNumAgents() - 1:
            nextAgentIndex = 0;
        else:
            nextAgentIndex = agentIndex+1;
        #get all successor game states
        legal_actions = gameState.getLegalActions(agentIndex)
        #if pacman, find the max successor
        if agentIndex==0:
            max_value, max_action = -sys.maxint-1, None
            for action in legal_actions:
                successor_state = gameState.generateSuccessor(agentIndex,action)
                (value,a) = self.evaluate(successor_state,nextAgentIndex,depth,alpha,beta)
                if value > max_value:
                    max_value, max_action = value, action
                if max_value > beta[0]:
                    return (max_value, action)
                alpha[0] = max(alpha[0], max_value)
            return (max_value, max_action)
        #if ghost, find the min successor
        else:
            min_value, min_action = sys.maxint, None
            #increase depth by 1 if a pile is finished
            if agentIndex == gameState.getNumAgents() - 1:
                depth += 1
            for action in legal_actions:
                successor_state = gameState.generateSuccessor(agentIndex,action)
                (value,a) = self.evaluate(successor_state,nextAgentIndex,depth,alpha,beta)
                if value < min_value:
                    min_value, min_action = value, action
                if min_value < alpha[0]:
                    return (min_value, action)
                beta[0] = min(beta[0], min_value)
            return (min_value, min_action)


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

