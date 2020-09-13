# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    #we need noise to be very low, so I wrote 0, if noise is too high, agent will fall off the cliff
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():
    #Prefer the close exit (+1), risking the cliff (-10)
    #noise is 0 so it's worse risk walking through the cliff,
    #discount and living reward is low, is it's not worse going to +10
    answerDiscount = 0.5
    answerNoise = 0
    answerLivingReward = -5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # Prefer the close exit (+1), but avoiding the cliff (-10)
    # noise is more than 0 so it's not worse risk walking through the cliff,
    # discount and living reward is low, is it's not worse going to +10
    answerDiscount = 0.1
    answerNoise = 0.01
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    #Prefer the distant exit (+10), risking the cliff (-10)
    # discount is not too low and living reward is 0 so, it's no problem going to +10
    # noise is 0 so it's no problem risking cliff
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    #Prefer the distant exit (+10), avoiding the cliff (-10)
    #noise is not too low so it's risky to go through the cliff
    answerDiscount = 0.9
    answerNoise = 0.11
    answerLivingReward = -0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    #Avoid both exits and the cliff (so an episode should never terminate)
    # living reward is too high, so it's worth to stay alive :)
    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = 10
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    # iterations number is too low
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
