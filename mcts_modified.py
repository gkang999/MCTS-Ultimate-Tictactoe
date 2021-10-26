from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

num_nodes = 2000
explore_faction = 2.


def traverse_nodes(node, board, state, identity):
    """ Traverses the tree until the end criterion are met.
    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 'red' or 'blue'.
    Returns:        A node from which the next stage of the search can proceed.
                    A state that corresponds to the node.
    """

    working_state = state
    working_node = node
    while (not board.is_ended(working_state) and len(working_node.untried_actions) == 0):
        # print('mathside   ',working_state)
        UCT_max = float('-inf')
        UCT_node = working_node
        UCT_state = working_state

        for child_node in working_node.child_nodes.values():
            # print(child_node.parent_action)
            child_state = board.next_state(working_state, child_node.parent_action)

            if board.is_ended(child_state):
                UCT_node = child_node
                UCT_state = child_state
                break
            reward = child_node.wins / child_node.visits
            if board.current_player(working_state) != identity:
                reward = 1 - reward
            exploration = sqrt(2 * log(working_node.visits) / child_node.visits)
            cur_UCT = reward + exploration
            # print("reward: ",cur_UCT)
            if cur_UCT > UCT_max:
                UCT_max = cur_UCT
                UCT_node = child_node
                UCT_state = child_state

        working_node = UCT_node
        working_state = UCT_state

    return working_node, working_state


def expand_leaf(node, board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node.
    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.
    Returns:    The added child node.
                The corresponing state.
    """
    if board.is_ended(state):
        return node, state
    action = node.untried_actions.pop(0)
    new_state = board.next_state(state, action)
    new_node = MCTSNode(parent=node, parent_action=action, action_list=board.legal_actions(new_state))
    node.child_nodes[action] = new_node
    return new_node, new_state


def check_win(board, state, action):
    # returns true if the action taken by player results in a win on that box
    # returns false otherwise
    temp_board = board
    temp_state = state

    temp_state = temp_board.next_state(temp_state, action)
    owned_boxes = temp_board.owned_boxes(temp_state)

    if owned_boxes[(action[0], action[1])] == board.previous_player(temp_state):
        return True

    return False


def rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.
    Args:
        board:  The game setup.
        state:  The state of the game.
    """

    # First looks for action that will win the box
    # Tries to avoid leading opponent to boxes that are already owned
    # Otherwise play random moves

    while board.is_ended(state) is False:
        winning_action = False
        legal_actions = board.legal_actions(state)
        owned_boxes = board.owned_boxes(state)
        better_actions = []

        for action in legal_actions:
            if check_win(board, state, action):
                state = board.next_state(state, action)
                winning_action = True
                break
            elif owned_boxes[(action[2], action[3])] == 0:
                better_actions.append(action)

        if len(better_actions) > 0 and winning_action is False:
            state = board.next_state(state, choice(better_actions))
        elif winning_action is False:
            state = board.next_state(state, choice(legal_actions))

    return board.points_values(state)


def backpropagate(node, won):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.
    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.
    """

    node.visits += 1
    node.wins += won
    if node.parent == None:
        return
    backpropagate(node.parent, won)
    return


def think(board, state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.
    Args:
        board:  The game setup.
        state:  The state of the game.
    Returns:    The action to be taken.
    """
    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))

    for step in range(num_nodes):
        # Copy the game for sampling a playthrough
        sampled_game = state

        # Start at root
        node = root_node

        # Do MCTS - This is all you!

        node_to_explore, explore_state = traverse_nodes(node, board, sampled_game, identity_of_bot)

        new_leaf_node, new_leaf_state = expand_leaf(node_to_explore, board, explore_state)

        win_result = rollout(board, new_leaf_state)
        if identity_of_bot == 1:
            win = win_result[1]
        elif identity_of_bot == 2:
            win = win_result[2]

        backpropagate(new_leaf_node, win)

        # node = traverse_nodes(node, board, sampled_game, identity_of_bot)

        # if node.parent_action:
        #     sampled_game = board.next_state(sampled_game, node.parent_action)

        # if node.untried_actions:
        #     node = expand_leaf(node, board, sampled_game)
        #     sampled_game = board.next_state(sampled_game, node.parent_action)

        # result_board = rollout(board, sampled_game)

        # if identity_of_bot == 1:
        #     win = result_board[1]
        # elif identity_of_bot == 2:
        #     win = result_board[2]

        # backpropagate(node, win)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    final_score = 0
    best_move = choice(board.legal_actions(state))
    for child in root_node.child_nodes.values():
        if (child.wins / child.visits) > final_score:
            final_score = child.wins / child.visits
            best_move = child.parent_action
    print("Vanilla bot picking %s with expected score %f" % (str(best_move), final_score))
    return best_move