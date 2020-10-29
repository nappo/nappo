import numpy as np

# obs state has 37 values (x frame_stack_num)
# robot_position: 9 ---> [0, 0:9]
# robot_velocity: 9 ---> [0, 9:18]
# robot_tip_positions: 9 ---> [0, 18:27]
# object_position: 3  ---> [0, -10:-7] or [0, 27:30]
# object_orientation: 4  ---> [0, -7:-3] or [0, 30:34]
# goal_object_position: 3  ---> [0, -3:] or [0, 34:37]

def her_function_robot(obs, rhs, obs2, rew, goal):

    frames_stacked =int(goal.shape[1] / 37)

    # get object_position from goal
    goal_position = goal[:, ..., 27:30]
    goal_orientation = goal[:, ..., 30:34]

    for offset in range(frames_stacked):

        offset *= 37

        # replace goal_object_orientation in obs and obs2 for goal position
        # obs[:, ..., offset + 30 :offset + 34] = goal_orientation
        # obs2[:, ..., offset + 30 :offset + 34] = goal_orientation

        # replace goal_object_position in obs and obs2 for goal position
        obs[:, ..., offset + 34 :offset + 37] = goal_position
        obs2[:, ..., offset + 34 :offset + 37] = goal_position

    # compute new rew
    for i in range(rew.shape[0]):
        rew[i] = _compute_reward(obs[i, -37:], obs2[i, -37:])

    return obs, rhs, obs2, rew


def _compute_reward(previous_observation, observation):
    # calculate first reward term

    current_distance_from_block = np.linalg.norm(
        observation[18:27].reshape(3, 3) - observation[27:30]
    )

    previous_distance_from_block = np.linalg.norm(
        previous_observation[18:27].reshape(3, 3) - previous_observation[27:30]
    )

    reward_term_1 = (
        previous_distance_from_block - current_distance_from_block
    )

    # calculate second reward term
    current_dist_to_goal = np.linalg.norm(
        observation[-3:] - observation[27:30]
    )

    previous_dist_to_goal = np.linalg.norm(
        previous_observation[-3:] - previous_observation[27:30]
    )

    reward_term_2 = previous_dist_to_goal - current_dist_to_goal

    reward = 750 * reward_term_1 + 250 * reward_term_2

    return reward