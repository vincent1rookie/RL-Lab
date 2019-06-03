from grid_world import GridWorld
from dqn import DoubleQNetworkAgent
import numpy as np
import warnings
warnings.filterwarnings("ignore")


EPISODES = 10000

reward_list = []
fidelity_list = []

env = GridWorld(shape=(3, 5), reward=(100, 0, 200), num_of_w=1, num_of_p=1, num_of_r=1, num_of_steps=10)

Agent = DoubleQNetworkAgent(
    n_obs=env.state_space - 1,
    n_action=env.action_space,
    units_layer=(8, 8),
    learning_rate=0.0025,
    name='Test1',
    gamma=0.8,
    buffer_size=500, batch_size=10, target_change_step=10, min_buffer_size=100,
    load_path=None,
    save_path=None
)


# def one_hot(obs, env):
#     obs_one_hot = np.zeros(env)
#     obs_one_hot[obs] = 1
#     return obs_one_hot


for episode in range(EPISODES):

    observation = env.state
    observation_oh = env.one_hot(env.state)
    episode_reward = 0
    while True:
        action = Agent.act(observation_oh, env.A[observation])
        state, reward, done = env.step(action)
        state_oh = env.one_hot(state)
        Agent.record(observation_oh, action, reward, state_oh, done)
        observation = state
        observation_oh = state_oh
        episode_reward += reward

        if done:
            reward_list.append(episode_reward)
            env.reset()
            Agent.learn()
            if episode % 10 == 0:
                print('Reward for episode %d is %f' % (episode, np.mean(reward_list[-10::])))
                f_list = []
                for test_state in range(env.state_space):
                    if test_state in env.w:
                        continue
                    else:
                        test_state_oh = env.one_hot(test_state)
                        action = Agent.act(test_state_oh, env.A[test_state], test_mode=True)
                        if action in np.where(env.Optimal[test_state])[0]:
                            f_list.append(1)
                        else:
                            f_list.append(0)
                fidelity = np.mean(f_list)
                print('Fidelity for episode %d is %f' % (episode, fidelity))
                fidelity_list.append(fidelity)

            break

# import matplotlib.pyplot as plt
#
# x = list(range(0, 10000, 10))
# plt.plot(x, f, color='red', label='training_1')
# plt.plot(x, fidelity_list, color='green', label='training_2')
# plt.ylim((0, 1))
# plt.xlabel('training_sample')
# plt.ylabel('fidelity')
# plt.title('Training result of DQN')
# plt.legend()
# plt.show()


