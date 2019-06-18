from grid_world import GridWorld
from dqn import DoubleQNetworkAgent
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ITER = 30
EPISODES = 2000

REWARD_MATRIX = []
FIDELITY_MATRIX = []
ENV_LIST = []
AGENT_LIST = []

for i in range(ITER):
    reward_list = []
    fidelity_list = []

    env = GridWorld(shape=(3, 5), reward=(100, 0, 200), num_of_w=1, num_of_p=1, num_of_r=1, num_of_steps=10)

    Agent = DoubleQNetworkAgent(
        n_obs=env.state_space - 1,
        n_action=env.action_space,
        units_layer=(8, 8),
        learning_rate=0.0025,
        name='dqn_' + str(i),
        gamma=0.9,
        buffer_size=500, batch_size=10, target_change_step=10, min_buffer_size=100,
        load_path=None,
        save_path=None
    )

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
                if episode % 100 == 0:
                    print('Reward for agent %d, episode %d, is %f' % (i, episode, np.mean(reward_list[-10::])))
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
                    print('Fidelity for agent %d, episode %d, is %f' % (i, episode, fidelity))
                    fidelity_list.append(fidelity)
                break

    REWARD_MATRIX.append(reward_list)
    FIDELITY_MATRIX.append(fidelity_list)
    ENV_LIST.append(env)
    AGENT_LIST.append(Agent)
    print("The %d model completed! Final Fidelity is %f" % (i, fidelity))

# import matplotlib.pyplot as plt
#
# f = np.array(FIDELITY_MATRIX)
# r = np.array(REWARD_MATRIX)
#
# x = list(range(0, 2000, 100))
# # plt.plot(x, f, color='red', label='training_1')
# for i in range(17):
#     plt.plot(x, f[dd][i,:], label='training_2')
# # plt.ylim((0, 1))
# plt.xlabel('training_sample')
# plt.ylabel('reward')
# plt.title('Training result of DQN')
# plt.show()
#
#
# x = list(range(0, 2000, 100))
# plt.plot(x, np.mean(f, axis=0), color='red', label='training_1')
# plt.xlabel('training_sample')
# plt.ylabel('reward')
# plt.title('Training result of DQN')
# plt.show()
#
# a_list = []
# for test_state in range(env.state_space):
#     if test_state in env.w:
#         continue
#     else:
#         test_state_oh = env.one_hot(test_state)
#         action = Agent.act(test_state_oh, env.A[test_state], test_mode=True)
#         a_list.append(action)
#         if action in np.where(env.Optimal[test_state])[0]:
#             f_list.append(1)
#         else:
#             f_list.append(0)
#
# fidelity = np.mean(f_list)
#
# sns.kdeplot(f[:,-1], label)
# sns.kdeplot(f[:,0])
# plt.show()