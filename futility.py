# actor_lr = 1e-3
# critic_lr = 1e-3
# num_episodes = 1000
# num_per_episodes = 100
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
#
# target_update = 10
# buffer_size = 10000
# minimal_size = 100
# batch_size = 64
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# num_pms = 100
# num_vms = 400
# pm_cpu_range = (32, 128)
# pm_mem_range = (128, 512)
#
# pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
# vms = generate_vms(num_vms)
# vms, centers = cluster_vms(vms, 3)
#
# env = VirtualMachineClusterEnv(pms, vms)
# env_name = "VirtualMachineCluster"
# replay_buffer = ReplayBuffer(buffer_size)
#
# state_dim = len(env.get_state(env.develop))
# action_dim = 50
# agent = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
# ppo_actor_loss = []
# ppo_critic_loss = []
#
# print_distribution(pms, env.get_develop(), 'Initialization', 3)
#
# return_list = []
# start_time = time.time()
# for i in range(1):
#     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return = 0
#             state = env.get_state()
#
#             trajectory = {}
#
#             for _ in range(num_per_episodes):
#                 action = agent.take_action(state)
#                 next_state, reward = env.step_ppo(action)
#                 trajectory['states'] = state
#                 trajectory['actions'] = action
#                 trajectory['rewards'] = reward
#                 trajectory['next_states'] = next_state
#
#                 state = next_state
#                 episode_return += reward
#
#                 al, cl = agent.update(trajectory)
#
#             return_list.append(episode_return)
#             ppo_actor_loss.append(al)
#             ppo_critic_loss.append(cl)
#
#             if (i_episode + 1) % 10 == 0:
#                 pbar.set_postfix({
#                     'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return': '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
# end_time = time.time()
# run_time = end_time - start_time
#
# ppo_develop = env.get_develop()
# ppo_ei = Reward.ei(ppo_develop, nums_category=3)
# ppo_si = Reward.si(ppo_develop, pms)
# print("PPO Reward：", Reward.compute_reward(0, 0, ppo_ei, ppo_si) / run_time)
#
# episodes_list = list(range(len(return_list)))
# mv_return = moving_average(return_list)
#
# print_returns(episodes_list, mv_return, "PPO", env_name)
# print_distribution(pms, env.get_develop(), 'PPO', 3)
#
# plt.figure(figsize=(12, 6))
# plt.plot(moving_average(ppo_actor_loss), label='PPO Actor Loss')
# plt.plot(moving_average(ppo_critic_loss), label='PPO Critic Loss')
# plt.xlabel('Training steps')
# plt.ylabel('Loss')
# plt.title('Loss Curve for DQN and PPO')
# plt.legend()
# plt.show()
#
#
# lr = 2e-3
# num_episodes = 1000
# num_per_episodes = 100
# hidden_dim = 128
# gamma = 0.98
#
# epsilon = 0.5
# decay = 0.995
# min_epsilon = 0.01
#
# target_update = 10
# buffer_size = 10000
# minimal_size = 100
# batch_size = 64
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# num_pms = 100
# num_vms = 400
# pm_cpu_range = (32, 128)
# pm_mem_range = (128, 512)
#
# pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
# vms = generate_vms(num_vms)
# vms, centers = cluster_vms(vms, 3)
#
# env = VirtualMachineClusterEnv(pms, vms)
# env_name = "VirtualMachineCluster"
# replay_buffer = ReplayBuffer(buffer_size)
#
# state_dim = len(env.get_state(env.develop))
# action_dim = 100
# agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
#
# print_distribution(pms, env.get_develop(), 'Initialization', 3)
#
# random_return_list = []
# return_list = []
#
# nopms = 0
# nom = 0
# ei = 0
# si = 0
#
# start_time = time.time()
# for i in range(1):
#     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return = 0
#             random_return = 0
#             random_develop = env.get_develop()
#             state = env.get_state()
#
#             for _ in range(num_per_episodes):
#                 action = agent.take_action(state, epsilon)
#                 epsilon = max(min_epsilon, epsilon * decay)
#                 next_state, reward = env.step_DQN(action)
#                 replay_buffer.add(state, action, reward, next_state)
#                 state = next_state
#                 episode_return += reward
#
#                 if replay_buffer.size() > minimal_size:
#                     b_s, b_a, b_r, b_ns = replay_buffer.sample(batch_size)
#                     transition_dict = {
#                         'states': b_s,
#                         'actions': b_a,
#                         'rewards': b_r,
#                         'next_states': b_ns
#                     }
#                     agent.update(transition_dict)
#
#                 random_develop, random_reward, nopms, nom, ei, si = greedy_migration(pms,
#                                                                                  random_develop, nopms, nom, ei, si)
#                 random_return += random_reward
#
#             return_list.append(episode_return)
#             random_return_list.append(random_return)
#
#             if (i_episode + 1) % 10 == 0:
#                 pbar.set_postfix({
#                     'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return': '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
# end_time = time.time()
# run_time = end_time - start_time
#
# dqn_develop = env.get_develop()
# dqn_ei = Reward.ei(dqn_develop, nums_category=3)
# dqn_si = Reward.si(dqn_develop, pms)
# print("DQN Reward：", 100.0 / (Reward.compute_reward(0, 0, dqn_ei, dqn_si) * run_time))
#
# random_ei = Reward.ei(random_develop, nums_category=3)
# random_si = Reward.si(random_develop, pms)
# print("Greedy Reward：", 100.0 / (Reward.compute_reward(0, 0, random_ei, random_si) * run_time))
#
# episodes_list = list(range(len(return_list)))
# episodes_list = list(range(len(random_return_list)))
#
# mv_return = moving_average(return_list)
# mv_random_return = moving_average(random_return_list)
#
# print_returns(episodes_list, mv_return, "DQN", env_name)
# print_returns(episodes_list, mv_random_return, "Greedy", env_name)
#
# plt.figure()
# plt.plot(episodes_list, mv_return, label='DQN')
# plt.plot(episodes_list, mv_random_return, label='Greedy')
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('DQN vs Greedy on {}'.format(env_name))
# plt.legend()
# plt.show()
#
# print_distribution(pms, env.get_develop(), 'DQN', 3)
# print_distribution(pms, random_develop, 'Greedy', 3)


# lr = 1e-3
# actor_lr = 1e-3
# critic_lr = 1e-3
# num_episodes = 1000
# num_per_episodes = 100
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
#
# epsilon = 0.5
# decay = 0.995
# min_epsilon = 0.01
#
# n_step = 3
# target_update = 10
# buffer_size = 10000
# minimal_size = 100
# batch_size = 64
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# num_pms = 200
# num_vms = 800
# pm_cpu_range = (32, 128)
# pm_mem_range = (128, 512)
#
# pms = generate_pms(num_pms, pm_cpu_range, pm_mem_range)
# vms = generate_vms(num_vms)
# vms, centers = cluster_vms(vms, 3)
#
# env = VirtualMachineClusterEnv(pms, vms)
# env_name = "VirtualMachineCluster"
#
# replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
# n_step_buffer = NStepBuffer(n_step, gamma=gamma)
#
# state_dim = len(env.get_state(env.develop))
# action_dim = 5
#
# agent_dqn = RainbowDQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent_ppo = PPO(state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
#
# print_distribution(pms, env.get_develop(), 'Initialization', 3)
# print_utilization(env.get_develop(), pms, 'Initialization')
#
# dqn_loss = []
# ppo_actor_loss = []
# ppo_critic_loss = []
#
# flag = False
# n_reward = 0
#
# return_list = []
# start_time = time.time()
# for i in range(1):
#     if flag:
#         break
#     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i+1}') as pbar:
#         if flag:
#             break
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return = 0
#
#             state = env.get_state()
#
#             trajectory = {}
#             trajectory_queue = []
#             for _ in range(num_per_episodes):
#                 action_dqn = agent_dqn.take_action(state)
#                 action_ppo = agent_ppo.take_action(state)
#
#                 next_state, reward = env.step(action_dqn, action_ppo)
#                 done = False
#                 experience = (state, action_dqn, reward, next_state, done)
#
#                 n_step_transition = n_step_buffer.push(experience)
#                 if n_step_transition:
#                     replay_buffer.add(n_step_transition)
#
#                 state = next_state
#                 episode_return += reward
#
#                 trajectory['states'] = state
#                 trajectory['actions'] = action_ppo
#                 trajectory['rewards'] = reward
#                 trajectory['next_states'] = next_state
#
#                 if reward == 0:
#                     n_reward += 1
#                 if n_reward >= 50:
#                     flag = True
#                     break
#
#                 if replay_buffer.size() > minimal_size:
#                     indices, batch, is_weights = replay_buffer.sample(batch_size)
#                     b_s = [exp[0] for exp in batch]
#                     b_a = [exp[1] for exp in batch]
#                     b_r = [exp[2] for exp in batch]
#                     b_ns = [exp[3] for exp in batch]
#
#                     is_weights = torch.tensor(is_weights, dtype=torch.float).view(-1, 1).to(device)
#
#                     transition_dict = {
#                         'states': b_s,
#                         'actions': b_a,
#                         'rewards': b_r,
#                         'next_states': b_ns,
#                         'is_weights': is_weights,
#                     }
#
#                     dl, td_errors = agent_dqn.update(transition_dict)
#                     al, cl = agent_ppo.update(trajectory)
#
#                     dqn_loss.append(dl)
#                     ppo_actor_loss.append(al)
#                     ppo_critic_loss.append(cl)
#
#                     replay_buffer.update_priorities(indices, td_errors)
#
#             for t in n_step_buffer.flush():
#                 replay_buffer.add(t)
#
#             return_list.append(episode_return)
#
#             if (i_episode + 1) % 10 == 0:
#                 pbar.set_postfix({
#                     'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return': '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
# end_time = time.time()
# run_time = end_time - start_time
#
# hrl_develop = env.get_develop()
# hrl_ei = Reward.ei(hrl_develop, nums_category=3)
# hrl_si = Reward.si(hrl_develop, pms)
# print("Time：", run_time)
# print("Reward：", Reward.compute_reward(0, 0, hrl_ei, hrl_si))
# print("HRL Reward：", 100.0 / (Reward.compute_reward(0, 0, hrl_ei, hrl_si) * run_time))
#
# episodes_list = list(range(len(return_list)))
# mv_return = moving_average(return_list)
#
# print_returns(episodes_list, mv_return, "HRL", env_name)
# print_distribution(pms, env.get_develop(), 'HRL', 3)
# print_utilization(env.get_develop(), pms, 'HRL')
#
# plt.figure(figsize=(12, 6))
# plt.plot(moving_average(dqn_loss), label='DQN Loss')
# plt.plot(moving_average(ppo_actor_loss), label='PPO Actor Loss')
# plt.plot(moving_average(ppo_critic_loss), label='PPO Critic Loss')
# plt.xlabel('Training steps')
# plt.ylabel('Loss')
# plt.title('Loss Curve for DQN and PPO')
# plt.legend()
# plt.show()
