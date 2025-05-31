import torch


class RainbowConfig:
    use_double_dqn = True
    use_prioritized_replay = True
    use_dueling_dqn = True
    use_multi_step = True
    use_noisy_net = True
    use_distributional_rl = True
    no_rainbow = False


class Config:
    lr = 1e-3
    actor_lr = 1e-3
    critic_lr = 1e-3
    num_episodes = 1000
    num_per_episodes = 100
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    epsilon = 0.5
    decay = 0.995
    min_epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 100
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_pms = 100
    num_vms = 400
    pm_cpu_range = (32, 128)
    pm_mem_range = (128, 512)
