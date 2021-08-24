class hparams:
    output_dir = 'logs'
    state_dim = 14
    action_dim = 5
    job_num = 100
    service_num = 5
    type_num = 3
    algo = "DQN"  # name of algo
    train_eps = 300  # max trainng episodes
    eval_eps = 50  # number of episodes for evaluating
    gamma = 0.95
    epsilon_start = 0.90  # start epsilon of e-greedy policy
    epsilon_end = 0.01
    epsilon_decay = 500
    lr = 0.0001  # learning rate
    memory_capacity = 100000  # capacity of Replay Memory
    batch_size = 64
    target_update = 4  # update frequency of target net
    hidden_dim = 128  # hidden size of net
    epochs_per_checkpoint = 1


    ckpt = None  # 用来断点继续训练
