video_every   = 25
print_every   = 5
batch_size    = 32
learning_rate = 0.0005
env_id = "Gravitar-v0"
env = gym.make(env_id)
env = NoopResetEnv(env, noop_max=30)
env = MaxAndSkipEnv(env, skip=4)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id%video_every)==0,force=True)
score    = 0.0
marking  = []

# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
seed = 742
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)


q = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
q_target = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)

q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer(100000)

score    = 0.0
marking  = []
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

for n_episode in range(int(1e32)):
    epsilon = max(0.01, 0.08 - 0.01*(n_episode/200)) # linear annealing from 8% to 1%
    s = env.reset()
    done = False
    score = 0.0

    while True:
        a = q.act(s)
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        memory.push(s,a,r/100.0,s_prime, done_mask)
        s = s_prime

        score += r
        if done:
            break
        
    if len(memory)>2000:
        # train(q, q_target, memory, optimizer)
        compute_td_loss(batch_size)

    # do not change lines 44-48 here, they are for marking the submission log
    marking.append(score)
    if n_episode%100 == 0:
        print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []

    # you can change this part, and print any data you like (so long as it doesn't start with "marking")
    if n_episode%print_every==0 and n_episode!=0:
        q_target.load_state_dict(q.state_dict())
        print("episode: {}, score: {:.1f}, epsilon: {:.2f}".format(n_episode, score, epsilon))