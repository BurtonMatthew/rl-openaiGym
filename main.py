import gym
env = gym.make('Pong-v0')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render('human')
env.close()  # https://github.com/openai/gym/issues/893