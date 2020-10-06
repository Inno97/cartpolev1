import gym
import numpy as np

from dqn import DQNAgent

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    load_model = True
    render = True

    agent = DQNAgent(state_size, action_size, render=render, load_model=load_model)

    done = False
    score = 0
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    while not done:
        if agent.render:
            env.render()

        # get action for the current state and go one step in environment
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        score += reward
        state = next_state

        if done:
            print(score)
