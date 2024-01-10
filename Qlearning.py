import gym
import numpy as np
import matplotlib.pyplot as plt


# Create world and Q table
def setup():
    array_size = (4, 4, 7, 4, 2)
    Q_table = np.zeros(array_size)
    state_shape = 4  # the dimension of the state

    # Set the world according to the requirements
    world = [
        [-2.4, -0.8, 0.8, 2.4],
        [-5, -0.5, 0.5, 5],
        [-12, -6, -1, 0, 1, 6, 12],
        [-100, -50, 50, 100],
    ]
    return world, state_shape, Q_table


# Put the state into Q table
def Discrete(state, world, state_shape):
    Index = []
    for i in range(state_shape):
        Index.append(np.digitize(state[i], world[i]) - 1)  # -1 for index
    return tuple(Index)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    learning_rate = 0.05
    learning_rate_decay = 0.1

    discount = 0.95
    num_trial = 200  # Number of iterations

    average = 10  # Calculate average reward every 10 runs

    # Epsilon starts to decay at the beginning and becomes 0 at the half of the runs
    epsilon = 1
    end_decay = int(num_trial / 2)
    epsilon_decay = epsilon / (end_decay - 1)

    world, state_shape, Q_table = setup()

    Q_record = []  # Save all Q values
    data_record_1 = {"ep": [], "avg": [], "cumu": []}

    for trial in range(num_trial):
        if trial == num_trial - 1:  # Record the last run
            data_record_2 = {"x": [], "theta": [], "time": []}
        table_state = Discrete(env.reset()[0], world, state_shape)
        done = False
        step = 0  # current time step

        while not done:
            step += 1

            if np.random.random() > epsilon:  # Best action
                action = np.argmax(Q_table[table_state])
            else:  # Random action
                action = np.random.randint(0, env.action_space.n)

            up_state, reward, done, _, _1 = env.step(action)
            reward = 0
            up_table_state = Discrete(up_state, world, state_shape)

            max_Q = np.max(Q_table[up_table_state])  # Best Q value of the next state
            curr_Q = Q_table[table_state + (action,)]  # Current Q value

            if done and step < 200:
                reward = -1

            # caculate Q value
            new_Q = (1 - learning_rate) * curr_Q + learning_rate * (
                reward + discount * max_Q
            )

            Q_table[table_state + (action,)] = new_Q  # Update Q_table
            table_state = up_table_state

            if trial == num_trial - 1:
                data_record_2["x"].append(up_state[0])
                data_record_2["theta"].append(up_state[2])
                # data_record_2["x"].append(world[0][up_table_state[0]])
                # data_record_2["theta"].append(world[2][up_table_state[2]])
                data_record_2["time"].append(step)
        Q_record.append(new_Q)

        # Epsilon decay
        if end_decay >= trial:
            epsilon -= epsilon_decay

        # Graph data collection
        if trial % average == 0:  # Calculate average reward every 10 runs
            lastest_Q = Q_record[-average:]
            Aver_Q = sum(lastest_Q) / len(lastest_Q)
            data_record_1["ep"].append(trial)
            data_record_1["avg"].append(Aver_Q)
            print("Episode:", trial)
    env.close()

    # Plot graph group 1 & 2
    # plt.plot(data_record_1["ep"], data_record_1["avg"], label="average rewards")
    # plt.plot(data_record_1["ep"], data_record_1["cumu"], label="cumulative reward")

    # Plot graph group 3
    plt.plot(data_record_2["time"], data_record_2["x"], label="x")
    plt.plot(data_record_2["time"], data_record_2["theta"], label="theta")

    # Graph settings
    plt.legend(loc=4)
    plt.show()
