import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

MAX_NEGATIVE_REWARD = -100 # Make sure that this matches the WRONG_END_TIME_REWARD in main.jl

def read_file(file_path):
    """
    Read the contents of a file and return it as a string.

    Args:
        file_path (str): Path to the file to read.

    Returns:
        json: JSON data
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    

def small_rewards(json_data):
    """
    Return rewards excluding the runs with a reward of -1.0e9.

    Args:
        json_data (dict): JSON data.

    Returns:
        list: List of rewards excluding MAX_NEGATIVE_REWARD rewards. (Reward when Ta != Ts when Ts = t)
    """
    valid_rewards = []

    for run in json_data.get("run_details", []):
        for timestep_details in run:
            reward = timestep_details.get("reward", None)
            if reward is not None and reward != MAX_NEGATIVE_REWARD:
                valid_rewards.append(reward)

    return valid_rewards


def num_changes(json_data):
    """
    Return the number of times a new time was announced.

    Args:
        json_data (dict): JSON data.

    Returns:
        list: Number of times for each run that a new time was announced.
    """
    run_details = json_data.get("run_details", [])
    num_changes = [0] * len(run_details)
    for i, run in enumerate(run_details):
        for timestep_details in run:
            action = timestep_details.get("action", {})
            if action != 'dont_announce' and action.get("announced_time", None) is not None:
                num_changes[i] += 1
    return num_changes
    

def plot_simulation(run):
    # Plot the policy a single simulation.
    # The x-axis should be the timestep and the y-axis should be the time announced.
    # There should also be points with each observation.
    timesteps = []
    announced_times = []
    observations = []
    for timestep, timestep_details in enumerate(run):
        timesteps.append(timestep)
        action = timestep_details.get("action", {})
        if action == 'dont_announce':
            announced_time = timestep_details.get("Ta_prev", None)
        else:
            announced_time = action.get("announced_time", None)
        if announced_time is not None:
            announced_times.append(announced_time)
        # else:
        #     announced_times.append(np.nan)  # Use NaN for timesteps without an announcement

        observation = timestep_details.get("To", None)
        observations.append((timestep, observation))
    plt.figure(figsize=(10, 6))
    # Plot the announced time
    plt.plot(timesteps, announced_times, label='Announced Time', marker='o')
    
    # Plot the observations
    obs_timesteps, obs_values = zip(*observations) if observations else ([], [])
    plt.scatter(obs_timesteps, obs_values, color='red', label='Observations', s=100)

    # Plot the highest belief Ts
    highest_belief_Ts = [timestep_details.get("high_b_Ts", None) for timestep_details in run]
    plt.plot(timesteps, highest_belief_Ts, label='Highest Belief Ts', linestyle='--')

    plt.xlabel('Timestep')
    plt.ylabel('Time Announced / Observations')
    plt.title('Policy Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()

    

def main():
    parser = argparse.ArgumentParser(description='Process solver results.')
    parser.add_argument('solver', type=str, help='The solver name')
    args = parser.parse_args()

    file_path = f'results/{args.solver}_results.json'   
    json_data = read_file(file_path)
    run = json_data["run_details"][1] # currently looking at second run for research

    rewards = small_rewards(json_data)
    plot_simulation(run)
    print(f"Average Rewards excluding {MAX_NEGATIVE_REWARD}: {np.mean(rewards)}")
    print(f"Avg Number of Changes per Run: {np.mean(num_changes(json_data))}")

if __name__ == "__main__":
    main()