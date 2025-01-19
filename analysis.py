import json

def calculate_average_reward_from_file(file_path):
    """
    Calculate the average reward across all runs excluding the runs with a reward of -1.0e9.

    Args:
        file_path (str): Path to the JSON file containing the data.

    Returns:
        float: Average reward excluding the specified runs, or None if no valid rewards exist.
    """
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file at {file_path}")
        return None

    total_reward = 0
    valid_rewards_count = 0

    for run in json_data.get("run_details", []):
        for timestep_details in run:
            reward = timestep_details.get("reward", None)
            if reward is not None and reward != -1.0e9:
                total_reward += reward
                valid_rewards_count += 1
    print(valid_rewards_count)

    if valid_rewards_count == 0:
        return None  # Return None if no valid rewards exist

    return total_reward / valid_rewards_count

# Example usage:
file_path = 'results.json'
average_reward = calculate_average_reward_from_file(file_path)
print(f"Average Reward: {average_reward}")
