import pandas as pd
import json

def compute_hamm_dist(final_map, goal):
    hamming_distance = 0
    for i in range(len(final_map)):
        for j in range(len(final_map[0])):
            if final_map[i][j] != goal[i][j]:
                hamming_distance += 1
    return float(hamming_distance / (len(final_map) * len(final_map[0]))), hamming_distance



obs_size = 15

file_ptr = open(f"novelty_results_{obs_size}.json", "r")
novelty_dict = json.load(file_ptr)


# Load the keys from novelty_dict into a list
map_keys = list(novelty_dict.keys())
num_unique_maps = 0
unique_maps = []


# Iterate thru the list and for each key in list
for curr_map_key in map_keys:
    print(f"comparing curr_map {curr_map_key}")
    #   Remove key, read in map and compare to every other map in the list of keys
    other_maps = list(novelty_dict.keys())
    other_maps.remove(curr_map_key)
    curr_map_unique = True

    curr_map_path = f'/Users/matt/gym_pcgrl/gym-pcgrl/playable_maps_obs_{obs_size}_ep_len_77/{curr_map_key}.txt'
    # read in curr_map
    curr_map_ptr = open(curr_map_path, 'r')
    curr_map = curr_map_ptr.readlines()
    curr_map_ptr.close()

    curr_map = [row[1:12] for row in curr_map]

    for other_map_key in other_maps:
        other_map_path = f'/Users/matt/gym_pcgrl/gym-pcgrl/playable_maps_obs_{obs_size}_ep_len_77/{other_map_key}.txt'
        # read in curr_map
        other_map_ptr = open(other_map_path, 'r')
        other_map = other_map_ptr.readlines()
        other_map_ptr.close()

        other_map = [row[1:12] for row in other_map]

        _, num_tile_diffs = compute_hamm_dist(curr_map, other_map)

        #       If find a match then continue; otherwise is no match then increment unique_maps
        if num_tile_diffs == 0:
            curr_map_unique = False
            break

    if curr_map_unique:
        print(f"curr_map {curr_map_key} is unique!")
        unique_maps.append(curr_map_key)
        num_unique_maps += 1







print(f"diversity pct: {round((num_unique_maps / len(map_keys)) * 100, 2)}")
print(f"unique_maps: {unique_maps}")
print(f"num unique_maps: {len(unique_maps)}")


