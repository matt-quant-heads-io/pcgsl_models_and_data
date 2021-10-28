import os
import numpy as np
import math
import json

def compute_hamm_dist(final_map, goal):
    hamming_distance = 0
    for i in range(len(final_map)):
        for j in range(len(final_map[0])):
            if final_map[i][j] != goal[i][j]:
                hamming_distance += 1
    return float(hamming_distance / (len(final_map) * len(final_map[0]))), hamming_distance


obs_size = 22
novelty_results_dict = {}
num_diff_tile_threshold = 4
goal_map_indices = [i for i in range(50)]
map_dom_dict = {j: 0 for j in goal_map_indices}
for file in os.listdir(f'/Users/matt/gym_pcgrl/gym-pcgrl/playable_maps_obs_{obs_size}_ep_len_77'):
    abs_path = f'/Users/matt/gym_pcgrl/gym-pcgrl/playable_maps_obs_{obs_size}_ep_len_77/{file}'
    # read in model generated map
    final_map_ptr = open(abs_path, 'r')
    final_map = final_map_ptr.readlines()
    final_map_ptr.close()

    final_map = [row[1:12] for row in final_map]

    # compute min hamming distance and min num tile diffs for that hamming distance
    filepath = '/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/playable_maps/zelda_lvl{}.txt'
    smallest_hamming_dist = math.inf
    num_diff_tiles_of_min_hamm_dist = None
    for goal_map_idx in goal_map_indices:
        next_idx = goal_map_idx
        curr_goal_map_ptr = open(filepath.format(next_idx), 'r')
        curr_goal_map = curr_goal_map_ptr.readlines()
        curr_goal_map_ptr.close()
        curr_goal_map = [row[1:12] for row in curr_goal_map]
        temp_hamm_distance, num_tile_diffs = compute_hamm_dist(final_map, curr_goal_map)
        if temp_hamm_distance < smallest_hamming_dist:
            smallest_hamming_dist = temp_hamm_distance
            num_diff_tiles_of_min_hamm_dist = num_tile_diffs

    # if num_diff_tiles_of_min_hamm_dist is less than threshold than move on to next final map
    if num_diff_tiles_of_min_hamm_dist < num_diff_tile_threshold:
        continue

    # Find all the goal maps that are closest
    closest_maps = []
    for goal_map_idx in goal_map_indices:
        next_idx = goal_map_idx
        curr_goal_map_ptr = open(filepath.format(next_idx), 'r')
        curr_goal_map = curr_goal_map_ptr.readlines()
        curr_goal_map_ptr.close()
        curr_goal_map = [row[1:12] for row in curr_goal_map]
        temp_hamm_distance, num_tile_diffs = compute_hamm_dist(final_map, curr_goal_map)
        if temp_hamm_distance ==  smallest_hamming_dist:
            closest_maps.append((next_idx, num_tile_diffs))
            # inc the map dom dict for this closest goal map
            map_dom_dict[next_idx] += 1

    # store list of tuples for final map [(goal_map_id, num_diff_tiles),]
    novelty_results_dict[file.split('.txt')[0]] = closest_maps
    print(f"for final map: {file} closest maps are: {closest_maps} and num diff tiles are: {num_diff_tiles_of_min_hamm_dist}")



print(f"Pct of new maps: {len(novelty_results_dict) / len(os.listdir(f'/Users/matt/gym_pcgrl/gym-pcgrl/playable_maps_obs_{obs_size}_ep_len_77'))}")
print(f"Number of new maps: {len(novelty_results_dict)}")
print(f"Success rate of final map being playable: {len(os.listdir(f'/Users/matt/gym_pcgrl/gym-pcgrl/playable_maps_obs_{obs_size}_ep_len_77')) / 10000}")

print(f"map_dom_dict: \n{map_dom_dict}")
map_dom_rep = []
for k,v in map_dom_dict.items():
    if v > 0:
        map_dom_rep.append(k)

print(f"len map_dom_dict: \n{len(map_dom_rep)}")

# write map dominance to json
f_map_dom_ptr = open(f"map_dom_results_{obs_size}.json", "w")
json.dump(map_dom_dict, f_map_dom_ptr, indent=4)
f_map_dom_ptr.close()

# write novelty dict to json
novelty_dict = {"novelty": closest_maps}
f_novelty_ptr = open(f"novelty_results_{obs_size}.json", "w")
json.dump(novelty_results_dict, f_novelty_ptr, indent=4)
f_novelty_ptr.close()
#
#
# for k,v in novelty_results_dict.items():
#     if v[0][1] > 30:
#         print(k)
