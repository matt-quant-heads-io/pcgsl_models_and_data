import math
import os

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
#from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv

from helper import TILES_MAP, str_arr_from_int_arr
from PIL import Image

import numpy as np
import random
import csv
import time
from helper import to_2d_array_level, int_arr_from_str_arr
import pprint
import struct
import hashlib
import numpy as np
import os
import random as _random
import struct
import sys
from gym import error
import random
from gym_pcgrl.wrappers import CroppedImagePCGRLWrapper


pp = pprint.PrettyPrinter(indent=4)

#This is for creating the directories
# path_dir = 'exp_trajectories_const_generated/narrow_greedy/init_maps_lvl{}'
# for idx in range(50):
#     os.makedirs(path_dir.format(idx))

################################################


# This code is for generating the maps
"""def render_map(map, prob, rep, filename='', ret_image=False):
    # format image of map for rendering
    if not filename:
        img = prob.render(map)
    else:
        img = to_2d_array_level(filename)
    img = rep.render(img, tile_size=16, border_size=(1, 1)).convert("RGB")
    img = np.array(img)
    if ret_image:
        return img
    else:
        ren = rendering.SimpleImageViewer()
        ren.imshow(img)
        input(f'')
        time.sleep(0.3)
        ren.close()
"""

# # TODO: Need to change this for Turtle and Narrow Reps
actions_list = [act for act in list(TILES_MAP.values())]
prob = ZeldaProblem()
rep = NarrowRepresentation()

# Reverse the k,v in TILES MAP for persisting back as char map .txt format
REV_TILES_MAP = { "door": "g",
                  "key": "+",
                  "player": "A",
                  "bat": "1",
                  "spider": "2",
                  "scorpion": "3",
                  "solid": "w",
                  "empty": "."}


def to_char_level(map, dir=''):
    level = []

    for row in map:
        new_row = []
        for col in row:
            new_row.append(REV_TILES_MAP[col])
        # add side borders
        new_row.insert(0, 'w')
        new_row.append('w')
        level.append(new_row)
    top_bottom_border = ['w'] * len(level[0])
    level.insert(0, top_bottom_border)
    level.append(top_bottom_border)

    level_as_str = []
    for row in level:
        level_as_str.append(''.join(row) + '\n')

    with open(dir, 'w') as f:
        for row in level_as_str:
            f.write(row)


def act_seq_to_disk(act_seq, path):
    with open(path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(act_seq)


def act_seq_from_disk(path):
    act_seqs = []
    with open(path, "r") as f:
        data = f.readlines()
        for row in data:
            act_seq = [int(n) for n in row.split('\n')[0].split(',')]
            act_seqs.append(act_seq)
    return act_seqs




# Test reading in act_seq
# print(act_seq_from_disk('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories_const_generated/narrow/init_maps_lvl0/repair_sequence_0.csv'))

"""Start with random map"""
def gen_random_map(random, width, height, prob):
    map = random.choice(list(prob.keys()),size=(height,width),p=list(prob.values())).astype(np.uint8)
    return map


def _int_list_from_bigint(bigint):
    # Special case 0
    if bigint < 0:
        raise error.Error('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints

# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def create_seed(a=None, max_bytes=8):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.

    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, int):
        a = a % 2**(8 * max_bytes)
    else:
        raise error.Error('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

def hash_seed(seed=None, max_bytes=8):
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:

    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928

    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)

    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(hash[:max_bytes])


def np_random(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = create_seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed


def find_closest_goal_map(random_map):
    smallest_hamming_dist = math.inf
    closest_map = None
    filepath = 'playable_maps/zelda_lvl{}.txt'
    map_indices = [i for i in range(50)]
    random.shuffle(map_indices)
    # print(f"shuffled map indices: {map_indices}")
    while len(map_indices) > 0:
        next_idx = map_indices.pop()
        curr_goal_map = int_arr_from_str_arr(to_2d_array_level(filepath.format(next_idx)))
        temp_hamm_distance = compute_hamm_dist(random_map, curr_goal_map)
        if temp_hamm_distance < smallest_hamming_dist:
            closest_map = curr_goal_map
    return closest_map

def compute_hamm_dist(random_map, goal):
    hamming_distance = 0.0
    for i in range(len(random_map)):
        for j in range(len(random_map[0])):
            if random_map[i][j] != goal[i][j]:
                hamming_distance += 1
    return float(hamming_distance / (len(random_map) * len(random_map[0])))


def transform(obs, x, y, crop_size):
    map = obs
    # print(f"map in transform is {map}")
    # print(f"self.pad_value is {self.pad_value}")
    # print(f"self.pad is {self.pad}")
    # print(f"x,y is {x},{y}")
    # print(f"self.size is {self.size}")
    # View Centering
    size = crop_size
    pad = crop_size // 2
    padded = np.pad(map, pad, constant_values=1)
    cropped = padded[y:y + size, x:x + size]
    obs = cropped
    new_obs = []
    # print(f"obs is {obs}")
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            new_tile = [0]*8
            new_tile[obs[i][j]] = 1
            new_obs.extend(new_tile)
    # print(f"new_obs is {new_obs}")
    # print(f"len of new_obs is {len(new_obs)}")
    return new_obs


def generate_play_trace_narrow_greedy(env, random_map, goal_map, total_steps, ep_len=10, crop_size=3, render=True):
    """
        The "no-change" action  is 1 greater than the number of tile types (value is 8)


    """
    play_trace = []
    # loop through from 0 to 13 (for 14 tile change actions)
    old_map = random_map.copy()



    # Insert the goal state into the play trace
    # play_trace.append([old_map, None, None])
    # current_loc = [random.randint(0, len(goal_map) - 1), random.randint(0, len(goal_map[0]) - 1)] # [0, 0]

    current_loc = [random.randint(0, len(random_map) - 1), random.randint(0, len(random_map[0]) - 1)]
    # env._rep.reset(3, 3, {0: 0.58, 1: 0.3, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02})
    env._rep._old_map = np.array([np.array(l) for l in random_map])  # np.ndarray(map.copy(), shape=(len(map), len(map[0])), ndim=2).astype(np.uint8)
    env._rep._x = current_loc[1] # 0
    env._rep._y = current_loc[0] # 0
    row_idx, col_idx = env._rep._y, env._rep._x #0, 0
    tile_count = 0


    # print(f"Viewing goal map first...")
    # if render:
    #     map_img = render_map(str_arr_from_int_arr(np.array(goal_map)), prob, rep, ret_image=True)
    #     ren = rendering.SimpleImageViewer()
    #     ren.imshow(map_img)
    #     input(f'')
    #     time.sleep(0.3)
    #     ren.close()

    hamm = compute_hamm_dist(goal_map, random_map)
    curr_step = 0
    episode_len = ep_len #random.randint(1, 77)
    env.reset()
    env.reset()
    while hamm > 0.0 and curr_step < episode_len and curr_step <= total_steps:
        new_map = old_map.copy()
        transition_info_at_step = [None, None, None]  # [current map, destructive_action, expert_action]
        # row_idx, col_idx = random.randint(0, len(map) - 1), random.randint(0, len(map[0]) - 1) # current_loc[1], current_loc[0]
        rep._x = col_idx
        rep._y = row_idx
        # print(f"position ({current_loc[0], current_loc[1]})")

        new_map[row_idx] = old_map[row_idx].copy()
        # print(f"old_tile_type is {old_tile_type}")
        # next_actions = [j for j in actions_list if j != old_tile_type] + ["No-change"]*27
        new_tile_type = goal_map[row_idx][col_idx]
        old_tile_type = random_map[row_idx][col_idx]

        expert_action = [row_idx, col_idx, new_tile_type]
        destructive_action = [row_idx, col_idx, old_tile_type]
        transition_info_at_step[1] = destructive_action.copy()
        transition_info_at_step[2] = expert_action.copy()
        new_map[row_idx][col_idx] = new_tile_type

        # obs, _, dones, info = env.step(expert_action[-1])
        # print(f"obs is {env.observation_space}")
        play_trace.append((transform(old_map.copy(),col_idx,  row_idx, crop_size), expert_action.copy()))
        curr_step += 1
        total_steps += 1


        # Update position
        # current_loc[0] += 1
        # rep._x += 1
        # if current_loc[0] >= rep._map.shape[1]:
        #     current_loc[0] = 0
        #     current_loc[1] += 1
        #     rep._x = 0
        #     rep._y += 1
        #     if current_loc[1] >= rep._map.shape[0]:
        #         current_loc[1] = 0
        #         rep._y = 0

        old_map = new_map

        # Render
        # if render:
        #     map_img = render_map(str_arr_from_int_arr(new_map), prob, rep, ret_image=True)
        #     ren = rendering.SimpleImageViewer()
        #     ren.imshow(map_img)
        #     print(f"tile_count is {tile_count}")
        #     input(f'')
        #     time.sleep(0.3)
        #     ren.close()

        tile_count += 1
        col_idx += 1
        if col_idx >= 11:
            col_idx = 0
            row_idx += 1
            if row_idx >= 7:
                row_idx = 0

        hamm = compute_hamm_dist(goal_map, old_map)
        if hamm == 0.0:
            return play_trace

    return play_trace, total_steps

# obs_size = 22
# episode_len = 10
# dict_len = ((obs_size**2)*8)
# exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}
# exp_traj_dict["target"] = []
# rng, seed = np_random(None)
# play_traces = []
# filepath = 'playable_maps/zelda_lvl{}.txt'
# act_seq_filepath = '/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories/narrow_greedy_obs_size_3/init_maps_lvl{}/repair_sequence_{}.csv'
# for idx in range(50):
#     for j_idx in range(250):
#         print(f"goal map: {idx}, episode: {j_idx}")
#         cropped_wrapper = CroppedImagePCGRLWrapper("zelda-narrow-v0", obs_size,
#                                                    **{'change_percentage': 1, 'trials': 1, 'verbose': True,
#                                                     'cropped_size': obs_size, 'render': False})
#         pcgrl_env = cropped_wrapper.pcgrl_env
#         start_map = gen_random_map(rng, 11, 7, {0: 0.58, 1: 0.3, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02})
#         goal_map = find_closest_goal_map(start_map)
#         play_trace = generate_play_trace_narrow_greedy(pcgrl_env, start_map, goal_map, ep_len=episode_len, crop_size=obs_size)
#         play_traces.append(play_trace)
#
# for episode in play_traces:
#     for p_i in episode:
#         # print(f"p_i is {p_i}")
#         # print(f"len of p_i is {len(p_i)}")
#         action = p_i[1][-1]
#         # print(f"action is {action}")
#         exp_traj_dict["target"].append(action)
#         pt = p_i[0]
#         # print(f"pt is {pt}")
#         assert dict_len == len(pt), f"len(pt) is {len(pt)} and dict_len is {dict_len}"
#         for i in range(len(pt)):
#             exp_traj_dict[f"col_{i}"].append(pt[i])
#
#
# import pandas as pd
#
# df = pd.DataFrame(data=exp_traj_dict)
# df.to_csv(f"narrow_greedy_obs_size_{obs_size}_ep_len_{episode_len}.csv", index=False)
import pandas as pd

goal_set_size = 50
obs_size = 22 # after 22 len 10 and 30 run 15, 9, 5 on both ep lens 10 and 30
episode_len = 10
obs_ep_comobs = [(22, 10), (22, 30), (15, 10), (15,30), (9, 10), (9, 30), (5, 10), (5, 30)]
rng, seed = np_random(None)
filepath = 'playable_maps/zelda_lvl{}.txt'
for obs_size, episode_len in obs_ep_comobs:
    dict_len = ((obs_size**2)*8)
    total_steps = 0
    while total_steps < 962500:
        for idx in range(goal_set_size):
            exp_traj_dict = {f"col_{i}": [] for i in range(dict_len)}
            exp_traj_dict["target"] = []
            play_traces = []
            cropped_wrapper = CroppedImagePCGRLWrapper("zelda-narrow-v0", obs_size,
                                                       **{'change_percentage': 1, 'trials': 1, 'verbose': True,
                                                        'cropped_size': obs_size, 'render': False})
            pcgrl_env = cropped_wrapper.pcgrl_env
            start_map = gen_random_map(rng, 11, 7, {0: 0.58, 1: 0.3, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02})
            goal_map = find_closest_goal_map(start_map)
            play_trace, temp_num_steps = generate_play_trace_narrow_greedy(pcgrl_env, start_map, goal_map, total_steps, ep_len=episode_len, crop_size=obs_size)
            total_steps = temp_num_steps
            print(f"({obs_size, episode_len}) goal map: {idx}, total_steps: {total_steps}")
            play_traces.append(play_trace)

            for episode in play_traces:
                for p_i in episode:
                    # print(f"p_i is {p_i}")
                    # print(f"len of p_i is {len(p_i)}")
                    action = p_i[1][-1]
                    # print(f"action is {action}")
                    exp_traj_dict["target"].append(action)
                    pt = p_i[0]
                    # print(f"pt is {pt}")
                    assert dict_len == len(pt), f"len(pt) is {len(pt)} and dict_len is {dict_len}"
                    for i in range(len(pt)):
                        exp_traj_dict[f"col_{i}"].append(pt[i])

            df = pd.DataFrame(data=exp_traj_dict)
            df.to_csv(f"exp_traj_obs_{obs_size}_ep_len_{episode_len}_goal_size_{goal_set_size}/narrow_greedy_obs_size_{obs_size}_ep_len_{episode_len}_goal_{idx}_step_{total_steps}_goal_size{goal_set_size}.csv", index=False)



        # repair_action_seq = [a[-1] for a in play_trace]
        # to_char_level(str_arr_from_int_arr(play_trace[0][0]), dir=f"/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories/narrow_greedy_obs_size_3/init_maps_lvl{idx}/init_map_{j_idx}.txt")
        # #Write action seq to .csv
        # act_seq_to_disk(repair_action_seq,
        #                 f"/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/exp_trajectories/narrow_greedy_obs_size_3/init_maps_lvl{idx}/repair_sequence_{j_idx}.csv")






    # # get the good level map
    # map = int_arr_from_str_arr(to_2d_array_level(filepath.format(idx)))
    # for j_idx in range(1):
    #     temp_map = map.copy()
    #     play_trace = generate_play_trace_narrow(temp_map, prob, rep, actions_list, render=False)
    #     repair_action_seq = [a[-1] for a in play_trace[:-1]]
    #
    #     # print(f"repair_action_seq is {repair_action_seq}")
    #
    #     # BUILDING TRAINING SET:
    #     # Write final destroyed map to .txt
    #     # to_char_level(str_arr_from_int_arr(play_trace[0][0]), dir=f"exp_trajectories_const_generated/narrow/init_maps_lvl{idx}/init_map_{j_idx}.txt")
    #     # Write action seq to .csv
    #     # act_seq_to_disk(repair_action_seq, f"exp_trajectories_const_generated/narrow/init_maps_lvl{idx}/repair_sequence_{j_idx}.csv")
    #
    #     # BUILDING VALIDATION SET
    #     to_char_level(str_arr_from_int_arr(play_trace[0][0]), dir=f"validation_set/narrow/init_map_{idx}_{j_idx}.txt")
        # act_seq_to_disk(repair_action_seq, f"validation_set/narrow/init_maps_lvl{idx}/repair_sequence_{j_idx}.csv")


        # print(f"repair_action_seq is {repair_action_seq} len is {len(repair_action_seq)}")


        # Test reading the written destroyed level
        # destroyed_map = to_2d_array_level(f'exp_trajectories_const_generated/narrow_greedy/init_maps_lvl{idx}/init_map_{j_idx}.txt')
        # render_map(destroyed_map, prob, rep, filename="", ret_image=False)
        #
        #
        #
        # # Testing repair from destroyed map to goal map
        #
        # print("Rendering repair map from random state")
        # init_map = play_trace[0][0]
        # # print(f"destroyed_map is {destroyed_map}")
        # repair_map = init_map.copy()
        # count = 0
        # for act_seq in repair_action_seq:
        #     repair_map[act_seq[0]][act_seq[1]] = act_seq[2]
        #     count += 1
        #     print(f"repair act count : {count}")
        #     map_img = render_map(str_arr_from_int_arr(repair_map), prob, rep, ret_image=False)
        #     ren = rendering.SimpleImageViewer()
        #     ren.imshow(map_img)
        #     input(f'')
        #     time.sleep(0.3)
        #     ren.close()
        # random_map = int_arr_from_str_arr(to_2d_array_level(f"exp_trajectories_const_generated/narrow_greedy/init_maps_lvl{idx}/init_map_{j_idx}.txt"))
        # repair_sequence = act_seq_from_disk(
        #     f'exp_trajectories_const_generated/narrow_greedy/init_maps_lvl{idx}/repair_sequence_{j_idx}.csv')
        # count = 0
        # for act_seq in repair_action_seq:
        #     random_map[act_seq[0]][act_seq[1]] = act_seq[2]
        #     count += 1
        #     print(f"repair act count : {count}")
        #     map_img = render_map(str_arr_from_int_arr(random_map), prob, rep, ret_image=True)
        #     ren = rendering.SimpleImageViewer()
        #     ren.imshow(map_img)
        #     input(f'')
        #     time.sleep(0.3)
        #     ren.close()
