from __future__ import division
from scipy.optimize import minimize
import numpy as np
from numpy.linalg import matrix_power
from math import log


import hdf5_operations


def get_states(local_anc):
    all_states = [0] * (-1 + len(local_anc))
    for i in range(1, len(local_anc)):
        left = local_anc[i - 1]
        right = local_anc[i]

        if left == 0 and right == 1:
            all_states[i - 1] = 1
        if left == 1 and right == 0:
            all_states[i - 1] = 2
        if left == 1 and right == 1:
            all_states[i - 1] = 3
        if left == 0 and right == 0:
            all_states[i - 1] = 4
        if left == 2 and right == 1:
            all_states[i - 1] = 5
        if left == 2 and right == 0:
            all_states[i - 1] = 6
        if left == 0 and right == 2:
            all_states[i - 1] = 7
        if left == 1 and right == 2:
            all_states[i - 1] = 8
        if left == 2 and right == 2:
            all_states[i - 1] = 9

    return all_states


def single_state(t, N, d):
    trans_matrix = np.zeros((7, 7))
    trans_matrix[1 - 1, :] = [1 - 1/(2 * N) - 2 * d, 2 * d, 0, 0, 0, 1/(2 * N), 0]
    trans_matrix[2 - 1, :] = [1 / (2 * N), 1 - 3 * 1/(2 * N) - d, d, 2 * 1/(2 * N), 0, 0, 0]
    trans_matrix[3 - 1, :] = [0, 2 * 1 / (2 * N), 1 - 4 * 1/(2 * N), 0, 2 * 1/(2 * N), 0, 0]
    trans_matrix[4 - 1, :] = [0, 0, 0, 1 - 1/(2 * N) - d, d, 1/(2 * N), 0]
    trans_matrix[5 - 1, :] = [0, 0, 0, 2 * 1/(2 * N), 1 - 3 * 1/(2 * N), 0, 1/(2 * N)]
    trans_matrix[6 - 1, :] = [0, 0, 0, 0, 0, 1 - d, d]
    trans_matrix[7 - 1, :] = [0, 0, 0, 0, 0, 1/(2 * N), 1 - 1/(2 * N)]

    initial_state = [1, 0, 0, 0, 0, 0, 0]

    matrix_to_the_power = matrix_power(trans_matrix, int(t))
    output_state = np.matmul(initial_state, matrix_to_the_power)
    return output_state


def get_expectation_O_state(P, p, focal_state):
    q = 1-p
    cond_prob = 1

    if focal_state == 1:
        cond_prob = p * q * (p * q * P[2] + q * P[4] + P[6])
    if focal_state == 2:
        cond_prob = p * q * (p * q * P[2] + q * P[4] + P[6])
    if focal_state == 3:
        cond_prob = (q ** 2) * (P[0] + P[3] + P[6]) + (q ** 3) * (P[1] + P[4]) + (q ** 4) * P[2] + q * P[5]
    if focal_state == 4:
        cond_prob = (p ** 2) * (P[0] + P[3] + P[6]) + (p ** 3) * (P[1] + P[4]) + (p ** 4) * P[2] + p * P[5]
    if focal_state == 5:
        cond_prob = p * q * (p * P[1] + 2 * (p ** 2) * P[2] + (1/2) * P[3] + p * P[4])
    if focal_state == 6:
        cond_prob = p * q * (q * P[1] + 2 * (q ** 2) * P[2] + (1/2) * P[3] + q * P[4])
    if focal_state == 7:
        cond_prob = p * q * (q * P[1] + 2 * (q ** 2) * P[2] + (1/2) * P[3] + q * P[4])
    if focal_state == 8:
        cond_prob = p * q * (p * P[1] + 2 * (p ** 2) * P[2] + (1/2) * P[3] + p * P[4])
    if focal_state == 9:
        cond_prob = p * q * (2 * P[0] + P[1] + 2 * p * q * P[2])

  #  print(cond_prob)
    return log(cond_prob)


def calc_ll_single_state(input_data_frame,
                         local_time,
                         pop_size,
                         freq_ancestor_1):
    if local_time < 2:
        return -1.0 * 1e20

    state = input_data_frame[0]
    di = input_data_frame[1]
    # in contig based analysis, jumps between contigs are indicated by a zero distance
    if di == 0:
        return 0.0

    # print(state, di, local_time)
    seven_states = single_state(local_time, pop_size, di)
    focal_prob = get_expectation_O_state(seven_states,
                                         freq_ancestor_1,
                                         state)
    return min(0.0, focal_prob)


def estimate_age_unphased(local_anc, distances, pop_size, freq_ancestor_1):
    local_states = get_states(local_anc)
    to_analyze = np.column_stack((local_states, distances))

    def to_optim(t):
        if t < 0:
            return 1e20 #arbitrary large number

        local_probs = np.apply_along_axis(calc_ll_single_state, 1, to_analyze, t, pop_size, freq_ancestor_1)
        return -sum(local_probs)

    res = minimize(to_optim, x0=np.array([200]), method='nelder-mead', options={'xtol': 0.1, 'disp': False, 'maxiter': 10000})
    return res.x[0], res.fun

def calc_age_contigs_unphased(hybrid_result,
                     contig_indices,
                     threshold,
                     initial_heterozygosity,
                     chromosome_size_in_bp,
                     chromosome_size_in_morgan):

    contig_list = np.unique(contig_indices)

    all_markers = []
    all_anc = []

    for contig in contig_list:
        local_indices = hdf5_operations.get_contig_indices2(contig_indices, contig)
        if len(local_indices) > 0:
            geno = np.full(len(local_indices), -1)

            geno[hybrid_result['11'][local_indices] >= (1 - threshold)] = 2
            geno[hybrid_result['02'][local_indices] >= (1 - threshold)] = 0
            geno[hybrid_result['20'][local_indices] >= (1 - threshold)] = 1

            informative_markers = geno >= 0

            geno = geno[informative_markers]
            if len(geno) >= 2:
                marker_locations = hybrid_result['position'][local_indices]
                marker_locations = marker_locations[informative_markers]
                lowest_val = min(marker_locations)
                marker_locations -= lowest_val
                if len(all_markers) > 0:
                    marker_locations += max(all_markers)
                    assert(marker_locations[0] == max(all_markers))

                all_markers = np.concatenate((all_markers, marker_locations))
                all_anc = np.concatenate((all_anc, geno))

                if min(marker_locations) < 0:
                    # local_min = min(marker_locations)
                    # focal_index = np.where(marker_locations < 0.0)
                    print("local diff had negative values, something went wrong")

    ages = np.full(5, -1)

    cnt = 0
    for population_size in [1000, 10000, 100000, 1000000]:
        if len(all_markers) > 0:
            local_diff = np.diff(chromosome_size_in_morgan * all_markers / chromosome_size_in_bp)
            if min(local_diff) < 0.0:
                print("local diff had negative values, something went wrong")

            inferred_age = estimate_age_unphased(all_anc, local_diff,
                                                 population_size, initial_heterozygosity)
            ages[cnt] = inferred_age[0]
        cnt += 1

    return ages