from __future__ import division

from math import log

import numpy as np
from numpy.linalg import matrix_power
from scipy.optimize import minimize

import hdf5_operations


def get_prob_from_matrix(left, right, p, P):
    q = 1 - p
    cond_prob = 0
    if left == 0 and right == 0:
        cond_prob = (p ** 2) * (P[1 - 1] + P[4 - 1] + P[7 - 1]) + \
                    (p ** 3) * (P[2 - 1] + P[5 - 1]) + \
                    (p ** 4) * P[3 - 1] + \
                    p * P[6-1]

    if left == 0 and right == 1:
        cond_prob = p * q * (p * q * P[3 - 1] +
                             (1 / 2) * P[5 - 1] +
                             P[7 - 1])

    if left == 0 and right == 2:
        cond_prob = p * q * (p * P[2 - 1] +
                             2 * (p ** 2) * P[3 - 1] +
                             (1 / 2) * P[4 - 1] +
                             p * P[5 - 1])

    if left == 1 and right == 0:
        cond_prob = p * q * (p * q * P[3 - 1] +
                             (1 / 2) * P[5 - 1] +
                             P[7 - 1])

    if left == 1 and right == 1:
        cond_prob = (q ** 2) * (P[1 - 1] + P[4 - 1] + P[7 - 1]) + \
                    (q ** 3) * (P[2 - 1] + P[5 - 1]) + \
                    (q ** 4) * P[3 - 1] + \
                    q * P[6 - 1]

    if left == 1 and right == 2:
        cond_prob = p * q * (q * P[2 - 1] +
                             2 * (q ** 2) * P[3 - 1] +
                             (1 / 2) * P[4 - 1] +
                             q * P[5 - 1])

    if left == 2 and right == 0:
        cond_prob = p * q * (p * P[2 - 1] +
                             2 * (p ** 2) * P[3 - 1] +
                             (1 / 2) * P[4 - 1] +
                             p * P[5 - 1])

    if left == 2 and right == 1:
        cond_prob = p * q * (q * P[2 - 1] +
                             2 * (q ** 2) * P[3 - 1] +
                             (1 / 2) * P[4 - 1] +
                             q * P[5 - 1])

    if left == 2 and right == 2:
        cond_prob = p * q * (2 * P[1 - 1] +
                             P[2 - 1] +
                             4 * p * q * P[3 - 1])

    return cond_prob


def single_state(t, N, d):
    trans_matrix = np.zeros((7, 7))
    trans_matrix[1 - 1, :] = [1 - 1 / (2 * N) - 2 * d, 2 * d, 0, 0, 0, 1 / (2 * N), 0]
    trans_matrix[2 - 1, :] = [1 / (2 * N), 1 - 3 * 1 / (2 * N) - d, d, 2 * 1 / (2 * N), 0, 0, 0]
    trans_matrix[3 - 1, :] = [0, 2 * 1 / (2 * N), 1 - 4 * 1 / (2 * N), 0, 2 * 1 / (2 * N), 0, 0]
    trans_matrix[4 - 1, :] = [0, 0, 0, 1 - 1 / (2 * N) - d, d, 1 / (2 * N), 0]
    trans_matrix[5 - 1, :] = [0, 0, 0, 2 * 1 / (2 * N), 1 - 3 * 1 / (2 * N), 0, 1 / (2 * N)]
    trans_matrix[6 - 1, :] = [0, 0, 0, 0, 0, 1 - d, d]
    trans_matrix[7 - 1, :] = [0, 0, 0, 0, 0, 1 / (2 * N), 1 - 1 / (2 * N)]

    initial_state = [1, 0, 0, 0, 0, 0, 0]

    matrix_to_the_power = matrix_power(trans_matrix, int(t))
    output_state = np.matmul(initial_state, matrix_to_the_power)

    return output_state


def calc_ll_single_state(input_data_frame,
                         local_time,
                         pop_size,
                         freq_ancestor_1,
                         condition):
    if local_time < 2:
        return -1.0 * 1e20

    di = input_data_frame[0]
    left = int(input_data_frame[1])
    right = int(input_data_frame[2])
    # in contig based analysis, jumps between contigs are indicated by a zero distance
    if di == 0:
        return 0.0

    # print(state, di, local_time)
    seven_states = single_state(local_time, pop_size, di)
    probs = [0, 0, 0]
    for j in range(0, 3):
        probs[j] = get_prob_from_matrix(left=left,
                                        right=j,
                                        p=freq_ancestor_1,
                                        P=seven_states)

    focal_prob = probs[right]

    if condition is True:
        rel_prob = focal_prob / sum(probs)
        final_prob = log(rel_prob)
        return final_prob
    else:
        return log(focal_prob)


def estimate_age_unphased(local_anc, distances, pop_size, freq_ancestor_1):
    left = local_anc[range(0, len(local_anc) - 1), ]
    right = local_anc[range(1, len(local_anc)), ]

    to_analyze = np.column_stack((distances, left, right))

    def to_optim(t):
        if t < 0:
            return 1e20  # arbitrary large number

        local_probs = np.apply_along_axis(calc_ll_single_state, 1, to_analyze, t, pop_size, freq_ancestor_1, True)
        local_probs[0] = calc_ll_single_state(to_analyze[0, :], t, pop_size, freq_ancestor_1, False)
        print(t, -sum(local_probs))
        return -sum(local_probs)

    res = minimize(to_optim, x0=np.array([200]), method='nelder-mead',
                   options={'xtol': 0.1, 'disp': False, 'maxiter': 10000})
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
                    assert (marker_locations[0] == max(all_markers))

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
