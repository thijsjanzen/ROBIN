import numpy as np
import os
from scipy.optimize import minimize
import sys
import prepare_robin


def calc_detect_j(markers, t, N, C, H_0):
    detect_j = 0
    for m in range(1, len(markers)):
        dm = markers[m] - markers[m - 1]
        local_k = 2 * N * C * H_0 / (2 * N * C + 1 / dm)  # this works for C = 1
        local_j = local_k - local_k * (1 - 1 / (2 * N) - C * dm) ** t

        detect_j += local_j

    return detect_j


def estimate_age(num_j, marker_locations, population_size, chromosome_size, initial_heterozygosity, chromosome_length):
    marker_distribution = marker_locations / chromosome_length

    def to_fit(params):
        d_j = calc_detect_j(marker_distribution, params[0], population_size, chromosome_size, initial_heterozygosity)
        return abs(d_j - num_j)

    res = minimize(to_fit, x0=np.array([200]), method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
    return res.x[0]


def infer_ages(input_panel, all_names, total_chromosome_size, anc_1_frequency):
    # now we have the basics of the inputfile for ANCESTRY_HMM
    # lets add the hybrids and write file
    hybrid_names = prepare_robin.extract_sample_names(all_names, 'Hybrid')

    hybrid_panel = input_panel[:, range(6, len(input_panel[0, ]))]
    allele_matrix = input_panel[:, range(1, 6)]

    for i in range(0, len(hybrid_names)):
        local_hybrid_panel = hybrid_panel[:, [(i * 2), (i * 2) + 1]]

        local_chrom_indic = input_panel[:, 1]
        output = np.column_stack((local_chrom_indic, allele_matrix, allele_matrix[:, 0], local_hybrid_panel))

        # now we have to remove all entries where there was no hybrid alleles
        allele_counts = local_hybrid_panel.sum(1)
        output = output[allele_counts == 2, ]

        diff_pos = np.diff(output[:, 1]) / total_chromosome_size
        diff_pos = np.insert(diff_pos, 0, -1)
        # output[:, 6] = diff_pos
        output2 = np.column_stack((output[:, [0, 1, 2, 3, 4, 5]], diff_pos, output[:, [7, 8]]))

        file_name = "hybrid_input_" + str(i) + ".txt"
        np.savetxt(fname=file_name, X=output2, fmt='%i %i %i %i %i %i %.20f %i %i')

        sample_file_name = "sample_" + str(i) + ".txt"
        f = open(sample_file_name, "w")
        f.write(hybrid_names[i] + "\t" + str(2))
        f.close()

        anc_2_frequency = 1 - anc_1_frequency

        # now we can call ancestry_hmm

        command = './Ancestry_HMM/src/ancestry_hmm -i ' + file_name + ' -s ' + \
                  sample_file_name + ' -a 2 ' + str(anc_1_frequency) + ' ' + str(anc_2_frequency) + \
                  ' -p 0 100000 ' + str(anc_1_frequency) + ' -p 1 -200 ' + str(anc_1_frequency) + ' -g >/dev/null 2>&1'
        sys.stderr.write(command + "\n")
        os.system(command)

        # read file
        result_file_name = hybrid_names[i] + '.posterior'
        hybrid_result = np.genfromtxt(result_file_name, names=True)

        # now to calculate J and the distribution of markers
        for threshold in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            geno = np.full(len(hybrid_result), -1)

            geno[hybrid_result['11'] > (1 - threshold)] = 0
            if i == 2:
                geno[hybrid_result['02'] > (1 - threshold)] = 1
            else:
                geno[hybrid_result['20'] > (1 - threshold)] = 1

            informative_markers = geno >= 0
            geno = geno[informative_markers]
            num_j = sum(abs(np.diff(geno)))
            marker_locations = hybrid_result['position']
            marker_locations = marker_locations[informative_markers]

            chromosome_size = 1
            initial_heterozygosity = 2 * anc_1_frequency * anc_2_frequency
            chromosome_length = total_chromosome_size
            f = open("output.txt", "a")

            for N in [1000, 10000, 100000, 1000000]:
                final_age = estimate_age(num_j, marker_locations, N, chromosome_size, initial_heterozygosity,
                                         chromosome_length)
                f.write(hybrid_names[i] + "\t" + str(threshold) + "\t" + str(N) + "\t" + str(num_j) + "\t" + str(
                    final_age) + "\n")

            f.close()
