from scipy.optimize import minimize
import numpy as np
import hdf5_operations
import os


def calc_detect_j(markers, t, n, c, h_0):
    detect_j = 0
    for m in range(1, len(markers)):
        dm = markers[m] - markers[m - 1]
        if dm > 0:
            local_k = 2 * n * c * h_0 / (2 * n * c + 1 / dm)  # this works for C = 1
            local_j = local_k - local_k * (1 - 1 / (2 * n) - c * dm) ** t

            detect_j += local_j

    return detect_j


def estimate_age(num_j, marker_locations, population_size, chromosome_size_in_morgan,
                 initial_heterozygosity, chromosome_length_in_bp):

    if len(marker_locations) < 1:
        return -1

    marker_distribution = marker_locations / chromosome_length_in_bp

    def to_fit(params):
        d_j = calc_detect_j(marker_distribution, params[0],
                            population_size, chromosome_size_in_morgan, initial_heterozygosity)
        return abs(d_j - num_j)

    res = minimize(to_fit, x0=np.array([200]), method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
    return res.x[0]


def find_contig_order(list_of_contigs):
    output = []
    for i in range(1, len(list_of_contigs)):
        if list_of_contigs[i] != list_of_contigs[i-1]:
            output.append(list_of_contigs[i])

    return output


def calc_age_contigs(hybrid_result,
                     contig_indices,
                     threshold,
                     initial_heterozygosity,
                     chromosome_size_in_bp,
                     chromosome_size_in_morgan,
                     sample_number):

    contig_list = find_contig_order(contig_indices)

    total_number_of_junctions = 0
    all_markers = []

    for contig in contig_list:
        local_indices = hdf5_operations.get_contig_indices(contig_indices, contig)
        geno = np.full(len(local_indices), -1)

        geno[hybrid_result['11'][local_indices] >= (1 - threshold)] = 0
        if sample_number == 2:
            geno[hybrid_result['02'][local_indices] >= (1 - threshold)] = 1
        else:
            geno[hybrid_result['20'][local_indices] >= (1 - threshold)] = 1

        informative_markers = geno >= 0

        geno = geno[informative_markers]
        if len(geno) > 0:
            marker_locations = hybrid_result['position'][local_indices]
            marker_locations = marker_locations[informative_markers]

            number_of_junctions_in_contig = sum(abs(np.diff(geno)))
            total_number_of_junctions += number_of_junctions_in_contig

            marker_locations -= min(marker_locations)
            if(len(all_markers) > 0):
                marker_locations += max(all_markers)

            all_markers = np.concatenate((all_markers, marker_locations))

    ages = np.full(5, -1)
    cnt = 0
    for N in [1000, 10000, 100000, 1000000]:
        ages[cnt] = estimate_age(total_number_of_junctions, all_markers,
                                 N, chromosome_size_in_morgan, initial_heterozygosity,
                                 chromosome_size_in_bp)
        cnt += 1
    ages[cnt] = total_number_of_junctions

    return ages


def infer_age_contigs(input_panel, all_names, chromosome_size_in_bp,
                      anc_1_frequency, chrom, contig_index, ancestry_hmm_path):
    # now we have the basics of the inputfile for ANCESTRY_HMM
    # lets add the hybrids and write file
    hybrid_names = hdf5_operations.extract_sample_names(all_names, 'Hybrid')

    hybrid_panel = input_panel[:, range(7, len(input_panel[0, ]))]
    allele_matrix = input_panel[:, range(1, 6)]

    for i in range(0, len(hybrid_names)):
        local_hybrid_panel = hybrid_panel[:, [(i * 2), (i * 2) + 1]]

        local_chrom_indic = np.full(len(allele_matrix[:, 0]), chrom)
        output = np.column_stack((local_chrom_indic, allele_matrix, allele_matrix[:, 0], local_hybrid_panel))

        local_contig_index = contig_index
        to_remove = []
        for k in range(0, len(output[:, 0])):
            for l in range(0, len(output[k, ])):
                if output[k, l] != output[k, l]:  # isnan
                    to_remove.append(k)
                    break

        if len(to_remove) > 0:
            output = np.delete(output, to_remove, 0)
            local_hybrid_panel = np.delete(local_hybrid_panel, to_remove, 0)
            local_contig_index = np.delete(local_contig_index, to_remove)

        # now we have to remove all entries where there was no hybrid alleles
        allele_counts = local_hybrid_panel.sum(1)

        output = output[allele_counts == 2, ]
        focal_contigs = local_contig_index[allele_counts == 2]

        diff_pos = np.diff(output[:, 1])
        diff_pos = np.insert(diff_pos, 0, -1)
        mult = 1.0 / chromosome_size_in_bp
        diff_pos = diff_pos * mult

        output2 = np.column_stack((output[:, [0, 1, 2, 3, 4, 5]], diff_pos, output[:, [7, 8]]))

        file_name = "hybrid_input_" + str(chrom) + "_" + str(i) + ".txt"
        np.savetxt(fname=file_name, X=output2, fmt='%i %i %i %i %i %i %.20f %i %i')

        sample_file_name = "sample_" + str(i) + ".txt"
        f = open(sample_file_name, "w")
        f.write(hybrid_names[i] + '_' + str(chrom) + "\t" + str(2))
        f.close()

        anc_2_frequency = 1 - anc_1_frequency

        # now we can call ancestry_hmm
        command = './' + ancestry_hmm_path + ' -i ' + file_name + ' -s ' + \
                  sample_file_name + ' -a 2 ' + str(anc_1_frequency) + ' ' + str(anc_2_frequency) + \
                  ' -p 0 100000 ' + str(anc_1_frequency) + ' -p 1 -200 ' + str(anc_1_frequency) + ' -g >/dev/null 2>&1'

        os.system(command)

        # read file
        result_file_name = hybrid_names[i] + '_' + str(chrom) + '.posterior'
        hybrid_result = np.genfromtxt(result_file_name, names=True)

        # now to calculate J and the distribution of markers
        for threshold in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:

            chromosome_size_in_morgan = 1
            initial_heterozygosity = 2 * anc_1_frequency * anc_2_frequency

            found_ages = calc_age_contigs(hybrid_result,
                                          focal_contigs,
                                          threshold,
                                          initial_heterozygosity,
                                          chromosome_size_in_bp,
                                          chromosome_size_in_morgan,
                                          i)

            popsize = [1000, 10000, 100000, 1000000]
            f = open("output.txt", "a")
            for k in range(0, 4):
                f.write(hybrid_names[i] + "\t" + str(chrom) + "\t" + str(threshold) +
                        "\t" + str(popsize[k]) + "\t" + str(found_ages[4]) + "\t" + str(found_ages[k]) + "\n")
            f.close()


def infer_ages_scaffolds(input_panel, all_names, total_chromosome_size, anc_1_frequency, ancestry_hmm_path):
    hybrid_names = hdf5_operations.extract_sample_names(all_names, 'Hybrid')

    hybrid_panel = input_panel[:, range(7, len(input_panel[0, ]))]
    allele_matrix = input_panel[:, range(1, 6)]

    for i in range(0, len(hybrid_names)):
        local_hybrid_panel = hybrid_panel[:, [(i * 2), (i * 2) + 1]]

        local_chrom_indic = input_panel[:, 0]
        output = np.column_stack((local_chrom_indic, allele_matrix, allele_matrix[:, 0], local_hybrid_panel))

        to_remove = []
        for k in range(0, len(output[:, 0])):
            for l in range(0, len(output[k, ])):
                if output[k, l] != output[k, l]:  # isnan
                    to_remove.append(k)
                    break

        if len(to_remove) > 0:
            output = np.delete(output, to_remove, 0)
            local_hybrid_panel = np.delete(local_hybrid_panel, to_remove, 0)

        # now we have to remove all entries where there was no hybrid alleles
        allele_counts = local_hybrid_panel.sum(1)
        output = output[allele_counts == 2, ]

        diff_pos = np.diff(output[:, 1]) / total_chromosome_size
        diff_pos = np.insert(diff_pos, 0, -1)
        
	print(diff_pos)
	
        output2 = np.column_stack((output[:, [0, 1, 2, 3, 4, 5]], diff_pos, output[:, [7, 8]]))
	print(output2[0,:])
	print(output2)

        file_name = "hybrid_input_" + str(i) + ".txt"
        np.savetxt(fname=file_name, X=output2, fmt='%i %i %i %i %i %i %.20f %i %i')

        sample_file_name = "sample_" + str(i) + ".txt"
        f = open(sample_file_name, "w")
        f.write(hybrid_names[i] + "\t" + str(2))
        f.close()

        anc_2_frequency = 1 - anc_1_frequency

        # now we can call ancestry_hmm
        command = './' + ancestry_hmm_path + ' -i ' + file_name + ' -s ' + \
                  sample_file_name + ' -a 2 ' + str(anc_1_frequency) + ' ' + str(anc_2_frequency) + \
                  ' -p 0 100000 ' + str(anc_1_frequency) + ' -p 1 -200 ' + str(anc_1_frequency) + ' -g >/dev/null 2>&1'

        os.system(command)

        # read file
        result_file_name = hybrid_names[i] + '.posterior'
        hybrid_result = np.genfromtxt(result_file_name, names=True)

        # now to calculate J and the distribution of markers
        for threshold in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            geno = np.full(len(hybrid_result), -1)

            geno[hybrid_result['11'] >= (1 - threshold)] = 0
            if i == 2:
                geno[hybrid_result['02'] >= (1 - threshold)] = 1
            else:
                geno[hybrid_result['20'] >= (1 - threshold)] = 1

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


def infer_ages_within_contigs(input_panel, all_names, chromosome_size_in_bp, anc_1_frequency, chrom, contig_index,
                              ancestry_hmm_path):
    # now we have the basics of the inputfile for ANCESTRY_HMM
    # lets add the hybrids and write file

    hybrid_names = hdf5_operations.extract_sample_names(all_names, 'Hybrid')

    hybrid_panel = input_panel[:, range(7, len(input_panel[0, ]))]
    allele_matrix = input_panel[:, range(1, 6)]

    for i in range(0, len(hybrid_names)):
        local_hybrid_panel = hybrid_panel[:, [(i * 2), (i * 2) + 1]]

        local_chrom_indic = input_panel[:, 0]
        output = np.column_stack((local_chrom_indic, allele_matrix, allele_matrix[:, 0], local_hybrid_panel))
        to_remove = []
        local_contig_index = contig_index

        for k in range(0, len(output[:, 0])):
            for l in range(0, len(output[k, ])):
                if output[k, l] != output[k, l]:  # isnan
                    to_remove.append(k)
                    break

        if len(to_remove) > 0:
            output = np.delete(output, to_remove, 0)
            local_hybrid_panel = np.delete(local_hybrid_panel, to_remove, 0)
            local_contig_index = np.delete(local_contig_index, to_remove)

        # now we have to remove all entries where there was no hybrid alleles
        allele_counts = local_hybrid_panel.sum(1)

        assert(len(local_contig_index) == len(allele_counts))
        assert(len(allele_counts) == len(local_hybrid_panel[:, 1]))
        assert(len(allele_counts) == len(output[:, 1]))

        output = output[allele_counts == 2, ]
        focal_contigs = local_contig_index[allele_counts == 2]

        diff_pos = np.diff(output[:, 1])
        diff_pos = np.insert(diff_pos, 0, -1)
        mult = 1.0 / chromosome_size_in_bp
        diff_pos = diff_pos * mult

        output2 = np.column_stack((output[:, [0, 1, 2, 3, 4, 5]], diff_pos, output[:, [7, 8]]))

        file_name = "hybrid_input_" + str(chrom) + "_" + str(i) + ".txt"
        np.savetxt(fname=file_name, X=output2, fmt='%i %i %i %i %i %i %.20f %i %i')

        sample_file_name = "sample_" + str(i) + ".txt"
        f = open(sample_file_name, "w")
        f.write(hybrid_names[i] + '_' + str(chrom) + "\t" + str(2))
        f.close()

        anc_2_frequency = 1 - anc_1_frequency

        # now we can call ancestry_hmm

        command = './' + ancestry_hmm_path + ' -i ' + file_name + ' -s ' + \
                  sample_file_name + ' -a 2 ' + str(anc_1_frequency) + ' ' + str(anc_2_frequency) + \
                  ' -p 0 100000 ' + str(anc_1_frequency) + ' -p 1 -200 ' + str(anc_1_frequency) + ' -g >/dev/null 2>&1'

        os.system(command)

        # read file
        result_file_name = hybrid_names[i] + '_' + str(chrom) + '.posterior'
        hybrid_result = np.genfromtxt(result_file_name, names=True)

        # now to calculate J and the distribution of markers
        for threshold in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:

            chromosome_size_in_morgan = 1
            initial_heterozygosity = 2 * anc_1_frequency * anc_2_frequency

            found_ages = calc_age_contigs(hybrid_result,
                                          focal_contigs,
                                          threshold,
                                          initial_heterozygosity,
                                          chromosome_size_in_bp,
                                          chromosome_size_in_morgan,
                                          i)

            popsizes = [1000, 10000, 100000, 1000000]
            f = open("output.txt", "a")
            for k in range(0, 4):
                f.write(hybrid_names[i] + "\t" + str(chrom) + "\t" + str(threshold) +
                        "\t" + str(popsizes[k]) + "\t" + str(found_ages[4]) + "\t" + str(found_ages[k]) + "\n")
            f.close()
