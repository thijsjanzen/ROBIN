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


def calc_detect_j_dist(distances, t, n, h_0):
    detect_j = 0
    for m in range(1, len(distances)):
        dm = distances[m]
        if dm > 0 and not np.isnan(dm):
            local_k = 2 * n * h_0 * dm / (2 * n * dm + 1)
            local_j = local_k - local_k * (1 - 1 / (2 * n) - dm) ** t

            detect_j += local_j

    return detect_j


def estimate_age_diff(num_j,
                      distances,
                      population_size,
                      initial_heterozygosity):

    if len(distances) < 1:
        return -1

    def to_fit(params):
        d_j = calc_detect_j_dist(distances, params[0],
                                 population_size, initial_heterozygosity)
        return abs(d_j - num_j)

    res = minimize(to_fit, x0=np.array([200]), method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
    return res.x[0]


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


def get_ancestor_types(array):
    name = 'Hybrid'
    indices = array[:, 1] == name

    output = array[indices, 2]
    if len(output) < 1:
        indices = array[:, 1] == name.lower()

    return array[indices, 2]


def calc_age_contigs(hybrid_result,
                     contig_indices,
                     threshold,
                     initial_heterozygosity,
                     chromosome_size_in_bp,
                     chromosome_size_in_morgan,
                     ancestor_type):

    contig_list = np.unique(contig_indices)

    total_number_of_junctions = 0
    all_markers = []
    prev_index = 0

    for contig in contig_list:
        local_indices = hdf5_operations.get_contig_indices(contig_indices, contig, prev_index)
        prev_index = local_indices[len(local_indices) - 1]
        geno = np.full(len(local_indices), -1)

        geno[hybrid_result['11'][local_indices] >= (1 - threshold)] = 0
        if int(ancestor_type) == 2:
            geno[hybrid_result['02'][local_indices] >= (1 - threshold)] = 1
        else:
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

            number_of_junctions_in_contig = sum(abs(np.diff(geno)))
            total_number_of_junctions += number_of_junctions_in_contig

            if min(marker_locations) < 0:
                local_min = min(marker_locations)
                focal_index = np.where(marker_locations < 0.0)

                print("local diff had negative values, something went wrong")

    ages = np.full(5, -1)



    cnt = 0
    for N in [1000, 10000, 100000, 1000000]:
        if len(all_markers) > 0:
            local_diff = np.diff(chromosome_size_in_morgan * all_markers / chromosome_size_in_bp)
            local_diff = np.insert(local_diff, 0, 0)
            if(min(local_diff) < 0.0):
                local_min = min(local_diff)
                focal_index = np.where(local_diff < 0.0)

                print("local diff had negative values, something went wrong")

            ages[cnt] = estimate_age_diff(total_number_of_junctions, local_diff,
                                          N, initial_heterozygosity)
        cnt += 1
    ages[cnt] = total_number_of_junctions


    return ages


def run_ancestry_hmm(ancestry_hmm_path,
                     anc_1_frequency,
                     file_name,
                     sample_file_name):
    anc_2_frequency = 1 - anc_1_frequency

    # now we can call ancestry_hmm
    command = './' + ancestry_hmm_path + ' -i ' + file_name + ' -s ' + \
              sample_file_name + ' -a 2 ' + str(anc_1_frequency) + ' ' + str(anc_2_frequency) + \
              ' -p 0 100000 ' + str(anc_1_frequency) + ' -p 1 -200 ' + str(anc_1_frequency) + ' -g >/dev/null 2>&1'

    os.system(command)
    return


def create_output_panel(hybrid_panel,
                        allele_matrix,
                        chrom,
                        contig_index,
                        map_length,
                        chromosome_size_in_bp,
                        i,
                        use_contig_index):

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
        if use_contig_index:
            local_contig_index = np.delete(local_contig_index, to_remove)

    # now we have to remove all entries where there was no hybrid alleles
    allele_counts = local_hybrid_panel.sum(1)

    output = output[allele_counts == 2, ]
    focal_contigs = []
    if use_contig_index:
        focal_contigs = local_contig_index[allele_counts == 2]

    diff_pos = np.diff(output[:, 1])
    diff_pos = np.insert(diff_pos, 0, 0)
    assert(max(diff_pos) > 0)
    mult = float(map_length) / chromosome_size_in_bp
    assert(map_length != 0)
    assert(chromosome_size_in_bp != 0)
    assert(mult != 0)
    diff_pos = diff_pos * mult
    assert (max(diff_pos) > 0)

    output2 = np.column_stack((output[:, [0, 1, 2, 3, 4, 5]], diff_pos, output[:, [7, 8]]))
    return output2, focal_contigs


def infer_age_contigs(input_panel,
                      all_names,
                      chromosome_size_in_bp,
                      map_length,
                      anc_1_frequency,
                      chrom,
                      contig_index,
                      ancestry_hmm_path):
    # now we have the basics of the inputfile for ANCESTRY_HMM
    # lets add the hybrids and write file
    hybrid_names = hdf5_operations.extract_sample_names(all_names, 'Hybrid')

    hybrid_panel = input_panel[:, range(7, len(input_panel[0, ]))]
    allele_matrix = input_panel[:, range(1, 6)]

    ancestor_types = get_ancestor_types(all_names)

    for hybrid_index in range(0, len(hybrid_names)):  # type: int

        use_contig_index = True
        local_data = create_output_panel(hybrid_panel, allele_matrix,
                                         chrom, contig_index,
                                         map_length, chromosome_size_in_bp,
                                         hybrid_index, use_contig_index)
        output = local_data[0]
        focal_contigs = local_data[1]

        file_name = "hybrid_input_" + str(chrom) + "_" + str(hybrid_index) + ".txt"
        np.savetxt(fname=file_name, X=output, fmt='%i %i %i %i %i %i %.20f %i %i')

        sample_file_name = "sample_" + str(hybrid_index) + ".txt"
        f = open(sample_file_name, "w")
        f.write(hybrid_names[hybrid_index] + '_' + str(chrom) + "\t" + str(2))
        f.close()

        run_ancestry_hmm(ancestry_hmm_path, anc_1_frequency, file_name, sample_file_name)

        # read file
        result_file_name = hybrid_names[hybrid_index] + '_' + str(chrom) + '.posterior'
        hybrid_result = np.genfromtxt(result_file_name, names = True)

        # now to calculate J and the distribution of markers
        #for threshold in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        for threshold in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0]:
            initial_heterozygosity = 2 * anc_1_frequency * (1 - anc_1_frequency)

            found_ages = calc_age_contigs(hybrid_result,
                                          focal_contigs,
                                          threshold,
                                          initial_heterozygosity,
                                          chromosome_size_in_bp,
                                          map_length,
                                          ancestor_types[hybrid_index])

            popsize = [1000, 10000, 100000, 1000000]
            f = open("output.txt", "a")
            for k in range(0, 4):
                f.write(hybrid_names[hybrid_index] + "\t" + str(chrom) + "\t" + str(threshold) +
                        "\t" + str(popsize[k]) + "\t" + str(found_ages[4]) + "\t" + str(found_ages[k]) + "\n")
            f.close()


def infer_ages_scaffolds(input_panel,
                         all_names,
                         chromosome_size_in_bp,
                         anc_1_frequency,
                         ancestry_hmm_path):
    hybrid_names = hdf5_operations.extract_sample_names(all_names, 'Hybrid')

    hybrid_panel = input_panel[:, range(7, len(input_panel[0, ]))]
    allele_matrix = input_panel[:, range(1, 6)]

    ancestor_types = get_ancestor_types(all_names)

    for i in range(0, len(hybrid_names)):

        map_length = 1
        chrom = 1
        use_contig_index = False
        contig_index = []

        local_data = create_output_panel(hybrid_panel, allele_matrix,
                                         chrom, contig_index,
                                         map_length,
                                         chromosome_size_in_bp,
                                         i, use_contig_index)

        output = local_data[0]

        rel_diff = output[:, 6]

        assert(max(rel_diff) > 0)
        assert(min(rel_diff) >= 0)
        assert(max(rel_diff) < 1)

        file_name = "hybrid_input_" + str(i) + ".txt"
        np.savetxt(fname=file_name, X=output, fmt='%i %i %i %i %i %i %.20f %i %i')

        sample_file_name = "sample_" + str(i) + ".txt"
        f = open(sample_file_name, "w")
        f.write(hybrid_names[i] + "\t" + str(2))
        f.close()

        run_ancestry_hmm(ancestry_hmm_path, anc_1_frequency, file_name, sample_file_name)

        # read file
        result_file_name = hybrid_names[i] + '.posterior'
        hybrid_result = np.genfromtxt(result_file_name, names=True)

        # now to calculate J and the distribution of markers
        for threshold in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            geno = np.full(len(hybrid_result), -1)

            geno[hybrid_result['11'] >= (1 - threshold)] = 0
            if ancestor_types[i] == 2:
                geno[hybrid_result['02'] >= (1 - threshold)] = 1
            else:
                geno[hybrid_result['20'] >= (1 - threshold)] = 1

            informative_markers = geno >= 0
            geno = geno[informative_markers]
            num_j = sum(abs(np.diff(geno)))
            marker_locations = hybrid_result['position']
            marker_locations = marker_locations[informative_markers]

            local_diff = np.diff(marker_locations / chromosome_size_in_bp)
            local_diff = np.insert(local_diff, 0, 0)

            initial_heterozygosity = 2 * anc_1_frequency * (1 - anc_1_frequency)

            f = open("output.txt", "a")

            for N in [1000, 10000, 100000, 1000000]:
                #final_age1 = estimate_age(num_j, marker_locations, N, map_length, initial_heterozygosity,
                #                          chromosome_size_in_bp)

                final_age = estimate_age_diff(num_j, local_diff, N, initial_heterozygosity)

                f.write(hybrid_names[i] + "\t" + str(threshold) + "\t" + str(N) + "\t" + str(num_j) + "\t" + str(
                    final_age) + "\n")

            f.close()
