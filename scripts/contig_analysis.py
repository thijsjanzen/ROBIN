import progressbar
import numpy as np
import os

import calculate_age
import hdf5_operations


def calc_genome_size(local_contig_array):
    local_contig_column = local_contig_array[:, 0]
    unique_contigs = np.unique(local_contig_column[:])
    print("calculating total genome size, this may take a while")
    bar = progressbar.ProgressBar(maxval=len(unique_contigs)).start()
    total_bp = 0
    cnt = 1
    for local_contig in unique_contigs:
        indices = hdf5_operations.get_contig_indices(local_contig_column[:], local_contig)

        contig_pos = local_contig_array[:, 1]
        contig_pos = contig_pos[indices]

        to_remove = []
        for i in range(0, len(contig_pos)):
            if contig_pos[i] != contig_pos[i]:
                to_remove.append(i)

        if len(to_remove) > 0:
            contig_pos = np.delete(contig_pos, to_remove, 0)

        if len(contig_pos) > 0:
            rel_distance = 0
            if len(indices) > 1:
                rel_distance = contig_pos - min(contig_pos)
                total_bp += max(rel_distance)
            if len(indices) == 1:
                total_bp += contig_pos[0]

        bar.update(cnt)
        cnt += 1

    return total_bp


def obtain_genome_size(genome_size_file, contig_array):
    genome_size = -1
    if os.path.exists(genome_size_file):
        input_file = open(genome_size_file, 'r')
        for val in input_file.read().split():
            genome_size = int(val)
        input_file.close()

    if genome_size < 0:
        print("We need to calculate the total size of the genome (in bp)")
        print("This may take quite a while")
        genome_size = calc_genome_size(contig_array)
        print("genome size = " + str(genome_size) + " bp")
        output_file = open(genome_size_file, "w")
        output_file.write(str(genome_size))
        output_file.close()

    return genome_size


def create_chromosome(remaining_contigs,
                      contig_array,
                      hybrid_names):

    chrom_bp = 0
    chromosome = np.empty((0, 7 + 2 * len(hybrid_names)))

    contig_list = contig_array[:, 0]

    contig_names = []
    for contig_iterator in range(0, len(remaining_contigs)):
        focal_contig = remaining_contigs[contig_iterator]

        contig_indices = (contig_list == focal_contig)
        # now we have to get a formatted input panel for these rows

        assert(len(contig_indices) == len(contig_array[:, 0]))

        local_contig_panel = contig_array[contig_indices, ]

        if len(local_contig_panel[:, 0]) > 1:

            contig_name = np.full(len(local_contig_panel[:, 0]), focal_contig)

            contig_positions = local_contig_panel[:, 1]
            local_minimum = min(contig_positions)
            contig_positions = contig_positions - local_minimum + 1
            local_contig_panel[:, 0] = 1
            local_contig_panel[:, 1] = contig_positions + chrom_bp
            local_contig_panel[:, 6] = contig_positions + chrom_bp

            contig_size = max(contig_positions)
            chrom_bp += contig_size

            chromosome = np.vstack((chromosome, local_contig_panel))
            contig_names = np.concatenate((contig_names, contig_name), axis=None)

    return chromosome, contig_names, chrom_bp


def chrom_contig_list(contig_list, chrom):
    indices = contig_list[:, 0] == chrom
    return contig_list[indices, 1]


def contigs_with_assembly(contig_array,
                          contig_chrom_assignment,
                          all_names,
                          anc_1_frequency,
                          ancestry_hmm_path):

    hybrid_names = hdf5_operations.extract_sample_names(all_names, 'Hybrid')

    number_of_chromosomes = len(np.unique(contig_chrom_assignment[:, 0]))

    for chrom in range(0, number_of_chromosomes):
        print("Analyzing Chromosome  " + str(chrom))
        # take subset of panel with only matching contigs
        remaining_contigs = chrom_contig_list(contig_chrom_assignment, chrom)

        np.random.shuffle(remaining_contigs)

        results = create_chromosome(remaining_contigs, contig_array, hybrid_names)
        chromosome = results[0]
        contig_names = results[1]
        genome_size = results[2]

        map_length = 1.0  # this could be changed depending on input by the user

        calculate_age.infer_age_contigs(chromosome, all_names,
                                        genome_size, map_length,
                                        anc_1_frequency,
                                        chrom, contig_names,
                                        ancestry_hmm_path)


def contigs_assembly_free(contig_array, all_samples,
                          genome_size, map_length, init_ratio,
                          ancestry_hmm_path):

    hybrid_names = hdf5_operations.extract_sample_names(all_samples, 'Hybrid')

   # genome_size = obtain_genome_size(genome_size_file, contig_array)

    remaining_contigs = np.unique(contig_array[:, 0])

    results = create_chromosome(remaining_contigs, contig_array, hybrid_names)
    chromosome = results[0]
    contig_names = results[1]

    chrom = 1
    calculate_age.infer_age_contigs(chromosome, all_samples,
                                    genome_size, map_length,
                                    init_ratio, chrom, contig_names,
                                    ancestry_hmm_path)


def contigs_sim_chroms(contig_array, all_samples,
                       genome_size_file, num_chromosomes, init_ratio,
                       ancestry_hmm_path):

    hybrid_names = hdf5_operations.extract_sample_names(all_samples, 'Hybrid')

    genome_size = obtain_genome_size(genome_size_file, contig_array)

    chromosome_size = genome_size / num_chromosomes

    remaining_contigs = np.unique(contig_array[:, 0])
    np.random.shuffle(remaining_contigs)
    chrom_num = 1
    chrom_bp = 0
    chromosome = np.empty((0, 7 + 2 * len(hybrid_names)))

    map_length = 1.0  # by definition here !

    contig_names = []
    print("starting generating random chromosomes and calculating age")

    for contig_iterator in range(0, len(remaining_contigs)):
        focal_contig = remaining_contigs[contig_iterator]
        contig_indices = hdf5_operations.get_contig_indices(contig_array[:, 0], focal_contig)
        # now we have to get a formatted input panel for these rows

        local_contig_panel = contig_array[contig_indices, ]

        if len(local_contig_panel[:, 0]) > 1:

            contig_name = np.full(len(local_contig_panel[:, 0]), focal_contig)

            contig_positions = local_contig_panel[:, 1]
            contig_positions = contig_positions - min(contig_positions) + 1
            local_contig_panel[:, 0] = 1
            local_contig_panel[:, 1] = contig_positions + chrom_bp
            local_contig_panel[:, 6] = contig_positions + chrom_bp

            contig_size = max(contig_positions)
            new_chrom_bp = chrom_bp + contig_size

            if new_chrom_bp > chromosome_size:
                overlap = (chrom_bp + contig_size - chromosome_size) / chromosome_size
                if overlap > 0.5:
                    contig_positions = local_contig_panel[:, 1]
                    contig_positions = contig_positions - min(contig_positions) + 1
                    local_contig_panel[:, 0] = 1
                    local_contig_panel[:, 1] = contig_positions
                    local_contig_panel[:, 6] = contig_positions

                    total_chromosome_size = max(chromosome[:, 1])
                    print("Created artificial chromosome " + str(chrom_num) + " of size " + str(total_chromosome_size))

                    # do check for positions not too small:

                    calculate_age.infer_age_contigs(chromosome, all_samples,
                                                    total_chromosome_size, map_length,
                                                    init_ratio, chrom_num, contig_names,
                                                    ancestry_hmm_path)

                    chromosome = np.empty((0, 7 + 2 * len(hybrid_names)))
                    chromosome = np.vstack((chromosome, local_contig_panel))
                    contig_names = []
                    contig_names = np.concatenate((contig_names, contig_name), axis=None)
                    chrom_num += 1
                    chrom_bp = max(local_contig_panel[:, 1])
                else:
                    chromosome = np.vstack((chromosome, local_contig_panel))
                    contig_names = np.concatenate((contig_names, contig_name), axis=None)
                    total_chromosome_size = max(chromosome[:, 1])
                    print("Created artificial chromosome " + str(chrom_num) + " of size " + str(total_chromosome_size))

                    calculate_age.infer_age_contigs(chromosome, all_samples,
                                                    total_chromosome_size, map_length,
                                                    init_ratio, chrom_num, contig_names,
                                                    ancestry_hmm_path)

                    chromosome = np.empty((0, 7 + 2 * len(hybrid_names)))
                    contig_names = []
                    chrom_num += 1
                    chrom_bp = 0
            else:
                chromosome = np.vstack((chromosome, local_contig_panel))
                chrom_bp = new_chrom_bp
                contig_names = np.concatenate((contig_names, contig_name), axis=None)
