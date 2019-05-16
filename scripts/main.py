import numpy as np
import os
import configparser
import pandas

import hdf5_operations
import contig_analysis
import calculate_age

import dependencies


def main(config_path, random_seed):
    dependencies.check_dependencies()

    print("Welcome to ROBIN:")
    print("ROBust INference of admixture time")
    config = configparser.ConfigParser()
    config.read(config_path)

    analysis = str(config['Analysis Type']['analysis'])
    phasing = str(config['Analysis Type']['phasing'])

    # read parameters
    max_dp = int(config['Parameters']['max_dp'])
    min_gq = int(config['Parameters']['min_gq'])
    min_alleles = int(config['Parameters']['required_alleles'])
    init_ratio = float(config['Parameters']['freq_ancestor_1'])
    num_chromosomes = int(config['Parameters']['number_of_chromosomes'])
    total_maplength = float(config['Parameters']['total_map_length_of_genome'])

    np.random.seed(int(random_seed))

    # file paths
    vcf_path = str(config['File Names']['vcf_path'])
    hdf5_path = config['File Names']['hdf5_file']
    panel_path = config['File Names']['panel_file']
    genome_size_file = config['File Names']['genome_size_file']
    all_names = hdf5_operations.read_sample_file(config['File Names']['sample_list'])
    ancestry_hmm_path = config['File Names']['ancestry_hmm']
    contig_assignment_path = config['File Names']['contig_assignment_file']

    # if there has been no pre-processing done, there is no panel, and no hdf5 file yet, so first
    # create a hdf5 file (may take some time)
    if not os.path.exists(hdf5_path) and not os.path.exists(panel_path):
        hdf5_operations.create_hdf5_file(vcf_path, hdf5_path)

    genome_size = hdf5_operations.calculate_genome_size(hdf5_path, genome_size_file, analysis)

    # either read the panel from file, or create it if necessary:
    input_panel = hdf5_operations.create_panel(hdf5_path, panel_path,
                                               all_names, max_dp, min_gq, min_alleles, analysis)


    # with all the data available, we now have to calculate local ancestry, and use that to estimate age:
    if analysis == 'assembly_free_chromosomes':
        print("performing assembly_free method assigning contigs to artificial chromosomes")
        contig_analysis.contigs_sim_chroms(input_panel,
                                           all_names, genome_size,
                                           num_chromosomes, init_ratio,
                                           ancestry_hmm_path,
                                           phasing)

    if analysis == 'scaffolds':
        print("performing scaffolding based method")

        calculate_age.infer_ages_scaffolds(input_panel, all_names, genome_size,
                                           init_ratio, ancestry_hmm_path,
                                           phasing)

    if analysis == 'contigs':
        print("performing contig associated method")
        contig_chrom_assignment = pandas.read_table(contig_assignment_path, sep="\t")
        contig_chrom_assignment = contig_chrom_assignment.values
        contig_analysis.contigs_with_assembly(input_panel, contig_chrom_assignment,
                                              all_names, init_ratio, ancestry_hmm_path,
                                              phasing)

    if analysis == 'assembly_free':
        print("performing assembly free method")
        contig_analysis.contigs_assembly_free(input_panel,
                                              all_names,
                                              genome_size,
                                              total_maplength,
                                              init_ratio,
                                              ancestry_hmm_path,
                                              phasing)

    print("ROBIN is done")
