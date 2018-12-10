import h5py
import numpy as np
import progressbar
import allel
import sys
import pandas
import os
import time


def is_diagnostic(v, local_limit):
    # pos = v[0] #just for completeness
    anc1_allele_1 = v[1]
    anc1_allele_2 = v[2]
    anc2_allele_1 = v[3]
    anc2_allele_2 = v[4]

    if sum(v[1:4]) == 0:
        return False

    is_this_snp_diagnostic = False
    if anc1_allele_1 > anc1_allele_2 and anc2_allele_1 < anc2_allele_2:
        is_this_snp_diagnostic = True
    if anc1_allele_1 < anc1_allele_2 and anc2_allele_1 > anc2_allele_2:
        is_this_snp_diagnostic = True

    if (anc1_allele_1 + anc1_allele_2) < local_limit:
        is_this_snp_diagnostic = False

    if (anc2_allele_1 + anc2_allele_2) < local_limit:
        is_this_snp_diagnostic = False

    return is_this_snp_diagnostic


def reduce_sample(v, factor, target_total):
    v_new = np.array(v)
    v_new = v_new / factor
    new_output = np.floor(v_new)
    while sum(new_output) < target_total:
        index = np.argmax(v)
        new_output[index] += 1

    while sum(new_output) > target_total:
        index = np.argmax(v)
        new_output[index] -= 1

    return new_output


def subsample(v, local_limit):
    pos = v[0]
    anc1_allel_1 = v[1]
    anc1_allel_2 = v[2]
    anc2_allel_1 = v[3]
    anc2_allel_2 = v[4]

    if sum(v[1:4] == 0):
        return v

    anc1 = [anc1_allel_1, anc1_allel_2]

    anc2 = [anc2_allel_1, anc2_allel_2]

    num_anc1_alleles = sum(anc1)
    num_anc2_alleles = sum(anc2)

    if num_anc1_alleles >= local_limit:
        if num_anc2_alleles >= local_limit:
            if num_anc1_alleles > num_anc2_alleles:
                anc1 = reduce_sample(anc1, num_anc1_alleles / num_anc2_alleles, num_anc2_alleles)
            elif num_anc2_alleles > num_anc1_alleles:
                anc2 = reduce_sample(anc2, num_anc2_alleles / num_anc1_alleles, num_anc1_alleles)

            # now the number of alleles should be identical
            assert (sum(anc1) == sum(anc2))

    output = np.array([pos, anc1[0], anc1[1], anc2[0], anc2[1]])

    return output

def read_sample_file(file_name):
    all_names = np.loadtxt(file_name, dtype='str')
    for i in range(0, len(all_names[:, 1])):
        x = all_names[i, 1]
        x = x.lower()
        all_names[i, 1] = x
    return all_names


def extract_sample_names(array, name):
    indices = array[:, 1] == name
    output = array[indices, 0]
    if len(output) < 1:
        indices = array[:, 1] == name.lower()
    return array[indices, 0]


def get_contig_indices(array, element, prev_index):
    output = []
    for i in range(prev_index, len(array)):
        if array[i] == element:
            output.append(i)
        else:
            if len(output) > 0:
                break
    return output

def get_contig_indices2(array, element):
    matches = array == element
    indices = range(0, len(array))
    return indices[matches]



def get_contig_list(local_callset):
    local_contig_column = local_callset['variants/CHROM']
    return np.unique(local_contig_column[:])


def obtain_positions(local_callset):
    return local_callset['variants/POS'][:]


def calc_gq_from_pl(array):
    if min(array) < 0:
        return -1

    if sum(array) == 0:
        return -1

    x = array[array > 0]
    output = -1
    if len(x) > 0:
        output = min(x)
    return output


def convert_if_necessary(input_str):
    try:
        input_str.decode('utf-8')
        return input_str.decode('utf-8')
    except AttributeError:
        return input_str


def convert_if_necessary2(input_str):
    try:
        input_str.decode()
        return input_str.decode()
    except AttributeError:
        return input_str



def calc_gq_for_all_samples(array):
    local_gq = np.apply_along_axis(calc_gq_from_pl, 1, array)
    return local_gq


def calc_gq(local_hdf5_dataset):
    local_numpy_array = local_hdf5_dataset[:]
    columns = len(local_hdf5_dataset[1])
    rows = len(local_hdf5_dataset)

    output = np.empty([rows, columns])
    bar = progressbar.ProgressBar(maxval=len(local_numpy_array)).start()

    for i in range(0, len(local_numpy_array)):
        output[i, ] = calc_gq_for_all_samples(local_numpy_array[i])
        bar.update(i)

    return output


def create_hdf5_file(vcf_path, hdf5_file_name):
    print("converting vcf into hdf5 file anc_panel.h5")
    allel.vcf_to_hdf5(vcf_path,
                      hdf5_file_name,
                      fields=['samples', 'calldata/GT', 'calldata/GQ', 'calldata/PL', 'calldata/DP',
                              'calldata/RR', 'calldata/VR', 'variants/POS', 'variants/CHROM'],
                      overwrite=True,
                      log=sys.stdout)

    print("performing formatting tests of hdf5 file, is all the data required there?")
    local_callset = h5py.File(hdf5_file_name, mode='r+')

    get_other_dp = 0
    list_of_attributes = list(local_callset['calldata'])
    if 'DP' not in list_of_attributes:
        get_other_dp = 1

    if get_other_dp == 0:
        dp = local_callset['calldata/DP']
        if np.amax(dp[:]) == -1:
            get_other_dp = 1

    if get_other_dp == 1:
        print("Found no DP entries, reconstructing DP from RR and VR\n")
        rr = local_callset['calldata/RR']
        vr = local_callset['calldata/VR']
        new_dp = rr[:] + vr[:]
        if np.amax(new_dp[:]) == -1:
            print("Failed to reconstruct DP entries from RR and VR\n")
            print("Please reformat your VCF to include DP\n")
            print("exiting\n")
            exit(1)

        data = local_callset['calldata/DP']
        data[...] = new_dp[:]

    gq = local_callset['calldata/GQ']
    if np.amax(gq[:] == -1):
        print("Found no GQ entries, reconstructing GQ from PL\n")
        pl = local_callset['calldata/PL']

        new_gq = calc_gq(pl[:])
        if np.amax(new_gq[:]) == -1:
            print("Failed to reconstruct GQ entries from PL\n")
            print("Please reformat your VCF to include GQ")
            print("exiting\n")
            exit(1)

        data = local_callset['calldata/GQ']
        data[...] = new_gq[:]

    local_callset.close()

    print("Data is stored in hdf5 container and verified")


def find_element_in_list(element, list_element):
    for i in range(0, len(list_element)):
        if list_element[i] == element:
                return i
    return -1


def obtain_indices(row_names, sample_list):
    indices = []
    for i in range(0, len(row_names)):
        # check if it is in there
        # to_check = row_names[i].decode('UTF-8')
        to_check = convert_if_necessary(row_names[i])
        a = find_element_in_list(to_check, sample_list)
        if a >= 0:
            # append the index in the row names,
            # not the index in the sample_list!
            indices.append(i)
    return indices


def obtain_ancestry_panel(local_callset, sample_list, max_read_count, gq_threshold):

    indices = obtain_indices(local_callset['samples'], sample_list)

    dp = local_callset['calldata/DP']
    dp = dp[:, indices]

    gq = local_callset['calldata/GQ']
    gq = gq[:, indices]

    dp_pass = dp < max_read_count
    gq_pass = gq >= gq_threshold

    snp_pass = dp_pass * gq_pass

    gt_all = local_callset['calldata/GT']
    gt = gt_all[:, indices]

    gt = allel.GenotypeArray(gt)

    alt_alleles = gt.to_n_alt()[:]

    alt_counts = (alt_alleles * snp_pass).sum(1)

    ref_alleles = gt.to_n_ref()[:]
    ref_counts = (ref_alleles * snp_pass).sum(1)

    panel_alleles = np.column_stack((ref_counts, alt_counts))
    return panel_alleles


def obtain_hybrid_alleles(local_callset, sample_list, max_read_count, gq_threshold):

    indices = obtain_indices(local_callset['samples'], sample_list)

    dp = local_callset['calldata/DP']
    dp = dp[:, indices]

    gq = local_callset['calldata/GQ']
    gq = gq[:, indices]

    # allright, now we have a genotype array (gt)
    # and we have the genotype qualities gq
    # and we have the number of reads
    # first, we have to filter out the gt by gq..

    dp_pass = dp < max_read_count
    gq_pass = gq >= gq_threshold

    snp_pass = dp_pass * gq_pass

    gt_all = local_callset['calldata/GT']
    gt = allel.GenotypeArray(gt_all[:, indices])

    hybrid_columns = dp_pass[:, 1]

    for i in range(0, len(sample_list)):
        alt_alleles = gt.to_n_alt()[:, i]

        alt_counts = (alt_alleles * snp_pass[:, i])

        ref_alleles = gt.to_n_ref()[:, i]
        ref_counts = (ref_alleles * snp_pass[:, i])

        to_add = np.column_stack((ref_counts, alt_counts))
        hybrid_columns = np.column_stack((hybrid_columns, to_add))

    # remove the bogus dp_pass[:,1] initialization column
    hybrid_columns = np.delete(hybrid_columns, 0, 1)
    return hybrid_columns


def read_sample_names(file_path):
    output_array = []
    f = open(file_path)
    for word in f.read().split():
        output_array.append(word)
    return output_array


def create_input_panel(local_callset, all_names, max_dp, min_gq, min_alleles,
                       analysis):
    anc1_names = extract_sample_names(all_names, 'Ancestor_1')
    anc2_names = extract_sample_names(all_names, 'Ancestor_2')
    hybrid_names = extract_sample_names(all_names, 'Hybrid')

    print("processing vcf to create ancestry panel 1")
    anc1_panel = obtain_ancestry_panel(local_callset, anc1_names, max_dp, min_gq)

    print("processing vcf to create ancestry panel 2")
    anc2_panel = obtain_ancestry_panel(local_callset, anc2_names, max_dp, min_gq)

    # now we add the position
    positions = obtain_positions(local_callset)

    allele_matrix = np.column_stack((positions, anc1_panel, anc2_panel))

    # we first subsample
    print("subsampling to obtain equal allele counts")
    allele_matrix2 = np.apply_along_axis(subsample, 1, allele_matrix, local_limit=min_alleles)

    # and then we check which ones are diagnostic
    print("removing non-diagnostic SNPs")
    markers_to_keep = np.apply_along_axis(is_diagnostic, 1, allele_matrix2, local_limit=min_alleles)

    allele_matrix3 = allele_matrix2[markers_to_keep, ]

    chrom_indicator = np.full(len(markers_to_keep), 1)

    if analysis != 'scaffolds':
        chrom_indicator = local_callset['variants/CHROM']

    chrom_indicator = chrom_indicator[markers_to_keep]

    hybrid_panel = obtain_hybrid_alleles(local_callset, hybrid_names, max_dp, min_gq)

    hybrid_panel = hybrid_panel[markers_to_keep, ]

    output = np.column_stack((chrom_indicator, allele_matrix3, positions[markers_to_keep], hybrid_panel))
    return output


def read_existing_panel(panel_path, hybrid_names):
    c_array = pandas.read_table(panel_path, sep=" ", header=None)
    contig_array = c_array.values
    num_columns = 7 + 2 * len(hybrid_names)
    while len(contig_array[0, ]) > num_columns:
        contig_array = np.delete(contig_array, -1 + len(contig_array[0, ]), 1)
    return contig_array


def create_panel(hdf5_path, panel_path, all_samples, max_dp, min_gq, min_alleles, analysis):
    hybrid_names = extract_sample_names(all_samples, 'Hybrid')

    if os.path.exists(panel_path):
        print("found panel file, loading from previous parsing")
        return read_existing_panel(panel_path, hybrid_names)
    else:
        print("previous panel file not found or provided, generating new panel file")
        callset = h5py.File(hdf5_path, mode='r')
        contig_array = create_input_panel(callset, all_samples, max_dp, min_gq, min_alleles, analysis)
        file_fmt = '%s %i %i %i %i %i %.20f '
        for x in range(0, len(hybrid_names)):
            file_fmt += '%i %i '

        print("panel file saved for future use as:" + panel_path)
        np.savetxt(panel_path, contig_array, fmt=file_fmt)
        return contig_array


def calculate_genome_size(hdf5_path, genome_size_file, analysis):

    genome_size = 0
    if os.path.exists(genome_size_file):
        input_file = open(genome_size_file, 'r')
        for val in input_file.read().split():
            val = convert_if_necessary(val)
            genome_size = int(val)
        input_file.close()
        if genome_size > 0:
            return genome_size

    local_callset = h5py.File(hdf5_path, mode='r')
    positions = obtain_positions(local_callset)
    if analysis == 'scaffolds':
        genome_size = max(positions) - min(positions)
    else:
        contigs = local_callset['variants/CHROM'][:]
        unique_contigs = np.unique(contigs)
        print("calculating total genome size, this may take a while")
        bar = progressbar.ProgressBar(maxval=len(unique_contigs)).start()
        total_bp = 0
        cnt = 1

        for local_contig in unique_contigs:
            indices = contigs == local_contig

            contig_pos = positions[indices]

            to_remove = []
            for i in range(0, len(contig_pos)):
                if contig_pos[i] != contig_pos[i]:
                    to_remove.append(i)

            if len(to_remove) > 0:
                contig_pos = np.delete(contig_pos, to_remove, 0)

            if len(contig_pos) > 0:
                if len(indices) > 1:
                    min_pos = min(contig_pos)
                    max_pos = max(contig_pos)
                    total_bp += max_pos - min_pos
                if len(indices) == 1:
                    total_bp += contig_pos[0]

            bar.update(cnt)
            cnt += 1
        genome_size = total_bp

    output_file = open(genome_size_file, "w")
    output_file.write(str(genome_size))
    output_file.close()
    return genome_size