import numpy as np
import allel
import h5py
import progressbar
import os


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


def calc_gq_for_all_samples(array):
    local_gq = np.apply_along_axis(calc_gq_from_pl, 1, array)
    return local_gq


def calc_gq_for_all_samples2(array):
    local_gq = np.empty(len(array))
    for i in range(0, len(array)):
        local_gq[i] = calc_gq_from_pl(array[i, ])
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


def read_sample_names(file_path):
    output_array = []
    f = open(file_path)
    for word in f.read().split():
        output_array.append(word)
    return output_array


def check_dp(name):
    if name == "calldata/GQ":
        return 1
    return 0


def is_diagnostic(v, local_limit):
    # pos = v[0] #just for completeness
    peri_A = v[1]
    peri_a = v[2]
    rhen_A = v[3]
    rhen_a = v[4]
    is_this_snp_diagnostic = False
    if peri_A > peri_a and rhen_A < rhen_a:
        is_this_snp_diagnostic = True
    if peri_A < peri_a and rhen_A > rhen_a:
        is_this_snp_diagnostic = True

    if (peri_A + peri_a) < local_limit:
        is_this_snp_diagnostic = False

    if (rhen_A + rhen_a) < local_limit:
        is_this_snp_diagnostic = False

    return is_this_snp_diagnostic


def reduce_sample(v, factor):
    v_new = np.array(v)
    v_new = v_new / factor
    remainder = v_new - np.floor(v_new)
    new_output = np.floor(v_new)
    if sum(remainder) > 0:
        index = np.argmax(remainder)
        new_output[index] += 1

    return new_output


def subsample(v, local_limit):
    pos = v[0]
    peri_A = v[1]
    peri_a = v[2]
    rhen_A = v[3]
    rhen_a = v[4]

    peri_part = [peri_A, peri_a]
    rhen_part = [rhen_A, rhen_a]

    num_rhen_alleles = rhen_A + rhen_a
    num_peri_alleles = peri_A + peri_a

    # hardcoded count limit here for now! could be removed perhaps, but this should be faster
    if num_peri_alleles >= local_limit and num_rhen_alleles >= local_limit:
        if num_rhen_alleles > num_peri_alleles:
            rhen_part = reduce_sample(rhen_part, num_rhen_alleles / num_peri_alleles)
        elif num_peri_alleles > num_rhen_alleles:
            peri_part = reduce_sample(peri_part, num_peri_alleles / num_rhen_alleles)

    output = np.array([pos, peri_part[0], peri_part[1], rhen_part[0], rhen_part[1]])

    return output


def obtain_positions():
    callset = h5py.File('anc_panel.h5', mode='r')
    return callset['variants/POS'][:]


def obtain_chromosome_size():
    positions = obtain_positions()
    return max(positions)


def obtain_chrom():
    callset = h5py.File('anc_panel.h5', mode='r')
    return callset['variants/CHROM'][:]


def get_all_chromosomes(chromosome):
    hdf5_file_name = chromosome + ".h5"
    callset = h5py.File(hdf5_file_name, mode='r')
    chroms = callset['variants/CHROM'][:]
    return np.unique(chroms)


def convert_if_necessary(input_str):
    try:
        input_str.decode('utf-8')
        return input_str.decode('utf-8')
    except AttributeError:
        return input_str


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


def find_element_in_list(element, list_element):
    for i in range(0, len(list_element)):
        if list_element[i] == element:
                return i
    return -1


def create_hdf5_file(vcf_path):

    if not os.path.isfile('anc_panel.h5'):
        allel.vcf_to_hdf5(vcf_path,
                          'anc_panel.h5',
                          fields=['samples', 'calldata/GT', 'calldata/GQ', 'calldata/PL', 'calldata/DP', 'calldata/RR',
                                  'calldata/VR', 'variants/POS', 'variants/CHROM'],
                          overwrite=False)

    print("performing formatting tests of hdf5 file, is all the data required there?")
    # now we have to verify
    callset = h5py.File('anc_panel.h5', mode='r+')

    get_other_dp = 0
    list_of_attributes = list(callset['calldata'])
    if 'DP' not in list_of_attributes:
        get_other_dp = 1

    if get_other_dp == 0:
        dp = callset['calldata/DP']
        if np.amax(dp[:]) == -1:
            get_other_dp = 1

    if get_other_dp == 1:
        print("Found no DP entries, reconstructing DP from RR and VR\n")
        rr = callset['calldata/RR']
        vr = callset['calldata/VR']
        new_dp = rr[:] + vr[:]
        if np.amax(new_dp[:]) == -1:
            print("Failed to reconstruct DP entries from RR and VR\n")
            print("Please reformat your VCF to include DP\n")
            print("exiting\n")
            exit(1)
        data = callset['calldata/DP']
        data[...] = new_dp[:]

    gq = callset['calldata/GQ']
    if np.amax(gq[:] == -1):
        print("Found no GQ entries, reconstructing GQ from PL\n")
        pl = callset['calldata/PL']

        new_gq = calc_gq(pl[:])
        if np.amax(new_gq[:]) == -1:
            print("Failed to reconstruct GQ entries from PL\n")
            print("Please reformat your VCF to include GQ")
            print("exiting\n")
            exit(1)
        data = callset['calldata/GQ']
        data[...] = new_gq[:]

    callset.close()

    print("Data is stored in hdf5 container and verified")


def obtain_hybrid_alleles(sample_list, max_read_count, gq_threshold):
    callset = h5py.File('anc_panel.h5', mode='r')

    indices = obtain_indices(callset['samples'], sample_list)

    dp = callset['calldata/DP']
    dp = dp[:, indices]

    gq = callset['calldata/GQ']
    gq = gq[:, indices]

    # allright, now we have a genotype array (gt)
    # and we have the genotype qualities gq
    # and we have the number of reads
    # first, we have to filter out the gt by gq..

    dp_pass = dp < max_read_count
    gq_pass = gq >= gq_threshold

    snp_pass = dp_pass * gq_pass

    gt_all = callset['calldata/GT']
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


def obtain_ancestry_panel(sample_list, max_read_count, gq_threshold):
    hdf5_file_name = 'anc_panel.h5'
    callset = h5py.File(hdf5_file_name, mode='r')

    indices = obtain_indices(callset['samples'], sample_list)

    dp = callset['calldata/DP']
    dp = dp[:, indices]

    gq = callset['calldata/GQ']
    gq = gq[:, indices]

    dp_pass = dp < max_read_count
    gq_pass = gq >= gq_threshold

    snp_pass = dp_pass * gq_pass

    gt_all = callset['calldata/GT']
    gt = gt_all[:, indices]

    gt = allel.GenotypeArray(gt)

    alt_alleles = gt.to_n_alt()[:]

    alt_counts = (alt_alleles * snp_pass).sum(1)

    ref_alleles = gt.to_n_ref()[:]
    ref_counts = (ref_alleles * snp_pass).sum(1)

    panel_alleles = np.column_stack((ref_counts, alt_counts))
    return panel_alleles


def extract_sample_names(array, name):
    indices = array[:, 1] == name
    return array[indices, 0]


def create_input_panel(all_names, max_dp, min_gq, min_alleles):
    anc1_names = extract_sample_names(all_names, 'Ancestor_1')
    anc2_names = extract_sample_names(all_names, 'Ancestor_2')
    hybrid_names = extract_sample_names(all_names, 'Hybrid')

    print("processing vcf to create ancestry panel 1\n")
    anc1_panel = obtain_ancestry_panel(anc1_names, max_dp, min_gq)

    print("processing vcf to create ancestry panel 2\n")
    anc2_panel = obtain_ancestry_panel(anc2_names, max_dp, min_gq)

    # now we add the position

    print("postprocessing data and preparing for ancestry_hmm analysis")
    positions = obtain_positions()

    allele_matrix = np.column_stack((positions, anc1_panel, anc2_panel))

    # we first subsample
    allele_matrix2 = np.apply_along_axis(subsample, 1, allele_matrix, local_limit = min_alleles)

    # and then we check which ones are diagnostic
    markers_to_keep = np.apply_along_axis(is_diagnostic, 1, allele_matrix2, local_limit = min_alleles)
   
    allele_matrix3 = allele_matrix2[markers_to_keep, ]

    chrom_indicator = np.full(len(markers_to_keep), 1)  # obtain_chrom(vcf_path)
    chrom_indicator = chrom_indicator[markers_to_keep]

    hybrid_panel = obtain_hybrid_alleles(hybrid_names, max_dp, min_gq)

    hybrid_panel = hybrid_panel[markers_to_keep, ]

    output = np.column_stack((chrom_indicator, allele_matrix3, hybrid_panel))
    return output
