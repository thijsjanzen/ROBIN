import analyze_robin
import prepare_robin
import numpy as np
import argparse

print("ROBIN v0.2 for a single chromosome only")

parser = argparse.ArgumentParser()
parser.add_argument("-v",  "--vcf_file", help="file name of vcf file of interest, can also be a full path")
parser.add_argument("-s", "--samples", help="file name of file containing a list of all samples")
parser.add_argument("-dp", default = 200, help="maximum read depth, default = 200")
parser.add_argument("-gq", default = 40, help="minimum GQ, default = 40")
parser.add_argument("-f", default = 0.5, help="frequency of ancestor 1 in the initial hybrid swarm, default = 0.5")
parser.add_argument("-a", default = -1, help="Minimum number of alleles collected in the smallest ancestry panel required to include a SNP, default value is twice the number of samples (e.g. no missing alleles)")
args = parser.parse_args()

vcf_path = args.vcf_file

all_names = np.loadtxt("/Users/janzen/MEGAsync2/VCF/sample_list.txt", dtype='str')
min_alleles = args.a

if min_alleles < 0: #default value
    num_samples_anc_1 = len(prepare_robin.extract_sample_names(all_names, 'Ancestor_1'))
    num_samples_anc_2 = len(prepare_robin.extract_sample_names(all_names, 'Ancestor_2'))
    min_alleles = 2*min(num_samples_anc_1, num_samples_anc_2)

print("Welcome to ROBIN:\n")
print("ROBust INference of admixture time\n")

#prepare_robin.create_hdf5_file(vcf_path)

input_panel = prepare_robin.create_input_panel(all_names, int(args.dp), int(args.gq), int(min_alleles))
np.savetxt('reference_panel.txt', input_panel)

total_chromosome_size = prepare_robin.obtain_chromosome_size()

analyze_robin.infer_ages(input_panel, all_names, total_chromosome_size, float(args.f))

print("ROBIN is done\n")