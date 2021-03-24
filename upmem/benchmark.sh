#!/bin/sh

# Workload Size
num_indices_per_lookup_set="32"
mini_batch_size_set="64 128" 
num_batches_set="10"
num_indices_per_lookup_fixed_set="True"

# Model Architecture
arch_sparse_feature_size_set="64" # DO NOT CHANGE
embedding_row_set="12" # SAME WITH NR_TABLES
embedding_table_size_set="65000"

# Unsure how to change DPU environment

set -e

build=false
run=false
debug=false
verbose=false

usage() {
    printf "%s %s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n" \
        "USAGE:" "${0}" \
        "[ -b ] - Build code"
        "[ -r ] - Run code"
    exit 0
}

die() {
    echo "${1}" 1>&2
    exit 1
}

random_env() {
    export NR_TABLES=12
    export NR_COLS=64
    export MAX_NR_BATCHES=64
    export NR_TASKLETS=16
    export MAX_IND_PER_BATCH=32
}

random_run() {
	for num_indices_per_lookup in $num_indices_per_lookup_set; do
		for mini_batch_size in $mini_batch_size_set; do
			for num_batches in $num_batches_set; do
				for num_indices_per_lookup_fixed in $num_indices_per_lookup_fixed_set; do
					for arch_sparse_feature_size in $arch_sparse_feature_size_set; do
						for embedding_row in $embedding_row_set; do
							for embedding_table_size in $embedding_table_size_set; do
								python3 "${cwd}/../dlrm/dlrm_dpu_pytorch.py" \
									--arch-embedding-size=65000-65000-65000-65000-65000-65000-65000-65000-65000-65000-65000-65000 \
									--arch-sparse-feature-size=${arch_sparse_feature_size} \
									--arch-mlp-bot=1440-720-${arch_sparse_feature_size} \
									--arch-mlp-top=40-20-10-1 \
									--data-generation=random \
									--mini-batch-size=${mini_batch_size} \
									--num-batches=${num_batches} \
									--num-indices-per-lookup=$num_indices_per_lookup \
									--num-indices-per-lookup-fixed=True \
									--inference-only

								mv "${cwd}/build/release/runtime.csv" "${cwd}/runtime/${num_indices_per_lookup}_${mini_batch_size}_${num_batches}_${num_indices_per_lookup_fixed}_${arch_sparse_feature_size}_${embedding_row}_${embedding_table_size}"
							done
						done
					done
				done
			done
		done
	done
}

build_code() {
    # Set ENV
    "random_env"

    make
}

run_code() {
    cwd="${PWD}"
    build_dir="./build/release/"
	rm -rf "./runtime"
	mkdir "./runtime"
    cd "${build_dir}"
    "random_run"
}

main() {
    "${build}" && build_code
    "${run}" && run_code
}

options=':br'
while getopts "${options}" option; do
    case $option in
        b  ) build=true;;
        r  ) run=true;;
        \? ) echo "Unknown option: -${OPTARG}" >&2; exit 1;;
        :  ) echo "Missing option argument for -${OPTARG}" >&2; exit 1;;
        *  ) echo "Unimplemented option: -${OPTARG}" >&2; exit 1;;
    esac
done

shift $((OPTIND - 1))

if ! "${build}" && ! "${run}"; then
    die "Neither -b or -r were used, doing nothing"
    exit 0
fi

main
