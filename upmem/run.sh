#!/bin/bash

set -e

build=false
run=false
debug=false
verbose=false

usage() {
    printf "%s %s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n" \
        "USAGE:" "${0}" \
        "[ -b ] - Build code" \
        "[ -r ] - Run code" \
        "[ -d ] - Debug" \
        "[ -V ] - Verbose" \
        "<DATASET - kaggle | random | toy>"
    exit 0
}

die() {
    echo "${1}" 1>&2
    exit 1
}

dataset_valid() {
    [ "${1}" = "kaggle" ] && return 0
    [ "${1}" = "random" ] && return 0
    [ "${1}" = "toy" ] && return 0

    return 1
}

kaggle_env() {
    export NR_TABLES=26
    export NR_COLS=16
    export MAX_NR_BATCHES=512
    export NR_TASKLETS=14
}
build_pytorch=false
random_env() {
    export NR_TABLES=32
    export NR_COLS=64
    export MAX_NR_BATCHES=100
    export NR_TASKLETS=14
    export MAX_INDICES_PER_BATCH=120
}

random_run() {
    echo "Check env: NR_TABLES = ${NR_TABLES}, NR_COLS = ${NR_COLS}"
    if "${build_pytorch}"; then
        cd ${cwd}/../PIM-Pytorch
        # NR_TABLES=${NR_TABLES} NR_COLS=${NR_COLS} MAX_NR_BATCHES=${MAX_NR_BATCHES} NR_TASKLETS=${NR_TASKLETS} REL_WITH_DEB_INFO=1 DEBUG=1 USE_DISTRIBUTED=1 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 BUILD_CAFFE2=1 python3 "${cwd}/../PIM-Pytorch/setup.py" develop
        python3 setup.py clean
        REL_WITH_DEB_INFO=1 DEBUG=1 USE_DISTRIBUTED=1 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 BUILD_CAFFE2=1 python3 "${cwd}/../PIM-Pytorch/setup.py" develop
    else
        echo "skipping pytorch build"
    fi
    cd "${cwd}/${build_dir}"
    python3 "${cwd}/../PIM-dlrm-new/dlrm_s_pytorch.py" \
           --arch-embedding-size=500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000 \
           --arch-sparse-feature-size="${NR_COLS}" \
           --arch-mlp-bot=256-128-"${NR_COLS}" \
           --arch-mlp-top=128-64-1 \
           --data-generation=random \
           --mini-batch-size="${MAX_NR_BATCHES}" \
           --num-batches=20 \
           --num-indices-per-lookup="${MAX_INDICES_PER_BATCH}" \
           --num-indices-per-lookup-fixed=True \
           --inference-only
}
# Old - pre-loadgenerator
# toy_env() {
#     export NR_TABLES=1
#     export NR_COLS=8
#     export DPU_TEST=1
#     export NR_TASKLETS=1
# }
toy_env() {
    export NR_TABLES=9
    export NR_COLS=64
    export NR_BATCHES=64
    export DPU_TEST=1
    export MAX_NR_BATCHES=64
    export NR_TASKLETS=14
    # rows?
}

global_env() {
    "${verbose}" && export V=1
    "${debug}" && export DEBUG=1
    return 0
}

kaggle_run() {
    dlrm="${cwd}/../PIM-dlrm-new"
    python "${dlrm}/dlrm_dpu_pytorch.py" \
           --arch-sparse-feature-size=16 \
           --arch-mlp-bot="13-512-256-64-16" \
           --arch-mlp-top="512-256-1" \
           --data-generation=dataset \
           --data-set=kaggle \
           --processed-data-file="${dlrm}/raw_data/kaggleAdDisplayChallenge_processed.npz" \
           --load-model="${dlrm}/trainedModels/kaggle-model-graham-final.pt" \
           --mini-batch-size=32 \
           --nepochs=1 \
           --inference-only
}

toy_run() {
    echo "DPU_TEST=${DPU_TEST}"
    ./emb_host
}

build_code() {
    # Set ENV
    "${dataset}_env"
    global_env

    make
}

run_code() {
    cwd="${PWD}"
    if "${debug}"; then
        build_dir="./build/debug/"
    else
        build_dir="./build/release/"
    fi
    cd "${build_dir}"
    "${dataset}_run"
}

main() {
    "${build}" && build_code
    "${run}" && run_code
}

options=':bVrdh'
while getopts "${options}" option; do
    case $option in
        b  ) build=true;;
        r  ) run=true;;
        d  ) debug=true;;
        V  ) verbose=true;;
        h  ) usage;;
        \? ) echo "Unknown option: -${OPTARG}" >&2; exit 1;;
        :  ) echo "Missing option argument for -${OPTARG}" >&2; exit 1;;
        *  ) echo "Unimplemented option: -${OPTARG}" >&2; exit 1;;
    esac
done

shift $((OPTIND - 1))

if ! dataset_valid "${1}"; then
    die "'${1}' is not a valid dataset (kaggle | random | toy)"
fi

if ! "${build}" && ! "${run}"; then
    die "Neither -b or -r were used, doing nothing"
    exit 0
fi

dataset="${1}"

main
