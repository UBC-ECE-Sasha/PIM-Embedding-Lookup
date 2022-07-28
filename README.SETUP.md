# Setup Instructions

## Repo Setup

Clone repo:

Via SSH:

```shell
git clone \
    --branch lookup-profiling \
    --recursive git@github.com:UBC-ECE-Sasha/PIM-Embedding-Lookup.git
```

or https:

```shell
git clone \
    --branch lookup-profiling \
    --recursive https://github.com/UBC-ECE-Sasha/PIM-Embedding-Lookup.git
```

## Python Setup

Enter `PIM-Embedding-Lookup`

```shell
cd PIM-Embedding-Lookup
```


Install minicona:

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Setup env:

```shell
conda env create -f environment.yml
conda activate pytorch
```

If graphviz fails to install change python-graphviz to graphviz in `environment.yml`,
delete old env and try again.

## Compiling PIM-Pytorch

Set required variables (subject to change, check `upmem/run.sh`)

```shell
export NR_TABLES=10
export NR_COLS=32
export MAX_NR_BATCHES=64
export NR_TASKLETS=14
```

Enter `upmem`, build and run 'random' workload.

```shell
./run.sh -br random
```

To only run, use `-r`

```shell
./run.sh -r random
```
