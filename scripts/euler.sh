#!/bin/bash

# Job details
TIME=5:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=1  # GPUs per node
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=16384  # RAM for each core (default: 1024)

#timestamp=$(date +%s)
OUTFILE=output__${timestamp}__process-data__${MODEL}__${DATASET}.out  # default: lsf.oJOBID
#NAME=process-data__${MODEL}__${DATASET}


# # Load modules
# module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1 eth_proxy

# Submit job
bsub -W $TIME \
     -n $NUM_CPUS \
     -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" \
     -R "select[gpu_mtotal0>=30000]" \
     -o ${OUTFILE} \
     "source ~/.bashrc; \
     conda activate rt-vs-entropy; \
     python get_predictors.py"
