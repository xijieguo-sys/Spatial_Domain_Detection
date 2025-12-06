import os
from subprocess import Popen, PIPE

batch_prefix = 'run_nichest_hvg'

# datasets = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
hvgs = [10000, 5000, 3000, 2000]
for hvg in hvgs:
    batchfile = open(f'{batch_prefix}{hvg}.sh','w')
    batchfile.write('#!/bin/bash\n'+
                    '#SBATCH -p batch\n'+
                    '#SBATCH -N 1\n'+
                    '#SBATCH -n 16\n'+
                    '#SBATCH --mem=100G\n'+
                    '#SBATCH -t 06:00:00\n'+
                    f'#SBATCH --job-name {batch_prefix}{hvg}\n'+
                    f'#SBATCH --output {batch_prefix}{hvg}_o.txt\n'+
                    f'#SBATCH --error {batch_prefix}{hvg}_e.txt\n'+
                    'set -e\n'+
                    'cd /users/zgao62/nicheST/model/\n'+
                    'source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh\n'+
                    'conda activate quest\n' + 
                    f'python3 main_run_finetune2.py --save_dir ../results_mft2/hvg{hvg}/ --hvg {hvg} --downstream clustering --adata_path ../../data/zgao62/dlpfc_h5ad/151507.h5ad \n')
    batchfile.close()

    process = Popen(['sbatch', f'{batch_prefix}{hvg}.sh'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()