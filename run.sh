#!/usr/bin/env bash
# Use your virtual environment's python
# Exemple: sbatch run.sh \$FILE_PATH [cleanup]

if [ -z "$1" ]; then
    echo "Erro: Forneça o caminho do arquivo Python como argumento"
    echo "Exemplo: sbatch run.sh \$FILE_PATH cleanup"
    exit 1
fi

# Check if second argument is 'cleanup'
do_cleanup=false
if [ "$2" = "cleanup" ]; then
    do_cleanup=true
fi

/home/exx/users/nicolas/venv/bin/python /home/exx/users/nicolas/TCC/$1

# Clean up slurm output files if cleanup flag is set
if [ "$do_cleanup" = true ]; then
    rm -f /home/exx/users/nicolas/TCC/slurm-*.out
fi

# PARA ENCERRAR O SCRIPT:
# scancel <JOBID>