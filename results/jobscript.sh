#!/bin/bash
#PBS -l walltime=20:00:00
#PBS -l vmem=2gb
#PBS -l nodes=1:ppn=1
#PBS -d /home/tretjakov/projects/research/joonatan/Neural-network-research

echo Starting NN task $1
echo "Start time: `date`"

/storage/software/octave-3.6.4/bin/octave -qf trainNN.m $1

echo "Completed NN task $1"
echo "End time: `date`"

