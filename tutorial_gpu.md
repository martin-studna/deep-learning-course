ssh alifuk@skirit.ics.muni.cz


qsub -l walltime=1:0:0 -q gpu@cerit-pbs.cerit-sc.cz -l select=1:ncpus=1:ngpus=1:mem=4000mb:scratch_local=4000mb:gpu_cap=cuda80 -I

ls /cvmfs/singularity.metacentrum.cz
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:21.02-tf2-py3.SIF

git clone https://github.com/martin-studna/deep-learning-course.git
cd deep-learning-course/03
git checkout albert_week3

pip3 install neptune-client


python3 uppercase.py








-------------------------------------------------------- OLDDD




qsub -I -l walltime=1:0:0 -q gpu@cerit-pbs.cerit-sc.cz -l select=1:ncpus=1:ngpus=1:mem=4000mb:scratch_local=2000mb:gpu_cap=cuda75

lspci
nvidia-smi


cd $SCRATCHDIR

python -m venv bbb
source bbb/bin/activate


cd $SCRATCHDIR
git clone https://github.com/martin-studna/deep-learning-course.git
cd deep-learning-course/03
git checkout albert_week3

module add python/3.8.0-gcc-rab6t
module add cudnn-7.6.4-cuda10.0 cudnn-7.6.4-cuda10.1
module add tensorflow-2.0.0-gpu-python3    #obsahuje python a nefunguje
pip3 install neptune-client

python3 uppercase.py



module add python/3.8.0-gcc-rab6t

module add tensorflow-1.7.1-gpu-python3                                       #má v sobě python, ale aspoň se načte
module add cudnn/cudnn-7.6.5.32-10.2-linux-x64-gcc-6.3.0-xqx4s5f






