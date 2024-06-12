module purge
module load python-3.7.12-gcc-4.8.5-mo7jpky
module load Anaconda3/2020.11
python3.7 -m virtualenv MU_ENV
virtualenv -p python3.7 MU_ENV
source MU_ENV/bin/activate
pip install -r MachineUnlearningUpdate/requirements.txt
pip install -r MachineUnlearningUpdate/opt_requirements.txt
pip install --user ipykernel
pip install ipykernel
ipython kernel install --user --name=MU_ENV
python -m ipykernel install --user --name=MU_ENV