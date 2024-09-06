# QWAS_DQAS

## Code Functions

### Phase1
Run DQAS01.py: Initialize training and sampling using DQAS method to get the gate choices and topology with highest test accuracy.

### Phase2
Run DQAS02.py: Hold the current best gate choices stable and train models by changing topologies.

### Phase3
Run DQAS03.py: Retrieve the best topology from Phase2, train models by sampling new gate choices using DQAS method.

### Repeat Phase2 and Phase3 to get new current best gate choices and topology.

### Phase4
Run DQAS04.py: Re-train the current best model found in previous phases.

### To run the entire process that includes all four phases, execute the 'run.py' file.

## Requirements

### pip install -r requirement.txt

python==3.7

torch==1.13.1

tensorflow==2.1.0

scipy==1.7.3

qiskit==0.43.3

torchpack==0.3.1

matplotlib==3.5.3

pathos==0.2.5

protobuf==3.8.0
