# QWAS_DQAS

## Code Functions

### Phase1
Run DQAS01.py: Initialize training and sampling using DQAS method to get the current best gate choices and topology.

### Phase2
Run DQAS02.py: Hold the current best gate choices stable and train models by changing topologies.

### Phase3
Run DQAS03.py: Retrieve the best topology from Phase2, train models by changing gate choices.

### Phase4
Run DQAS04.py: Re-train the current best model found in previous phases.

## Requirements

python==3.7.12

tensorflow==2.1.0

tensornetwork==0.3.1

qiskit==0.43.3

qiskit-aer==0.12.2
