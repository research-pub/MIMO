# MIMO

Source code and dataset for "MIMO: Mutual Integration of Patient Journey and Medical Ontology for Healthcare Representation Learning"

## Reqirements:

* Pytorch>=1.4.0
* Python3

## Data Preparation
### [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
```bash
mimic_processing.py 
--output output_path/mimic/ 
--admission mimic_path/ADMISSIONS.csv  
--diagnosis mimic_path/DIAGNOSES_ICD.csv 
--single_level ../utils/ccs/ccs_single_dx_tool_2015.csv 
--multi_level ../utils/ccs/ccs_multi_dx_tool_2015.csv
```

### [eICU](https://physionet.org/content/eicu-crd/2.0/)
```bash
eicu_processing.py 
--output output_path/eicu/ 
--patient eicu_path/patient.csv  
--diagnosis eciu/diagnosis.csv
--single_level ../utils/ccs/ccs_single_dx_tool_2015.csv 
--multi_level ../utils/ccs/ccs_multi_dx_tool_2015.csv
```

### Knowledge Graph Building

```bash
data_graph_building.py 
--output output_path/mimic/  
--seqs output_path/mimic/inputs_all.seqs 
--vocab output_path/mimic/vocab.txt 
--multi_level ../utils/ccs/ccs_multi_dx_tool_2015.csv
```

##  Model Training:Validating:Testing
```bash
mimo_train.py 
--output_dir model_output_path 
--data_dir processed_data_path 
--data_source mimic
--num_train_epochs 100 
--train_batch_size 32 
--gpu 2 
--learning_rate 0.1 
 --task next_dx
```