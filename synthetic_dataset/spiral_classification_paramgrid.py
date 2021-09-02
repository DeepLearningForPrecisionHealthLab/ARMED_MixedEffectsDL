'''
Compares conventional with MEDL models over a range of spiral generation
parameters. This range is defined in the dictConfigSpace variable.

1. conventional neural network (multilevel perceptron)
2. conventional neural network with concatenated cluster membership input
3. mixed-effect neural network which models random effects
4. conventional neural network trained with meta-learning domain generalization (Li 2018)

The results are saved in RESULTSDIR/spirals/paramgrid_<datetime>.
'''

import sys, os
import json
import ray
from ray import tune
from medl.settings import RESULTSDIR
from medl.misc import get_datestamp
from spiral_classification_main import main as main_run

class Comparison(tune.Trainable):
    
    def setup(self, config):
        # Construct argument list
        self.lsConfig = ['--classes', str(config['classes']),
                         '--radius_sd', str(config['radius_sd']),
                         '--confound_sd', str(config['confound_sd']),
                         '--confounded_vars', str(config['confound_vars']),
                         '--noise', str(config['noise']),
                         '--degrees', str(config['degrees']),
                         '--clusters', str(config['clusters']),
                         '--epochs', '100',
                         '--evaluate_test',
                         '--output_dir', self.logdir]
            
        if config['false_negatives']:
            self.lsConfig += ['--false_negatives']
        
    def step(self):
        sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL/')
        dfResults = main_run(self.lsConfig)
        
        dfResultsTest = dfResults.loc[dfResults['Partition'] == 'Test']
        dfResultsTest['Accuracy'] = dfResultsTest['Accuracy'].astype(float)
        dfAccByModel = dfResultsTest.groupby('Model')['Accuracy'].mean()
        
        return dfAccByModel.to_dict()
    
ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0])

# Parameter space
dictConfigSpace = {
    'classes': 2,
    'radius_sd': tune.grid_search([0.1, 0.3, 0.5]),
    'confound_sd': 0.0,
    'confound_vars': 0,
    'noise': tune.grid_search([0.05, 0.1, 0.3]),
    'degrees': tune.grid_search([360, 540, 720]),
    'clusters': tune.grid_search([10, 20, 30]),
    'false_negatives': False,
    }

strOutputDir = os.path.join(RESULTSDIR, 'spirals')
strRunName = 'paramgrid_' + get_datestamp(with_time=True)

os.makedirs(os.path.join(strOutputDir, strRunName), exist_ok=True)
with open(os.path.join(strOutputDir, strRunName, 'simulation_params.json'), 'w') as f:
    json.dump(dictConfigSpace, f, indent=4)

tune.run(Comparison,
         config=dictConfigSpace,
         metric='ME-MLP', # doesn't actually matter since we aren't performing optimization
         mode='max',
         local_dir=strOutputDir,
         name=strRunName,
         stop={'training_iteration': 1}, 
         resources_per_trial={'gpu': 1}, 
         verbose=2,
         num_samples=1, 
         reuse_actors=False)