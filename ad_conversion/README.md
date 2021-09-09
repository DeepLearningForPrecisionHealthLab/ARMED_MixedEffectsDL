## 1. Nested K-folds setup
Run `create_kfolds.py` to create 10x10 nested K-folds, stratified by conversion and by site. These K-folds are saved as a [NestedKFoldUtil](../medl/crossvalidation/splitting.py) object, which is pickled to a file. 

## 2. Model training
1. Conventional MLP: 
An HPO search using Bayesian Optimization is run using 
```
python conventional_nn_hpo.py --output_dir </path/to/output/directory> --outer_fold <outer_fold_to_run>
```
for each outer cross-validation fold (run again for each outer fold or use a for loop). This creates a subdirectory for each outer fold. An independent HPO search is performed for each outer fold and the best performing model (highest mean Youden's index on inner validation sets) is selected. This model is trained and evaluated on the outer training/test fold and saved in fold<nn>/best_model. 

2. Site input MLP:
This should be run after completing the HPO for the conventional MLP. Using this script:
```
python siteinput_nn.py --hpo_dir </path/to/conventional_nn/output> --output_dir </path/to/output/directory>
```
the best performing model from each outer fold is selected and modified to use an additional site input (through concatenation with the main input). This new model is trained and evaluated on the outer training/test fold and saved in fold<nn>/best_model. 

3. ME-MLP
A separate HPO is needed to optimize the additional random effect hyperparameters of the mixed effects model:
```
python me_nn_hpo.py --output_dir </path/to/output/directory> --outer_fold <outer_fold_to_run>
```
Results are saved in the same manner as the conventional MLP.

## 3. Collect results
For convenience, this script collects results from each outer fold into a single dataframe. 
```
python collect_nn_hpo_results.py --output_dir </path/to/output/directory>
```

This creates a combined table of test performance and another for the best
hyperparameters of each outer fold.

## 4. Measure feature importance
Finally, estimate feature importance for each model by computing the partial derivatives of the model output w.r.t. each input feature. 

For the conventional MLP:
```
python nn_feature_importance.py --output_dir </path/to/output/directory>
```

For the site input MLP:
```
python nn_feature_importance.py --output_dir </path/to/output/directory> --site_input
```

For the ME-MLP:
```
python nn_feature_importance.py --output_dir </path/to/output/directory> --site_input --zero_re
```

This produces a long table containing the feature importance for every training sample in every outer fold. It also produces a box plot showing the distribution of importance across samples for each feature and a scatter-plot of median importance vs. H-statistic. This H-statistic is computed via the Kruskal Wallis test and measures the difference in importance across sites.