# License: BSD 3 clause
"""
Functions related to running experiments and parsing configuration files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Chee Wee Leong (cleong@ets.org)
"""

import csv
import json
import math
import sys
from collections import defaultdict
from os.path import exists, join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
import seaborn as sns

from skll.utils.logging import get_skll_logger

# Turn off interactive plotting for matplotlib
plt.ioff()


def _compute_ylimits_for_featureset(df, metrics):
    """
    Compute the y-limits for learning curve plots.

    Parameters
    ----------
    df : pd.DataFrame
        A data_frame with relevant metric information for
        train and test.
    metrics : list of str
        A list of metrics for learning curve plots.

    Returns
    -------
    ylimits : dict
        A dictionary, with metric names as keys
        and a tuple of (lower_limit, upper_limit) as values.
    """

    # set the y-limits of the curves depending on what kind
    # of values the metric produces
    ylimits = {}
    for metric in metrics:
        # get the real min and max for the values that will be plotted
        df_train = df[(df['variable'] == 'train_score_mean') & (df['metric'] == metric)]
        df_test = df[(df['variable'] == 'test_score_mean') & (df['metric'] == metric)]
        train_values_lower = df_train['value'].values - df_train['train_score_std'].values
        test_values_lower = df_test['value'].values - df_test['test_score_std'].values
        min_score = np.min(np.concatenate([train_values_lower,
                                           test_values_lower]))
        train_values_upper = df_train['value'].values + df_train['train_score_std'].values
        test_values_upper = df_test['value'].values + df_test['test_score_std'].values
        max_score = np.max(np.concatenate([train_values_upper,
                                           test_values_upper]))

        # squeeze the limits to hide unnecessary parts of the graph
        # set the limits with a little buffer on either side but not too much
        if min_score < 0:
            lower_limit = max(min_score - 0.1, math.floor(min_score) - 0.05)
        else:
            lower_limit = 0

        if max_score > 0:
            upper_limit = min(max_score + 0.1, math.ceil(max_score) + 0.05)
        else:
            upper_limit = 0

        ylimits[metric] = (lower_limit, upper_limit)

    return ylimits


def generate_learning_curve_plots(experiment_name,
                                  output_dir,
                                  learning_curve_tsv_file):
    """
    Generate the learning curve plots given the TSV output
    file from a learning curve experiment.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment.
    output_dir : str
        Path to the output directory for the plots.
    learning_curve_tsv_file : str
        The path to the learning curve TSV file.
    """

    # use pandas to read in the TSV file into a data frame
    # and massage it from wide to long format for plotting
    df = pd.read_csv(learning_curve_tsv_file, sep='\t')
    num_learners = len(df['learner_name'].unique())
    num_metrics = len(df['metric'].unique())
    df_melted = pd.melt(df, id_vars=[c for c in df.columns
                                     if c not in ['train_score_mean', 'test_score_mean']])

    # if there are any training sizes greater than 1000,
    # then we should probably rotate the tick labels
    # since otherwise the labels are not clearly rendered
    rotate_labels = np.any([size >= 1000 for size in df['training_set_size'].unique()])

    # set up and draw the actual learning curve figures, one for
    # each of the featuresets
    for fs_name, df_fs in df_melted.groupby('featureset_name'):
        fig = plt.figure()
        fig.set_size_inches(2.5 * num_learners, 2.5 * num_metrics)

        # compute ylimits for this feature set for each objective
        with sns.axes_style('whitegrid', {"grid.linestyle": ':',
                                          "xtick.major.size": 3.0}):
            g = sns.FacetGrid(df_fs, row="metric", col="learner_name",
                              hue="variable", height=2.5, aspect=1,
                              margin_titles=True, despine=True, sharex=False,
                              sharey=False, legend_out=False, palette="Set1")
            colors = train_color, test_color = sns.color_palette("Set1")[:2]
            g = g.map_dataframe(sns.pointplot, "training_set_size", "value",
                                scale=.5, ci=None)
            ylimits = _compute_ylimits_for_featureset(df_fs, g.row_names)
            for ax in g.axes.flat:
                plt.setp(ax.texts, text="")
            g = (g.set_titles(row_template='', col_template='{col_name}')
                 .set_axis_labels('Training Examples', 'Score'))
            if rotate_labels:
                g = g.set_xticklabels(rotation=60)

            for i, row_name in enumerate(g.row_names):
                for j, col_name in enumerate(g.col_names):
                    ax = g.axes[i][j]
                    ax.set(ylim=ylimits[row_name])
                    df_ax_train = df_fs[(df_fs['learner_name'] == col_name) &
                                        (df_fs['metric'] == row_name) &
                                        (df_fs['variable'] == 'train_score_mean')]
                    df_ax_test = df_fs[(df_fs['learner_name'] == col_name) &
                                       (df_fs['metric'] == row_name) &
                                       (df_fs['variable'] == 'test_score_mean')]
                    ax.fill_between(list(range(len(df_ax_train))),
                                    df_ax_train['value'] - df_ax_train['train_score_std'],
                                    df_ax_train['value'] + df_ax_train['train_score_std'],
                                    alpha=0.1,
                                    color=train_color)
                    ax.fill_between(list(range(len(df_ax_test))),
                                    df_ax_test['value'] - df_ax_test['test_score_std'],
                                    df_ax_test['value'] + df_ax_test['test_score_std'],
                                    alpha=0.1,
                                    color=test_color)
                    if j == 0:
                        ax.set_ylabel(row_name)
                        if i == 0:
                            # set up the legend handles for this plot
                            plot_handles = [matplotlib.lines.Line2D([],
                                                                    [],
                                                                    color=c,
                                                                    label=l,
                                                                    linestyle='-')
                                            for c, l in zip(colors, ['Training',
                                                                     'Cross-validation'])]
                            ax.legend(handles=plot_handles,
                                      loc=4,
                                      fancybox=True,
                                      fontsize='x-small',
                                      ncol=1,
                                      frameon=True)
            g.fig.tight_layout(w_pad=1)
            plt.savefig(join(output_dir, f'{experiment_name}_{fs_name}.png'),
                        dpi=300)
            # explicitly close figure to save memory
            plt.close(fig)


def _print_fancy_output(learner_result_dicts, output_file=sys.stdout):
    """
    Function to take all of the results from all of the folds and print
    nice tables with the results.

    Parameters
    ----------
    learner_result_dicts : list of str
        A list of paths to the individual result JSON files.
    output_file : file buffer, optional
        The file buffer to print to.
        Defaults to ``sys.stdout``.
    """
    if not learner_result_dicts:
        raise ValueError('Result dictionary list is empty!')

    lrd = learner_result_dicts[0]
    print(f'Experiment Name: {lrd["experiment_name"]}', file=output_file)
    print(f'SKLL Version: {lrd["version"]}', file=output_file)
    print(f'Training Set: {lrd["train_set_name"]}', file=output_file)
    print(f'Training Set Size: {lrd["train_set_size"]}', file=output_file)
    print(f'Test Set: {lrd["test_set_name"]}', file=output_file)
    print(f'Test Set Size: {lrd["test_set_size"]}', file=output_file)
    print(f'Shuffle: {lrd["shuffle"]}', file=output_file)
    print(f'Feature Set: {lrd["featureset"]}', file=output_file)
    print(f'Learner: {lrd["learner_name"]}', file=output_file)
    print(f'Task: {lrd["task"]}', file=output_file)
    if lrd['folds_file']:
        print(f'Specified Folds File: {lrd["folds_file"]}', file=output_file)
    if lrd['task'] == 'cross_validate':
        print(f'Number of Folds: {lrd["cv_folds"]}', file=output_file)
        if not lrd['cv_folds'].endswith('folds file'):
            print(f'Stratified Folds: {lrd["stratified_folds"]}',
                  file=output_file)
    print(f'Feature Scaling: {lrd["feature_scaling"]}', file=output_file)
    print(f'Grid Search: {lrd["grid_search"]}', file=output_file)
    if lrd['grid_search']:
        print(f'Grid Search Folds: {lrd["grid_search_folds"]}',
              file=output_file)
        print(f'Grid Objective Function: {lrd["grid_objective"]}',
              file=output_file)
    if (lrd['task'] == 'cross_validate' and
            lrd['grid_search'] and
            lrd['cv_folds'].endswith('folds file')):
        print('Using Folds File for Grid Search: '
              f'{lrd["use_folds_file_for_grid_search"]}',
              file=output_file)
    if lrd['task'] in ['evaluate', 'cross_validate'] and lrd['additional_scores']:
        print('Additional Evaluation Metrics: '
              f'{list(lrd["additional_scores"].keys())}',
              file=output_file)
    print(f'Scikit-learn Version: {lrd["scikit_learn_version"]}',
          file=output_file)
    print(f'Start Timestamp: {lrd["start_timestamp"]}', file=output_file)
    print(f'End Timestamp: {lrd["end_timestamp"]}', file=output_file)
    print(f'Total Time: {lrd["total_time"]}', file=output_file)
    print('\n', file=output_file)

    for lrd in learner_result_dicts:
        print(f'Fold: {lrd["fold"]}', file=output_file)
        print(f'Model Parameters: {lrd.get("model_params", "")}',
              file=output_file)
        print(f'Grid Objective Score (Train) = {lrd.get("grid_score", "")}',
              file=output_file)
        if 'result_table' in lrd:
            print(lrd['result_table'], file=output_file)
            print(f'Accuracy = {lrd["accuracy"]}', file=output_file)
        if 'descriptive' in lrd:
            print('Descriptive statistics:', file=output_file)
            for desc_stat in ['min', 'max', 'avg', 'std']:
                actual = lrd['descriptive']['actual'][desc_stat]
                predicted = lrd['descriptive']['predicted'][desc_stat]
                print(f' {desc_stat.title()} = {actual:.4f} (actual), '
                      f'{predicted:.4f} (predicted)',
                      file=output_file)
            print(f'Pearson = {lrd["pearson"]:f}', file=output_file)
        print(f'Objective Function Score (Test) = {lrd.get("score", "")}',
              file=output_file)

        # now print the additional metrics, if there were any
        if lrd['additional_scores']:
            print('', file=output_file)
            print('Additional Evaluation Metrics (Test):', file=output_file)
            for metric, score in lrd['additional_scores'].items():
                score = '' if np.isnan(score) else score
                print(f' {metric} = {score}', file=output_file)
        print('', file=output_file)


def _write_learning_curve_file(result_json_paths, output_file):
    """
    Function to take a list of paths to individual learning curve
    results json files and writes out a single TSV file with the
    learning curve data.

    Parameters
    ----------
    result_json_paths : list of str
        A list of paths to the individual result JSON files.
    output_file : str
        The path to the output file (TSV format).
    """

    learner_result_dicts = []

    # Map from feature set names to all features in them
    logger = get_skll_logger('experiment')
    for json_path in result_json_paths:
        if not exists(json_path):
            logger.error(f'JSON results file {json_path} not found. Skipping '
                         'summary creation. You can manually create the '
                         'summary file after the fact by using the '
                         'summarize_results script.')
            return
        else:
            with open(json_path) as json_file:
                obj = json.load(json_file)
                learner_result_dicts.extend(obj)

    # Build and write header
    header = ['featureset_name', 'learner_name', 'metric',
              'train_set_name', 'training_set_size', 'train_score_mean',
              'test_score_mean', 'train_score_std', 'test_score_std',
              'scikit_learn_version', 'version']
    writer = csv.DictWriter(output_file,
                            header,
                            extrasaction='ignore',
                            dialect=csv.excel_tab)
    writer.writeheader()

    # write out the fields we need for the learning curve file
    # specifically, we need to separate out the curve sizes
    # and scores into individual entries.
    for lrd in learner_result_dicts:
        training_set_sizes = lrd['computed_curve_train_sizes']
        train_scores_means_by_size = lrd['learning_curve_train_scores_means']
        test_scores_means_by_size = lrd['learning_curve_test_scores_means']
        train_scores_stds_by_size = lrd['learning_curve_train_scores_stds']
        test_scores_stds_by_size = lrd['learning_curve_test_scores_stds']

        # rename `grid_objective` to `metric` since the latter name can be confusing
        lrd['metric'] = lrd['grid_objective']

        for (size,
             train_score_mean,
             test_score_mean,
             train_score_std,
             test_score_std) in zip(training_set_sizes,
                                    train_scores_means_by_size,
                                    test_scores_means_by_size,
                                    train_scores_stds_by_size,
                                    test_scores_stds_by_size):
            lrd['training_set_size'] = size
            lrd['train_score_mean'] = train_score_mean
            lrd['test_score_mean'] = test_score_mean
            lrd['train_score_std'] = train_score_std
            lrd['test_score_std'] = test_score_std

            writer.writerow(lrd)

    output_file.flush()


def _write_skll_folds(skll_fold_ids, skll_fold_ids_file):
    """
    Function to take a dictionary of id->test-fold-number and
    write it to a file.

    Parameters
    ----------
    skll_fold_ids : dict
        Dictionary with ids as keys and test-fold-numbers as values.
    skll_fold_ids_file : file buffer
        An open file handler to write to.
    """

    f = csv.writer(skll_fold_ids_file)
    f.writerow(['id', 'cv_test_fold'])
    for example_id in skll_fold_ids:
        f.writerow([example_id, skll_fold_ids[example_id]])

    skll_fold_ids_file.flush()


def _write_summary_file(result_json_paths, output_file, ablation=0):
    """
    Function to take a list of paths to individual result
    json files and returns a single file that summarizes
    all of them.

    Parameters
    ----------
    result_json_paths : list of str
        A list of paths to the individual result JSON files.
    output_file : str
        The path to the output file (TSV format).
    ablation : int, optional
        The number of features to remove when doing ablation experiment.
        Defaults to 0.
    """
    learner_result_dicts = []
    # Map from feature set names to all features in them
    all_features = defaultdict(set)
    logger = get_skll_logger('experiment')
    for json_path in result_json_paths:
        if not exists(json_path):
            logger.error(f'JSON results file {json_path} not found. Skipping '
                         'summary creation. You can manually create the '
                         'summary file after the fact by using the '
                         'summarize_results script.')
            return
        else:
            with open(json_path) as json_file:
                obj = json.load(json_file)
                featureset_name = obj[0]['featureset_name']
                if ablation != 0 and '_minus_' in featureset_name:
                    parent_set = featureset_name.split('_minus_', 1)[0]
                    all_features[parent_set].update(
                        yaml.safe_load(obj[0]['featureset']))
                learner_result_dicts.extend(obj)

    # Build and write header
    header = set(learner_result_dicts[0].keys()) - {'result_table',
                                                    'descriptive'}
    if ablation != 0:
        header.add('ablated_features')
    header = sorted(header)
    writer = csv.DictWriter(output_file,
                            header,
                            extrasaction='ignore',
                            dialect=csv.excel_tab)
    writer.writeheader()

    # Build "ablated_features" list and fix some backward compatible things
    for lrd in learner_result_dicts:
        featureset_name = lrd['featureset_name']
        if ablation != 0:
            parent_set = featureset_name.split('_minus_', 1)[0]
            ablated_features = all_features[parent_set].difference(
                yaml.safe_load(lrd['featureset']))
            lrd['ablated_features'] = ''
            if ablated_features:
                lrd['ablated_features'] = json.dumps(sorted(ablated_features))

        # write out the new learner dict with the readable fields
        writer.writerow(lrd)

    output_file.flush()
