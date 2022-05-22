import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import os

def write_characteristic_table(aags_dir, dataset_grp, dataset_name):

    columns = ['Set', 'Name', 'Pred.', 'DL R.',
               'Facts', 'Derived', 'Rules',
               'Nodes', 'Edges']

    to_str = " & ".join(columns)
    print('\t\t', to_str, '\\\\')
    print('\t\t\\hline\hline')

    values_lists = []
    if aags_dir:
        dir_path = Path(aags_dir)
        for fname in sorted(os.listdir(dir_path)):
            # if 'synthetic_xl' in fname or 'i1K_orig.pickle' in fname:
            #     continue
            if fname.endswith(".pickle"):
                with open(str(dir_path.absolute()) + '/' + fname, 'rb') as output_file:
                    dataset = pickle.load(output_file)
                    aag, nx_aag = dataset['aag'], dataset['nx_aag']

                    predicates = set([pred.name for pred in aag.predicates.values()])
                    dl_rules = set([rule.name for rule in aag.rules.values()])
                    edges = len(nx_aag.edges())
                    nodes = len(nx_aag.nodes)
                    facts = len([n for n in nx_aag.nodes if len(nx_aag.in_edges(n)) == 0])
                    rules = len(aag.rules)
                    derived_facts = nodes - facts - rules
                    name = fname.replace('.pickle','')
                    features = {'Dataset':dataset_grp[name],
                                'Name':dataset_name[name],
                                'Predicates': len(predicates),
                                'DL rules':len(dl_rules),
                                'Facts': facts,
                                'Derived facts': derived_facts,
                                'Rules': rules,
                                'Nodes': nodes,
                                'Edges': edges
                    }
                    values = [str(features[c]) for c in columns]
                    values_lists.append(values)
    values_lists = sorted(values_lists, key=lambda x: (x[0],int(x[-1])))
    for values in values_lists:
        to_str = " & ".join(values)
        print('\t', to_str, '\\\\')
        print('\t\hline')

def rqa(df):

    df['graph_size'] = df['nodes'] + df['edges']
    df['rectify_time_abstract_tot'] = df['rectify_time_abstract'] + df['bisimulation_time']
    df['time_reduction_concrete'] = (df['rectify_time_concrete_naive'] - df['rectify_time_concrete']) / df[
        'rectify_time_concrete_naive']
    rq1_summary = df.groupby(by=['dataset', 'name']).median()[
        ['rectify_time_concrete_naive', 'rectify_time_concrete', 'graph_size']]
    ax = rq1_summary.plot.scatter(x='graph_size', y='rectify_time_concrete_naive', c='blue')
    rq1_summary.plot.scatter(x='graph_size', y='rectify_time_concrete', c='green', ax=ax, marker='x')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('graph size (log scale)')
    plt.ylabel('running time (s, log scale)')

    fig = ax.get_figure()
    fig.savefig('results/rq1')

    timeouts_naive = df[df['rectify_time_concrete_naive'] == -1]
    timeouts_optimized = df[df['rectify_time_concrete'] == -1]
    timeouts_abstract = df[df['rectify_time_abstract'] == -1]

    non_timeouts_naive = df[df['rectify_time_concrete_naive'] != -1]
    non_timeouts_optimized = df[df['rectify_time_concrete'] != -1]
    non_timeouts_abstract = df[df['rectify_time_abstract'] != -1]

    df = df.iloc[non_timeouts_naive.index]
    large_dfs = df[df['graph_size'] > 10000]
    small_dfs = df[df['graph_size'] < 10000]


    print('timeouts naive, optimized, abstract: ', timeouts_naive.shape[0], timeouts_optimized.shape[0],
          timeouts_abstract.shape[0])

    print('small naive, optimized, abstract mean: ', small_dfs['rectify_time_concrete_naive'].mean(), small_dfs['rectify_time_concrete'].mean(),
          small_dfs['rectify_time_abstract_tot'].mean())

    print('small naive, optimized, abstract median: ', large_dfs['rectify_time_concrete_naive'].median(),
          large_dfs['rectify_time_concrete'].median(),
          large_dfs['rectify_time_abstract_tot'].median())

    print('large naive, optimized, abstract mean: ', large_dfs['rectify_time_concrete_naive'].mean(), large_dfs['rectify_time_concrete'].mean(),
          large_dfs['rectify_time_abstract_tot'].mean())

    print('large naive, optimized, abstract median: ', large_dfs['rectify_time_concrete_naive'].median(),
          large_dfs['rectify_time_concrete'].median(),
          large_dfs['rectify_time_abstract_tot'].median())

    small_diffs = (small_dfs['rectify_time_concrete_naive'] - small_dfs['rectify_time_concrete']) / (
    small_dfs['rectify_time_concrete_naive'])

    large_diffs = (large_dfs['rectify_time_concrete_naive'] - large_dfs['rectify_time_concrete']) / (
    large_dfs['rectify_time_concrete_naive'])

    print('mean, median time reduction (repair, small)', small_diffs.mean(), small_diffs.median())
    print('mean, median time reduction (repair, large)', large_diffs.mean(), large_diffs.median())

    small_mean_median = small_dfs.groupby(['dataset', 'name'])['time_reduction_concrete'].median().mean()
    large_mean_median = large_dfs.groupby(['dataset', 'name'])['time_reduction_concrete'].median().mean()
    print('mean-median time reduction (repair, small)', small_mean_median)
    print('mean-median time reduction (repair, large)', large_mean_median)


    small_diffs = 1- (small_dfs['rectify_time_abstract_tot']) / (
    small_dfs['rectify_time_concrete'])

    large_diffs = 1 - (large_dfs['rectify_time_abstract_tot']) / (
    large_dfs['rectify_time_concrete'])

    print('mean, median large time reduction (repair, small, lagfold)', small_diffs.mean(), small_diffs.median())
    print('mean, median large time reduction (repair, large, lagfold)', large_diffs.mean(), large_diffs.median())
    df.to_csv('results/rqa_df_all.csv')
    small_dfs.to_csv('results/rqa_df_small.csv')
    large_dfs.to_csv('results/rqa_df_large.csv')

def rqbc(df):

    df['graph_size'] = df['nodes'] + df['edges']
    df['abs_graph_size'] = df['abstract_nodes'] + df['abstract_edges']
    df['size_reduction'] = df['abstract_nodes'] / df['graph_size']
    df['rectify_time_abstract_total'] = df['rectify_time_abstract'] + df['bisimulation_time']

    df['core_ratio'] = df['abstract_core_final'] / df['core_concrete']
    # print('Core ratio', df['core_ratio'].median())
    print('mean size core reduction:', df['core_ratio'].mean())
    print('median size core reduction:', df['core_ratio'].median())
    rqbc_summary = df.groupby(by=['dataset', 'name']).median()[
        ['graph_size', 'abs_graph_size', 'size_reduction', 'bisimulation_time',
         'rectify_time_concrete', 'rectify_time_abstract',
         'rectify_time_abstract_total',
         'core_concrete', 'abstract_core_final']]

    columns = ["Set", "Name", "$|G|$", "$|G_F|$", "Ratio (\%)", "LAGfold (s)", "LAG-R (s)",
               "LAGfold-R (s)", "LAGfold-R-tot (s)", "Core", "Core LAGfold"]

    to_str = " & ".join(columns)
    print('\t\t', to_str, '\\\\')
    print('\t\t\\hline\hline')

    rqbc_summary.reset_index(inplace=True)
    rqbc_summary.sort_values(["dataset", "graph_size"], inplace=True)

    for r in rqbc_summary.values:
        str_vals = []
        for i, v in enumerate(r):
            if i < 2:
                str_vals.append(v)
            elif i < 4:
                str_vals.append(str(int(v)))
            else:
                str_vals.append(str(round(v, 2)))
        print("\t" + " & ".join(str_vals) + " \\\\")
        print( "\t\\hline")


def summarize_results(results_path, dataset_grp, dataset_name):

    with open(results_path) as fr:

        rows = []
        for i, l in enumerate(fr.readlines()):
            if i == 0:
                columns = l
            else:
                values = l.split(',')
                fname = values[0].replace(".pickle",'')
                if fname not in dataset_name:
                    continue
                values = [float(v) for v in values[1:]]
                dataset = dataset_grp[fname]
                name = dataset_name[fname]
                row = [dataset, name]
                row.extend(values)
                rows.append(row)

        df = pd.DataFrame(columns=['dataset', 'name', 'nodes', 'edges', 'abstract_nodes', 'abstract_edges',
                                       'flip_edges', 'bisimulation_time', 'rectify_time_concrete_naive',
                                       'rectify_time_concrete', 'rectify_time_abstract', 'number_of_rules',
                                       'immutable_rules', 'core_concrete', 'abstract_concrete',
                                       'abstract_core_final'], data=rows)

        # print(df)
        rqa(df.copy())
        rqbc(df.copy())
        print('done')




if __name__ == '__main__':

    aags = {}
    aags_dir = "datasets/pickle/all/"

    dataset_grp = {'1_pulp_and_paper_input':'1', '2_vehicle_assembly_input': '1',
                  'aag_1_Aug2019':'2', 'aag_29Jul2019':'2', 'aag_test':'2',
                  'aag_sg': '2',
                  'toy_input':'2', 'i1K_orig':'2', 'input_159':'2',
                  'original_aag_re':'3', 're_aag':'3',
                  'synthetic_large':'4',
                  'synthetic_medium':'4',
                  'synthetic_small': '4',
                  'synthetic_xl': '4'}

    dataset_name = {'1_pulp_and_paper_input': 'Retail', '2_vehicle_assembly_input': 'Auto.',
                  'aag_1_Aug2019': 'IT (2)', 'aag_29Jul2019': 'IT (3)',
                  'aag_test': 'IT (6)',
                  'aag_sg': 'IT (5)', 'toy_input':'IT (1)',
                  'i1K_orig': 'IT (7)', 'input_159': 'IT (4)',
                  'original_aag_re': 'RE (1)', 're_aag': 'RE (2)',
                  'synthetic_large': 'Syn. L',
                  'synthetic_medium': 'Syn. M',
                  'synthetic_small': 'Syn. S',
                  'synthetic_xl': 'Syn. XL'}

    # write_characteristic_table(aags_dir, dataset_grp, dataset_name)
    results_path = "data/last_results.csv"
    summarize_results(results_path, dataset_grp, dataset_name)
