import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


def prepare_design_matrix_crosstable():
    df = pd.read_csv('data/41588_2018_78_MOESM4_ESM.txt', sep='\t', low_memory=False, skiprows=1)

    exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']

    df = df[~df['Variant_Classification'].isin(exclude)].copy()
    df_table = pd.pivot_table(data=df, index='Tumor_Sample_Barcode', columns='Hugo_Symbol',
                              values='Variant_Classification',
                              aggfunc='count')
    df_table = df_table.fillna(0)
    df_table.to_csv('data/P1000_final_analysis_set_cross_important_only.csv')


def prepare_response():
    df = pd.read_excel('data/41588_2018_78_MOESM5_ESM.xlsx', sheet_name='Supplementary_Table3.txt',
                       skiprows=2)
    response = pd.DataFrame()
    response['id'] = df['Patient.ID']
    response['response'] = df['Sample.Type']
    response['response'] = response['response'].replace('Metastasis', 1)
    response['response'] = response['response'].replace('Primary', 0)
    response = response.drop_duplicates()
    response.to_csv('data/response_paper.csv', index=False)


def prepare_cnv():
    df = pd.read_csv('data/41588_2018_78_MOESM10_ESM.txt', sep='\t', low_memory=False, skiprows=1,
                     index_col=0)
    df = df.T
    df = df.fillna(0.)
    df.to_csv('data/P1000_data_CNA_paper.csv')


def load_data(filename, selected_genes=None):
    data = pd.read_csv(filename, index_col=0)

    labels = pd.read_csv('data/response_paper.csv')
    labels = labels.set_index('id')

    # join with the labels
    all = data.join(labels, how='inner')
    all = all[~all['response'].isnull()]

    response = all['response']
    info = all.index

    genes = set(all.columns)
    genes = genes.intersection(selected_genes)
    x = all.loc[:, list(genes)]

    return x, response, info, genes


def load_data_type(data_type='gene', cnv_levels=5, cnv_filter_single_event=True, mut_binary=True,
                   selected_genes=None):
    if data_type == 'mut_important':
        x, response, info, genes = load_data('data/P1000_final_analysis_set_cross_important_only.csv',
                                             selected_genes)
        if mut_binary:
            x[x > 1.] = 1.
    elif data_type == 'cnv_del':
        x, response, info, genes = load_data('data/P1000_data_CNA_paper.csv', selected_genes)
        x[x >= 0.0] = 0.
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == -1.] = 0.0
                x[x == -2.] = 1.0
            else:
                x[x < 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == -1.] = 0.5
            x[x == -2.] = 1.0
    elif data_type == 'cnv_amp':
        x, response, info, genes = load_data('data/P1000_data_CNA_paper.csv', selected_genes)
        x[x <= 0.0] = 0.
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == 1.0] = 0.0
                x[x == 2.0] = 1.0
            else:
                x[x > 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == 1.] = 0.5
            x[x == 2.] = 1.0
    else:
        raise NotImplementedError
    return x, response, info, genes


def combine(x_list, y_list, rows_list, cols_list, data_type_list, use_coding_genes_only=True):
    cols = set.union(*cols_list)

    if use_coding_genes_only:
        coding_genes_df = pd.read_csv('data/protein-coding_gene_with_coordinate_minimal.txt', sep='\t',
                                      header=None)
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'])
        cols = cols.intersection(coding_genes)

    # Ensure reproducibility
    cols = list(cols)
    cols.sort()
    np.random.seed(0)
    np.random.shuffle(cols)

    all_cols_df = pd.DataFrame(index=cols)

    df_list = []
    for x, r, c in zip(x_list, rows_list, cols_list):
        df = pd.DataFrame(x, index=list(r), columns=list(c))
        df = df.T.join(all_cols_df, how='right')
        df = df.T
        df = df.fillna(0)
        df_list.append(df)

    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1)
    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)

    x = all_data.values

    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y_list[0], how='left')
    y = y.values

    cols = all_data.columns
    rows = all_data.index

    return x, y, rows, cols


def load_final(data_type, cnv_levels):
    if not os.path.isfile('data/P1000_final_analysis_set_cross_important_only.csv'):
        prepare_design_matrix_crosstable()
    if not os.path.isfile('data/response_paper.csv'):
        prepare_response()
    if not os.path.isfile('data/P1000_data_CNA_paper.csv'):
        prepare_cnv()

    df = pd.read_csv('data/tcga_prostate_expressed_genes_and_cancer_genes.csv', header=0)
    selected_genes = list(df['genes'])

    x_list, y_list, rows_list, cols_list = [], [], [], []
    for t in data_type:
        x, y, rows, cols = load_data_type(t, cnv_levels, selected_genes=selected_genes)
        x_list.append(x), y_list.append(y), rows_list.append(rows), cols_list.append(cols)

    x, y, rows, cols = combine(x_list, y_list, rows_list, cols_list, data_type)
    return x, y, rows, cols
