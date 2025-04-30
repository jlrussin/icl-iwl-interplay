import os
import pickle
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns

from itertools import product

def load_results(labels, fns, all_ctx_conds=False, is_ctx_step=False, 
                 rot_cond=None):
    """
    Load results from files based on the provided seeds and filenames.
    Args:
        seeds (list): List of seeds to load results for.
        fns (list): List of filenames corresponding to the seeds.
        all_ctx_conds (bool): If True, load all context conditions.
        is_ctx_step (bool): If True, load context step conditions.
    """
    assert len(labels) == len(fns)

    if rot_cond is None:
        rot_conds = ['rotated', 'unrotated']
    else:
        rot_conds = [rot_cond]
    if all_ctx_conds:
        ctx_conds = ['aligned', 'misaligned', 'blocked', 'interleaved']
        stp_conds = ['blocked', 'blocked', 'blocked', 'interleaved']
    else:
        if is_ctx_step:
            assert not all_ctx_conds
            ctx_conds = ['blocked', 'blocked', 'interleaved', 'interleaved']
            stp_conds = ['blocked', 'interleaved', 'blocked', 'interleaved']
        else:
            ctx_conds = ['blocked', 'interleaved']
            stp_conds = ['blocked', 'interleaved']

    results = {}
    for label, fn in zip(labels, fns):
        r = {}
        for rot, (ctx, stp) in product(rot_conds, zip(ctx_conds, stp_conds)):
            path = fn + f'_ft_{rot}_{ctx}_{stp}_run0.pickle'
            if os.path.exists(path):
                key = (rot, ctx, stp)
                with open(path, 'rb') as f:
                    r[key] = pickle.load(f)
            else:
                print(f'File not found: {path}')
        if len(r) == len(rot_conds) * len(ctx_conds):
            results[label] = r
        else:
            print(f'Incomplete data for label {label}')
    return results

def get_dataframes(results, n_groups=2, thresh=None, task='category',
                   thresh_rotation='Rule-like', thresh_curriculum='Blocked',
                   is_ctx_step=False):
    # Make numpy arrays
    result_arrays = {}
    for seed, result in results.items():
        # check if initial test data exists in all conditions
        if not all(['initial_test' in r for r in result.values()]):
            print(f'Skipping seed {seed}, missing initial test data')
            continue
        
        for key, r in result.items():
            rotation, curriculum_ctx, curriculum_stp = key
            # Check if curriculum conditions match
            if is_ctx_step:
                cond_key = (seed, rotation, curriculum_ctx, curriculum_stp)
            else:
                if curriculum_ctx != curriculum_stp:
                    continue
                cond_key = (seed, rotation, curriculum_ctx)

            # Get group order
            train_order = tuple(r['block_name_data'][0])
            assert all([tuple(o) == train_order for o in r['block_name_data']])
            if 'train_all' in train_order:
                assert all([g == 'train_all' for g in train_order])
                assert curriculum_stp == 'interleaved'
                # Order doesn't matter - groups trained simultaneously
                if task == 'category':
                    group_order = {i:i for i in range(n_groups)}
                elif task == 'grid':
                    group_order = {i:string.ascii_uppercase[i] for i in range(n_groups)}
            else:
                group_order = {} 
                unique_order = list(dict.fromkeys(train_order))
                for i, group_name in enumerate(unique_order):
                    msg = f'seed{seed}, {group_name}, {key}'
                    assert group_name.startswith('train'), msg
                    if len(group_name) != 6:
                        # skip this seed - probably came from an older file
                        print(f'Skipping {msg}, probably an older file')
                        break
                    group_id = group_name[-1]
                    if group_id.isdigit():
                        group_order[i] = int(group_id) # 0, 1
                    else:
                        assert group_id.isalpha() and group_id.isupper() # A, B
                        group_order[i] = group_id
            if len(group_order) != n_groups:
                break

            # Group names
            group_names = []
            new_names = []
            if task == 'category': 
                for i in range(n_groups):
                    for s in ['train', 'test']:
                        group_name = f'{s}{group_order[i]}'
                        new_name = s.capitalize() + string.ascii_uppercase[i]
                        group_names.append(group_name)
                        new_names.append(new_name)
            elif task == 'grid':
                for i in range(n_groups):
                    group_name = f'train{group_order[i]}'
                    new_name = 'Train' + string.ascii_uppercase[i]
                    group_names.append(group_name)
                    new_names.append(new_name)
                group_names.append('test')
                new_names.append('Test')

            # Loss data
            loss_data = np.array(r['loss_data']) # [episodes, blocks, steps]
            n_episodes, n_blocks, n_steps = loss_data.shape
            loss_data = loss_data.reshape(n_episodes, n_blocks*n_steps)

            # Accuracy data
            acc_data = {}
            for group_name, new_name in zip(group_names, new_names):
                episodes = []
                for episode in r['acc_data']:
                    blocks = []
                    for block in episode:
                        blocks.append(block[group_name])
                    episodes.append(blocks)
                episodes = np.array(episodes) # [episodes, blocks, steps]
                assert episodes.shape[0] == n_episodes
                assert episodes.shape[1] == n_blocks
                n_tests = episodes.shape[2]
                acc_data[new_name] = episodes.reshape(n_episodes, n_blocks*n_tests)

            # Downsample loss data to match accuracy data (n_tests)
            total_steps = n_blocks * n_steps
            total_tests = n_blocks * n_tests
            indices = np.linspace(0, total_steps-1, total_tests, dtype=int)
            loss_data = loss_data[:, indices]

            # Check shapes
            assert loss_data.shape[1] == total_tests
            for k, v in acc_data.items():
                assert v.shape[1] == total_tests
            
            # Add to dictionary
            result_arrays[cond_key] = {'loss': loss_data}
            for k, v in acc_data.items():
                result_arrays[cond_key][k] = v

    good_seeds = set([key[0] for key in result_arrays.keys()])
    # Make dataframe
    test_every = 20
    rows = []
    for key, result_dict in result_arrays.items():
        if is_ctx_step:
            seed, rotation, curriculum_ctx, curriculum_stp = key
        else:
            seed, rotation, curriculum = key
        n_episodes = result_dict['loss'].shape[0]
        n_tests = result_dict['loss'].shape[1]        
        for episode_i in range(n_episodes):
            for test_i in range(n_tests):
                row = {'Seed': seed, 'Rotation': rotation, 
                       'Episode': episode_i, 
                       'Step': test_i*test_every}
                if is_ctx_step:
                    row['Curriculum (context)'] = curriculum_ctx
                    row['Curriculum (step)'] = curriculum_stp
                else:
                    row['Curriculum'] = curriculum
                for k, array in result_dict.items():
                    row[k] = array[episode_i, test_i]
                rows.append(row)

    # Dataframe
    df = pd.DataFrame(rows)

    # Capitalize Rotation and Curriculum
    df['Rotation'] = df['Rotation'].str.capitalize()
    if is_ctx_step:
        df['Curriculum (context)'] = df['Curriculum (context)'].str.capitalize()
        df['Curriculum (step)'] = df['Curriculum (step)'].str.capitalize()
        id_vars = ['Seed', 'Rotation', 'Curriculum (context)',
                   'Curriculum (step)', 'Step']
    else:
        df['Curriculum'] = df['Curriculum'].str.capitalize()
        id_vars = ['Seed', 'Rotation', 'Curriculum', 'Step']
        

    # Average over episodes
    df_avg = df.groupby(id_vars).mean().reset_index()
    df_avg = df_avg.drop(columns=['Episode'])

    # Melt data so all train and test accuracies are in one column
    df_melted = df_avg.drop(columns=['loss'])
    df_melted = df_melted.melt(id_vars=id_vars, 
                               var_name='test_condition', value_name='Accuracy')

    # Add Group column from last character of test_condition
    if task == 'category':
        df_melted['Group'] = df_melted['test_condition'].str[-1] # A or B
        df_melted['Split'] = df_melted['test_condition'].str[:-1] # Train or Test
    elif task == 'grid':
        df_melted['Split'] = df_melted['test_condition']

    # Get few-shot results
    first_seed = list(results.keys())[0]
    first_key = list(results[first_seed].keys())[0]
    n_episodes = len(results[first_seed][first_key]['initial_test'])
    few_shot_results = []
    for seed, result in results.items():
        if seed not in good_seeds:
            continue
        for key, r in result.items():
            rotation, curriculum_ctx, curriculum_stp = key
            test_accs = []
            for episode in r['initial_test']:
                for k, v in episode.items():
                    if 'test' in k:
                        test_accs.append(v)
            ave_acc = np.mean(test_accs)
            row = {'Seed': seed, 'Rotation': rotation, 
                   'Curriculum': curriculum_ctx, 
                   'Accuracy': ave_acc}
            few_shot_results.append(row)

    few_shot_df = pd.DataFrame(few_shot_results)

    # Capitalize Rotation and Curriculum
    few_shot_df['Rotation'] = few_shot_df['Rotation'].str.capitalize()
    few_shot_df['Curriculum'] = few_shot_df['Curriculum'].str.capitalize()

    # Get finetune results
    max_step = df.groupby(['Seed', 'Episode']).max().reset_index()
    assert max_step['Step'].nunique() == 1
    max_step = max_step['Step'].iloc[0]
    finetune_df = df_avg[df_avg['Step'] == max_step]
    finetune_df = finetune_df.drop(columns=['Step'])
    finetune_df['Accuracy'] = (finetune_df['TrainA'] + finetune_df['TrainB']) / 2

    # Change variable names
    df_melted['Rotation'] = df_melted['Rotation'].replace('Unrotated', 'Rule-like')
    df_avg['Rotation'] = df_avg['Rotation'].replace('Unrotated', 'Rule-like')
    few_shot_df['Rotation'] = few_shot_df['Rotation'].replace('Unrotated', 'Rule-like')
    finetune_df['Rotation'] = finetune_df['Rotation'].replace('Unrotated', 'Rule-like')

    # Filter seeds by accuracy
    if thresh is not None:
        accs_by_seed = few_shot_df[(few_shot_df['Rotation'] == thresh_rotation) &
                                   (few_shot_df['Curriculum'] == thresh_curriculum)]
        good_seeds = accs_by_seed[accs_by_seed['Accuracy'] > thresh]['Seed']
        df = df[df['Seed'].isin(good_seeds)]
        df_avg = df_avg[df_avg['Seed'].isin(good_seeds)]
        df_melted = df_melted[df_melted['Seed'].isin(good_seeds)]
        few_shot_df = few_shot_df[few_shot_df['Seed'].isin(good_seeds)]
        finetune_df = finetune_df[finetune_df['Seed'].isin(good_seeds)]
        bad_seeds = set(accs_by_seed['Seed']) - set(good_seeds)
        if bad_seeds:
            print(f"Removed seeds {bad_seeds} due to threshold {thresh}")

    return df, df_avg, df_melted, few_shot_df, finetune_df


def plot_results(dfs, save_fn=None, task='category', n_blocks=2, n_steps=1000,
                 downsample_step=1, all_ctx_conds=False):
    df, df_avg, df_melted, few_shot_df, finetune_df = dfs
    max_loss = df['loss'].max()

    # Figure parameters
    fig_h = 5 # height
    fig_w = fig_h * 18/5 # width
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12

    # Set up figure
    if all_ctx_conds:
        width = 9/8 * fig_w
        fig = plt.figure(figsize=(width, fig_h))
        gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 3, 3])
    else:
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 3, 3])
        
    axes = [] # store axes for later

    # Colors
    pplot_palette = ['tab:blue', 'tab:red']
    lplot_palette = ['tab:purple', 'tab:orange', 'tab:green']
    if task == 'category':
        lplot_palette = lplot_palette[:2]

    # Point plot for few-shot accuracy
    if not all_ctx_conds:
        pplot1 = fig.add_subplot(gs[:, 0]) # bar plot in first column, spanning both rows
        sns.pointplot(data=few_shot_df, x='Curriculum', 
                      y='Accuracy', hue='Rotation', 
                    hue_order=['Rule-like', 'Rotated'],
                    palette=pplot_palette,
                    capsize=0.2, errwidth=1.5, errcolor='black', markers='o',
                    jitter=True, dodge=0.2,
                    ax=pplot1)
        pplot1.set_title('Few-shot\n(in context)', fontsize=title_fontsize)
        pplot1.set_ylabel('Test accuracy', fontsize=label_fontsize)
        pplot1.set_xlabel('')
        pplot1.set_ylim([-0.05, 1.05])
        pplot1.tick_params(axis='x', labelsize=tick_fontsize)
        pplot1.tick_params(axis='y', labelsize=tick_fontsize)
        # pplot1.legend(title=None, loc='best', fontsize=label_fontsize)
        pplot1.get_legend().remove()
        gs_i = 1
    else:
        pplot1 = fig.add_subplot(gs[:, 0]) # bar plot in first column, spanning both rows
        # get only Aligned and Misaligned Curriculum conditions
        few_shot1 = few_shot_df[few_shot_df['Curriculum'].isin(['Aligned', 'Misaligned'])]
        sns.pointplot(data=few_shot1, x='Curriculum', 
                      y='Accuracy', hue='Rotation', 
                      hue_order=['Rule-like', 'Rotated'],
                      palette=pplot_palette,
                      capsize=0.2, errwidth=1.5, errcolor='black', markers='o',
                      jitter=True, dodge=0.2,
                      ax=pplot1)
        pplot1.set_title('Few-shot\n(in context)')
        pplot1.set_ylabel('Test accuracy')
        pplot1.set_xlabel('')
        pplot1.set_ylim([-0.05, 1.05])        
        pplot1.tick_params(axis='x', labelsize=tick_fontsize)
        pplot1.tick_params(axis='y', labelsize=tick_fontsize)
        pplot1.get_legend().remove()

        pplot2 = fig.add_subplot(gs[:, 1]) # bar plot in first column, spanning both rows
        few_shot2 = few_shot_df[few_shot_df['Curriculum'].isin(['Blocked', 'Interleaved'])]
        sns.pointplot(data=few_shot2, x='Curriculum', 
                      y='Accuracy', hue='Rotation', 
                      hue_order=['Rule-like', 'Rotated'],
                      palette=pplot_palette,
                      capsize=0.2, errwidth=1.5, errcolor='black', markers='o',
                      jitter=True, dodge=0.2,
                      ax=pplot2)
        pplot2.set_title('Few-shot\n(in context)', fontsize=title_fontsize)
        pplot2.set_ylabel('Test accuracy', fontsize=label_fontsize)
        pplot2.set_xlabel('')
        pplot2.set_ylim([-0.05, 1.05])
        pplot2.tick_params(axis='x', labelsize=tick_fontsize)
        pplot2.tick_params(axis='y', labelsize=tick_fontsize)
        pplot2.get_legend().remove()
        gs_i = 2

    # Point plot for fintening accuracy
    pplot3 = fig.add_subplot(gs[:, gs_i]) # bar plot in first column, spanning both rows
    sns.pointplot(data=finetune_df, x='Curriculum', y='Accuracy', hue='Rotation', 
                hue_order=['Rule-like', 'Rotated'],
                palette=pplot_palette,
                capsize=0.2, errwidth=1.5,
                jitter=True, dodge=0.2,
                ax=pplot3)
    pplot3.set_title('After training\n(in weights)', fontsize=title_fontsize)
    pplot3.set_ylabel('')
    pplot3.set_xlabel('')
    pplot3.set_ylim([-0.05, 1.05])
    pplot3.set_yticks([]) # remove yticks
    pplot3.tick_params(axis='x', labelsize=tick_fontsize)
    pplot3.legend(title=None, loc='best', fontsize=label_fontsize)

    # 2x2 grid of line plots for finetuning loss and accuracies
    rot_conds = ['Rule-like', 'Rotated']
    cur_conds = ['Blocked', 'Interleaved']
    for i, (cur, rot) in enumerate(product(cur_conds, rot_conds)):
        # Set up axes
        ax1 = fig.add_subplot(gs[i//2, (i%2) + 1 + gs_i])
        ax2 = ax1.twinx()

        # Filter data
        df_loss = df_avg[(df_avg['Rotation'] == rot) & 
                            (df_avg['Curriculum'] == cur)]
        df_acc = df_melted[(df_melted['Rotation'] == rot) &
                            (df_melted['Curriculum'] == cur)]

        # Downsample data
        df_loss = df_loss.iloc[::downsample_step]
        df_acc = df_acc.iloc[::downsample_step]

        # Plot finetuning loss and accuracy
        if task == 'category':
            sns.lineplot(data=df_acc, x='Step', y='Accuracy', 
                         hue='Group', style='Split',
                         palette=lplot_palette, ax=ax1)
        elif task == 'grid':
            sns.lineplot(data=df_acc, x='Step', y='Accuracy', 
                         hue='Split',
                         palette=lplot_palette, ax=ax1)
        sns.lineplot(data=df_loss, x='Step', y='loss', label='Loss', 
                    color='dimgrey',
                    ax=ax2) 
        axes.append(ax1)

        # Vertical lines indicating block transitions
        # NOTE: not correct when using downsampling
        if i < 2:
            vlines = [j * n_steps for j in range(1, n_blocks)]
            for x in vlines:
                ax1.axvline(x=x, color='black', linestyle='--', linewidth=1)

        # X-axes
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        if i < 2:
            ax1.set_title(f'{rot.capitalize()}', fontsize=title_fontsize)
        if i >= 2:
            ax1.set_xlabel('Step', fontsize=label_fontsize)
        
        # Y-axes
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_ylim([0, max_loss])
        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.tick_params(axis='y', labelsize=tick_fontsize)
        if i % 2 == 0:
            ax1.set_ylabel(f'{cur.capitalize()}', fontsize=label_fontsize)
        if i % 2 == 1:
            ax1.set_ylabel('Accuracy', fontsize=label_fontsize)
            ax2.set_ylabel('Loss', fontsize=label_fontsize)
            
        # Save legend 
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.get_legend().remove()
        ax2.get_legend().remove()

    # Add legend back into bottom left plot
    fig.legend(lines+lines2, labels+labels2, 
               loc='center left', bbox_to_anchor=(0.99, 0.5), 
               fontsize=label_fontsize)

    if save_fn is not None:
        plt.savefig(save_fn, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_ctx_step(dfs, task='grid', n_blocks=4, n_steps=1000, downsample_step=1):

    df, df_avg, df_melted, few_shot_df, finetune_df = dfs
    max_loss = df['loss'].max()

    # Figure parameters
    fig_h = 10 # height
    fig_w =  3*fig_h/2  # width
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12

    # Set up figure
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1.5, 1.5])
        
    axes = [] # store axes for later

    # Colors
    pplot_palette = ['tab:blue', 'tab:red']
    lplot_palette = ['tab:purple', 'tab:orange', 'tab:green']
    if task == 'category':
        lplot_palette = lplot_palette[:2]

    # # Point plot for fintening accuracy
    ave_plot = fig.add_subplot(gs[:, 0])  # First column for the line plot

    # Plot the line without the shaded area for the error
    sns.lineplot(data=finetune_df, 
                x='Curriculum (step)', y='Accuracy', 
                hue='Rotation', style='Curriculum (context)', 
                markers=True, dashes=True, 
                ci=95, err_style='bars',
                hue_order=['Rule-like', 'Rotated'], 
                style_order=['Blocked', 'Interleaved'], 
                palette=pplot_palette,
                ax=ave_plot)

    ave_plot.set_title('After training\n(in weights)', fontsize=title_fontsize)
    ave_plot.set_ylabel('Accuracy', fontsize=label_fontsize)
    ave_plot.set_xlabel('Curriculum (step)', fontsize=label_fontsize)
    ave_plot.set_ylim([-0.05, 1.05])
    ave_plot.tick_params(axis='x', labelsize=tick_fontsize)
    # ave_plot.tick_params(axis='y', labelsize=tick_fontsize)
    ave_plot.legend(title=None, 
                    loc='lower center', 
                    # bbox_to_anchor=(0.5, 0.5),
                    fontsize=label_fontsize)



    # 2x2 grid of line plots for finetuning loss and accuracies
    rot_conds = ['Rule-like', 'Rotated']
    ctx_conds = ['Blocked', 'Interleaved']
    stp_conds = ['Blocked', 'Interleaved']
    cur_conds = [(ctx, stp) for ctx in ctx_conds for stp in stp_conds]
    for (cur_i, (ctx, stp)), (rot_i, rot) in product(enumerate(cur_conds), 
                                                    enumerate(rot_conds)):
        # Set up axes
        row_i = cur_i
        col_i = rot_i + 1
        ax1 = fig.add_subplot(gs[row_i, col_i])
        ax2 = ax1.twinx()

        # Filter data
        df_loss = df_avg[(df_avg['Rotation'] == rot) & 
                         (df_avg['Curriculum (context)'] == ctx) &
                         (df_avg['Curriculum (step)'] == stp)]
        df_acc = df_melted[(df_melted['Rotation'] == rot) &
                           (df_melted['Curriculum (context)'] == ctx) &
                           (df_melted['Curriculum (step)'] == stp)]
        
        # Downsample data
        df_loss = df_loss.iloc[::downsample_step]
        df_acc = df_acc.iloc[::downsample_step]

        # Plot finetuning loss and accuracy
        if task == 'category':
            sns.lineplot(data=df_acc, x='Step', y='Accuracy', 
                            hue='Group', style='Split',
                            palette=lplot_palette, ax=ax1)
        elif task == 'grid':
            sns.lineplot(data=df_acc, x='Step', y='Accuracy', 
                            hue='Split',
                            palette=lplot_palette, ax=ax1)
        sns.lineplot(data=df_loss, x='Step', y='loss', label='Loss', 
                    color='dimgrey',
                    ax=ax2) 
        axes.append(ax1)

        # Vertical lines indicating block transitions
        if stp == 'Blocked':
            vlines = [j * n_steps for j in range(1, n_blocks)]
            for x in vlines:
                ax1.axvline(x=x, color='black', linestyle='--', linewidth=1)

        # X-axes
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        if row_i == 0:
            ax1.set_title(f'{rot.capitalize()}', fontsize=title_fontsize)
        if row_i == len(cur_conds) - 1:
            ax1.set_xlabel('Step', fontsize=label_fontsize)
        
        # Y-axes
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_ylim([0, max_loss])
        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.tick_params(axis='x', labelsize=tick_fontsize)
        ax1.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.tick_params(axis='y', labelsize=tick_fontsize)
        if rot_i == 0:
            ax1.set_ylabel(f'{ctx.capitalize()},{stp.capitalize()}', 
                        fontsize=label_fontsize)
        if rot_i == 1:
            ax1.set_ylabel('Accuracy', fontsize=label_fontsize)
            ax2.set_ylabel('Loss', fontsize=label_fontsize)
            
        # Save legend 
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.get_legend().remove()
        ax2.get_legend().remove()

    # Add legend back into bottom left plot
    fig.legend(lines+lines2, labels+labels2, 
                loc='center left', bbox_to_anchor=(0.99, 0.5), 
                fontsize=label_fontsize)

    # Add a meta-label for the y-axis, positioned between first and second columns
    fig.text(0.25, 0.5, 
            'Curriculum: context, steps',
            va='center', ha='center', rotation='vertical',
            fontsize=label_fontsize)


    plt.tight_layout()

    plt.show()

######################## Functions for plotting tradeoff #######################
# Figure dimensions and appearance
FIG_WIDTH = 24
FIG_HEIGHT = 4
DPI = 100
FONT_SCALE = 2
MAIN_FONT_SIZE = 16
TITLE_FONT_SIZE = 18
TASK_TITLE_FONT_SIZE = 20
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 12
COLORBAR_LABEL_SIZE = 14
COLORBAR_WIDTH = 0.05  # Width of colorbars relative to plots
LINEWIDTH = 2
MARKERSIZE = 5
GRID_ALPHA = 0.3
BAR_WIDTH = 0.7  # Width for bar plots

# Individual spacing between plots (as percentages of the plot width)
WSPACE_GEN_LEARN = 0.25  # Between generalization and learning
WSPACE_LEARN_CBAR = 0.02 # Between learning and its colorbar
WSPACE_CBAR_LOSS = 0.45  # Between learning colorbar and loss
WSPACE_LOSS_CBAR = 0.02  # Between loss and its colorbar
WSPACE_CBAR_RET = 0.5   # Between loss colorbar and retention

# Color settings
COLORMAP_TRUNCATE = 0.9  # Truncate colormap (0-1, lower values show less yellow)
color_vals = plt.cm.viridis(np.linspace(0.0, COLORMAP_TRUNCATE, 256))
COLORMAP = mcolors.ListedColormap(color_vals)
MARKER_EDGE_COLOR = 'white'
MARKER_EDGE_WIDTH = 0.5
GRID_COLOR = 'gray'
GRID_STYLE = '--'

# Spacing and layout
WSPACE = 0.3  # Width space between elements in the same task group
HSPACE = 0.3  # Height space between elements
TOP_MARGIN = 0.85  # Top margin for task titles

def get_tradeoff_dfs(results, rot_cond, 
                     test_every=200, steps_per_block=2000, first_block=False):
    # Get results from iwl testing
    gen_results = [] # test acc for various ctxp during initial test
    ret_results = [] # train acc with ctxp=1.0 at each step of finetuning
    loss_results = [] # loss at each step of finetuning

    for ab_level_train in results.keys():
        ab_level_train_str = ab_level_train.replace('p', '.')

        # Get the first block name
        if first_block:
            key = (rot_cond, 'blocked', 'blocked')
            train_name = results[ab_level_train][key]['block_name_data'][0][0]
        else:
            key = (rot_cond, 'interleaved', 'interleaved')
            train_name = 'train'

        # Get generalization results
        initial_test = results[(ab_level_train)][key]['ab_initial_test']
        for episode in initial_test:
            # Get episode number
            episode_i = None
            any_key = next(iter(episode))
            for k in episode[any_key].keys():
                if 'episode' in k:
                    split_k = k.split('_')
                    for i, s in enumerate(split_k):
                        if 'episode' in s:
                            episode_i = split_k[i+1]
                            break
            # Get generalization accuracy
            for ab_level_test in episode.keys():
                for acc_type in episode[ab_level_test].keys():
                    if 'test' in acc_type:
                        acc = episode[ab_level_test][acc_type][0]
                        row = {'ab_level_train': float(ab_level_train_str),
                               'ab_level_test': round(ab_level_test, 1), 
                               'episode': episode_i,
                               'Test Type': 'Generalization',
                               'Accuracy': acc}
                gen_results.append(row)

        # Get retention results
        acc_data = results[ab_level_train][key]['ab_acc_data']
        for episode in acc_data:
            # Get episode number
            episode_i = None
            any_key = next(iter(episode[0]))
            for k in episode[0][any_key].keys():
                if 'episode' in k:
                    split_k = k.split('_')
                    for i, s in enumerate(split_k):
                        if 'episode' in s:
                            episode_i = split_k[i+1]
                            break
            # Retention results
            for block_i, block in enumerate(episode):
                for ab_level_test in block.keys():
                    train_accs = []
                    test_accs = []
                    for acc_type in block[ab_level_test].keys():
                        if train_name in acc_type:
                            acc = block[ab_level_test][acc_type]
                            train_accs.append(acc)
                        elif 'test' in acc_type:
                            acc = block[ab_level_test][acc_type]
                            test_accs.append(acc)
                    train_accs = np.mean(train_accs, axis=0)
                    test_accs = np.mean(test_accs, axis=0)
                    for step_i, acc in enumerate(train_accs):
                        prev_blocks_steps = steps_per_block * block_i
                        curr_block_steps = step_i * test_every
                        true_step_i = prev_blocks_steps + curr_block_steps
                        row = {'ab_level_train': float(ab_level_train_str),
                               'ab_level_test': ab_level_test, 
                               'episode': episode_i,
                               'block': block_i,
                               'step': true_step_i,
                               'Test Type': 'Retention',
                               'Trial Type': 'Studied',
                               'Accuracy': acc}
                        ret_results.append(row)
                    for step_i, acc in enumerate(test_accs):
                        prev_blocks_steps = steps_per_block * block_i
                        curr_block_steps = step_i * test_every
                        true_step_i = prev_blocks_steps + curr_block_steps
                        row = {'ab_level_train': float(ab_level_train_str),
                               'ab_level_test': ab_level_test, 
                               'episode': episode_i,
                               'block': block_i,
                               'step': true_step_i,
                               'Test Type': 'Retention',
                               'Trial Type': 'Novel',
                               'Accuracy': acc}
                        ret_results.append(row)
        # Get loss results
        loss_data = results[ab_level_train][key]['loss_data']
        for episode_i, episode in enumerate(loss_data):
            for block_i, block in enumerate(episode):
                for step_i, loss in enumerate(block):
                    prev_blocks_steps = steps_per_block * block_i
                    true_step_i = prev_blocks_steps + step_i
                    row = {'ab_level_train': float(ab_level_train_str), 
                           'episode': episode_i,
                           'block': block_i,
                           'step': true_step_i,
                           'loss': loss}
                    loss_results.append(row)

    # Convert to dataframes
    gen_df = pd.DataFrame(gen_results)
    ret_df = pd.DataFrame(ret_results)
    loss_df = pd.DataFrame(loss_results)

    # Average
    group_by = ['ab_level_train', 'ab_level_test', 'episode', 'Test Type']
    gen_df = gen_df.groupby(group_by).mean().reset_index()

    return gen_df, ret_df, loss_df

def plot_generalization_bars(ax, gen_df, ab_level_train_max, ylim, 
                             title=None, ab_label=None):
    """
    Plot generalization performance as bar plots with attention dropout on the x-axis.
    Only shows data for the maximum number of in-context examples (final vertical slice).
    """
    # Check if ab_level_train == 0.0 exists
    if 0.0 not in gen_df['ab_level_train'].unique():
        print('No data for ab_level_train == 0.0. Using the minimum value instead.')
        ab_level_train_plot = min(gen_df['ab_level_train'].unique())
    else:
        ab_level_train_plot = 0.0

    # Filter the dataframes
    gen_filtered = gen_df[gen_df['ab_level_train'] == ab_level_train_plot]
    gen_filtered = gen_filtered[gen_filtered['ab_level_test'] <= ab_level_train_max]
    
    # Get unique ab_level_test values and sort them
    ab_level_test_values = sorted(gen_filtered['ab_level_test'].unique())
    norm = plt.Normalize(0, ab_level_train_max)
    
    # Group data by ab_level_test to get mean Accuracy
    gen_grouped = gen_filtered.groupby(['ab_level_test'])['Accuracy'].mean().reset_index()
    
    # Create x-positions for bars
    x_pos = np.arange(len(ab_level_test_values))
    
    # Plot bars with color based on ab_level_test
    bars = ax.bar(x_pos, gen_grouped['Accuracy'], width=BAR_WIDTH)
    
    # Color each bar based on ab_level_test
    for i, bar in enumerate(bars):
        ab_level = ab_level_test_values[i]
        bar.set_color(COLORMAP(norm(ab_level)))
        bar.set_edgecolor(MARKER_EDGE_COLOR)
        bar.set_linewidth(MARKER_EDGE_WIDTH)
    
    # Customize plot appearance
    if ab_label is None:
        ax.set_xlabel('$p_a$', fontsize=AXIS_LABEL_SIZE)
    else:
        ax.set_xlabel(ab_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Accuracy', fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.set_ylim(ylim)
    
    # Set x-ticks to show ab_level_test values
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{p:.1f}" for p in ab_level_test_values], 
                       fontsize=TICK_LABEL_SIZE)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return norm, ab_level_test_values

# Function for plotting retention
def plot_learning(ax, ret_df, ab_level_train_max, xlim, ylim, title=None):
    # Filter the dataframe
    ret_filtered = ret_df[ret_df['ab_level_train'] <= ab_level_train_max]
    ret_filtered = ret_filtered[ret_filtered['ab_level_test'] == ret_filtered['ab_level_train']]
    ret_filtered = ret_filtered[ret_filtered['Trial Type'] == 'Studied']
    
    # Get unique ab_level_train values and sort them
    ab_level_train_values = sorted(ret_filtered['ab_level_train'].unique())
    min_ab_level_train = min(ab_level_train_values)
    max_ab_level_train = max(ab_level_train_values)
    norm = plt.Normalize(min_ab_level_train, max_ab_level_train)
    
    # Group data by step and ab_level_train to get mean Accuracy
    ret_grouped = ret_filtered.groupby(['step', 'ab_level_train'])['Accuracy'].mean().reset_index()
    
    # Plot each line with color based on ab_level_train
    for ab_level in ab_level_train_values:
        subset = ret_grouped[ret_grouped['ab_level_train'] == ab_level]
        subset = subset.sort_values('step')  # Sort for proper line connection
        
        color = COLORMAP(norm(ab_level))
        
        ax.plot(subset['step'], subset['Accuracy'], '-o', color=color,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, 
                markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH)
    
    # Customize plot appearance
    ax.set_xlabel('Training Step', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Accuracy', fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.set_xlim(xlim[0], xlim[1]) # only show first block
    ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format x-axis with scientific notation if numbers are large
    if ret_filtered['step'].max() > 10000:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    return norm, ab_level_train_values

def plot_loss(ax, loss_df, ab_level_train_max, loss_xlim, loss_ylim, 
              smooth_window=10, title='Loss'):
    # Filter the dataframe
    loss_filtered = loss_df[loss_df['ab_level_train'] <= ab_level_train_max]
    loss_filtered = loss_filtered[loss_filtered['step'] >= loss_xlim[0]]
    loss_filtered = loss_filtered[loss_filtered['step'] < loss_xlim[1]]
    
    # Get unique ab_level_train values and sort them
    ab_level_train_values = sorted(loss_filtered['ab_level_train'].unique())
    min_ab_level_train = min(ab_level_train_values)
    max_ab_level_train = max(ab_level_train_values)
    norm = plt.Normalize(min_ab_level_train, max_ab_level_train)
    
    # Group data by step and ab_level_train to get mean Accuracy
    loss_grouped = loss_filtered.groupby(['step', 'ab_level_train'])['loss'].mean().reset_index()

    # Smooth the loss values over time
    loss_grouped['loss'] = loss_grouped.groupby('ab_level_train')['loss'].transform(
        lambda x: x.rolling(window=smooth_window, min_periods=1).mean())
    
    # Plot each line with color based on ab_level_train
    for ab_level in ab_level_train_values:
        subset = loss_grouped[loss_grouped['ab_level_train'] == ab_level]
        subset = subset.sort_values('step')  # Sort for proper line connection
        
        color = COLORMAP(norm(ab_level))
        
        ax.plot(subset['step'], subset['loss'], color=color,
                linewidth=LINEWIDTH)
    
    # Customize plot appearance
    ax.set_xlim(loss_xlim[0], loss_xlim[1]) # only show first block
    ax.set_ylim(loss_ylim)
    ax.set_xlabel('Training Step', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Loss', fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.set_xlim(loss_xlim) # only show first block
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format x-axis with scientific notation if numbers are large
    if loss_filtered['step'].max() > 10000:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    return norm, ab_level_train_values

def plot_retention_bars(ax, ret_df, ab_level_train_max, last_step, ylim, 
                        title=None, ab_label=None):
    # Filter the dataframe
    ret_filtered = ret_df[ret_df['ab_level_test'] == ab_level_train_max]
    ret_filtered = ret_filtered[ret_filtered['ab_level_train'] <= ab_level_train_max]
    ret_filtered = ret_filtered[ret_filtered['Trial Type'] == 'Studied']
    ret_filtered = ret_filtered[ret_filtered['step'] == last_step]
    
    # Get unique ab_level_train values and sort them
    ab_level_train_values = sorted(ret_filtered['ab_level_train'].unique())
    min_ab_level_train = min(ab_level_train_values)
    max_ab_level_train = max(ab_level_train_values)
    norm = plt.Normalize(min_ab_level_train, max_ab_level_train)

    # Group data by ab_level_test to get mean Accuracy
    ret_grouped = ret_filtered.groupby(['ab_level_train'])['Accuracy'].mean().reset_index()
    
    # Create x-positions for bars
    x_pos = np.arange(len(ab_level_train_values))
    
    # Plot bars with color based on ab_level_test
    bars = ax.bar(x_pos, ret_grouped['Accuracy'], width=BAR_WIDTH)
    
    # Color each bar based on ab_level_test
    for i, bar in enumerate(bars):
        ab_level = ab_level_train_values[i]
        bar.set_color(COLORMAP(norm(ab_level)))
        bar.set_edgecolor(MARKER_EDGE_COLOR)
        bar.set_linewidth(MARKER_EDGE_WIDTH)
    
    # Customize plot appearance
    if ab_label is None:
        ax.set_xlabel('$p_a$', fontsize=AXIS_LABEL_SIZE)
    else:
        ax.set_xlabel(ab_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Accuracy', fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.set_ylim(ylim)
    
    # Set x-ticks to show ab_level_test values
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{p:.1f}" for p in ab_level_train_values], 
                       fontsize=TICK_LABEL_SIZE)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return norm, ab_level_train_values

def plot_tradeoff(gen_df, ret_df, loss_df, ab_level_train_max,
                  last_step, xlim, ylim, loss_ylim, smooth_window, ab_label):
    # ===================== PLOT CONFIGURATION =====================
    # Set the style for a publication-ready look
    sns.set_style('whitegrid')
    sns.set_context("paper", font_scale=FONT_SCALE)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = 'black'

    # Create a figure with GridSpec for precise control
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    # Create a gridspec with 11 columns - content and spacing alternating
    gs = fig.add_gridspec(1, 11, 
                        width_ratios=[
                            1,                    # Generalization
                            WSPACE_GEN_LEARN,     # Space
                            1,                    # Learning
                            WSPACE_LEARN_CBAR,    # Space
                            COLORBAR_WIDTH,       # Learning colorbar
                            WSPACE_CBAR_LOSS,     # Space
                            1,                    # Loss
                            WSPACE_LOSS_CBAR,     # Space
                            COLORBAR_WIDTH,       # Loss colorbar
                            WSPACE_CBAR_RET,      # Space
                            1                     # Retention
                        ], 
                        wspace=0)  # No wspace as we're handling it manually

    # Create the axes with proper spacing (skipping the spacing columns)
    ax_gen = fig.add_subplot(gs[0, 0])           # Generalization
    ax_learn = fig.add_subplot(gs[0, 2])         # Learning
    ax_learn_cbar = fig.add_subplot(gs[0, 4])    # Learning Colorbar
    ax_loss = fig.add_subplot(gs[0, 6])          # Loss
    ax_loss_cbar = fig.add_subplot(gs[0, 8])     # Loss Colorbar
    ax_ret = fig.add_subplot(gs[0, 10])          # Retention

    # Plot generalization
    title = 'Few-shot generalization (ICL)'
    gen_norm, gen_values = plot_generalization_bars(ax_gen, gen_df, 
                                                    ab_level_train_max, ylim,
                                                    title=title,
                                                    ab_label=ab_label)

    # Plot learning
    learn_norm, learn_values = plot_learning(ax_learn, ret_df, 
                                             ab_level_train_max, xlim, ylim, 
                                             title='Incremental learning (IWL)')
    
    # Add colorbar for learning
    sm_learn = plt.cm.ScalarMappable(cmap=COLORMAP, norm=learn_norm)
    sm_learn.set_array([])
    cbar_learn = fig.colorbar(sm_learn, cax=ax_learn_cbar)
    if ab_label is None:
        cbar_learn.set_label('$p_a$', fontsize=COLORBAR_LABEL_SIZE)
    else:
        cbar_learn.set_label(ab_label, fontsize=COLORBAR_LABEL_SIZE)
    

    # Plot loss
    loss_norm, loss_values = plot_loss(ax_loss, loss_df,
                                       ab_level_train_max,
                                       loss_xlim=xlim,
                                       loss_ylim=loss_ylim,
                                       smooth_window=smooth_window,
                                       title='Errors')

    # Add colorbar for loss
    sm_loss = plt.cm.ScalarMappable(cmap=COLORMAP, norm=loss_norm)
    sm_loss.set_array([])
    cbar_loss = fig.colorbar(sm_loss, cax=ax_loss_cbar)
    if ab_label is None:
        cbar_loss.set_label('$p_a$', fontsize=COLORBAR_LABEL_SIZE)
    else:
        cbar_loss.set_label(ab_label, fontsize=COLORBAR_LABEL_SIZE)

    # Plot retention
    ret_norm, ret_values = plot_retention_bars(ax_ret, ret_df, 
                                               ab_level_train_max,
                                               last_step, ylim, 
                                               title='Retention',
                                               ab_label=ab_label)

    plt.show()
