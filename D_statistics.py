import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn as sns
import pandas as pd
import numpy as np


def plot_box_MI_ME(data, columns, freq, path=None):
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.boxplot(data)

    # Customize the plot
    #plt.title('Box and Whisker Plot')
    plt.xlabel('Task')
    plt.ylabel('ERD/S (%)')
    plt.xticks([1, 2, 3, 4], columns)

    if path:
        plt.savefig('{}/box_whisker_ME_MI_{}.svg'.format(path, freq), format='svg')
        plt.close()
    else:
        plt.show()


def plot_EMM(data_array, columns, freq, path=None):

    # Convert ndarray to DataFrame with correct mapping
    data_df = pd.DataFrame(data_array, columns=columns)

    # Melt DataFrame to long format for easier plotting
    df_long = pd.melt(data_df.reset_index(), id_vars='index', value_vars=columns,
                      var_name='Condition', value_name='Value')

    # Extract 'Setting', 'Task', and 'ROI' from 'Condition' column
    df_long['Setting'] = df_long['Condition'].apply(lambda x: '2D' if '2D' in x else '3D')
    df_long['Task'] = df_long['Condition'].apply(lambda x: 'MI left' if 'MI left' in x else 'MI right')
    df_long['ROI'] = df_long['Condition'].apply(lambda x: x.split()[-1])
    df_long.rename(columns={'index': 'Sample'}, inplace=True)

    # Plotting
    fig, axes = plt.subplots(1, 6, figsize=(12, 6), gridspec_kw={'height_ratios': [1], 'hspace': 0.25}, sharey=True)

    # Unique markers and colors for the two tasks
    markers = {'MI left': 'o', 'MI right': 's'}
    colors = {'MI left': 'green', 'MI right': 'orange'}  # Task MI left is green, MI right is orange
    x_offsets = {'MI left': -0.15, 'MI right': 0.15}  # Offset for the tasks

    # Iterate over ROIs
    for i, roi in enumerate(df_long['ROI'].unique()):
        ax = axes[i]
        subset = df_long[df_long['ROI'] == roi]

        # Offset data points for each task
        for setting in subset['Setting'].unique():
            for task in subset['Task'].unique():
                subset_task = subset[(subset['Setting'] == setting) & (subset['Task'] == task)]
                x_pos = subset_task['Setting'].apply(lambda x: x_offsets[task] + list(subset['Setting'].unique()).index(x))

                sns.scatterplot(
                    x=x_pos, y=subset_task['Value'], color=colors[task], marker=markers[task], alpha=0.2, ax=ax, legend=False
                )

        # Calculate means and confidence intervals
        emms = subset.groupby(['Setting', 'Task'])['Value'].agg(['mean', 'sem']).reset_index()
        emms['lower_ci'] = emms['mean'] - 1.96 * emms['sem']
        emms['upper_ci'] = emms['mean'] + 1.96 * emms['sem']

        # Plot means with lines and error bars for each task
        for task in emms['Task'].unique():
            task_means = emms[emms['Task'] == task]
            x_positions = [x_offsets[task] + i for i in range(len(task_means['Setting'].unique()))]

            ax.errorbar(
                x=x_positions,
                y=task_means['mean'].values,
                yerr=[task_means['mean'].values - task_means['lower_ci'].values,
                      task_means['upper_ci'].values - task_means['mean'].values],
                fmt=markers[task], capsize=5, capthick=1, color=colors[task], label=task
            )

            # Plot line connecting the means for each task across settings
            ax.plot(x_positions, task_means['mean'].values, color=colors[task], linestyle='-', linewidth=1)

        ax.set_title(roi)
        ax.set_xlabel('')

        # Set x-axis labels to "Setting 1" and "Setting 2"
        ax.set_xticks(range(len(subset['Setting'].unique())))
        ax.set_xticklabels(subset['Setting'].unique())

    axes[0].set_ylabel('ERD/S (%)')

    # Remove individual legends and add a single legend outside of all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc='center right', title='Task')  # Only use first 2 items for legend

    #fig.suptitle('Estimated Marginal Means for Different ROIs', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the plot area to make space for the legend

    if path:
        plt.savefig('{}/EEM_plot_{}.svg'.format(path, freq), format='svg')
        plt.close()
    else:
        plt.show()

def plot_participant_survey(monitor, vr, path=None):
    categories = ['Preferred training method', 'Better according to subjective perception',
                  'Better according to avg. accuracy']

    bar_width = 0.35
    y_pos = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(y_pos, monitor, height=bar_width, label='2D', color='#f47942')
    ax.barh(y_pos + bar_width, vr, height=bar_width, label='3D', color='#00587a')

    # Labels and Title
    ax.set_yticks(y_pos + bar_width / 2)
    ax.set_yticklabels(categories, fontsize=14)
    ax.legend()
    ax.set_xlim(0, 17)
    ax.set_xticks(np.arange(1, 18, 1))  # Set x-ticks to show all numbers from 1 to 17
    ax.set_xlabel('Number of subjects', fontsize=14)
    ax.set_title('Participant Survey', fontsize=20)

    plt.tight_layout()

    if path:
        plt.savefig('{}/participant_survey.svg'.format(path), format='svg')
        plt.close()
    else:
        plt.show()