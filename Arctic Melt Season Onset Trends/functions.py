import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import csv
import seaborn as sns
import re
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
# from mpl_toolkits.axes_grid1 import host_subplot
# import mpl_toolkits.axisartist as AA
from mpl_toolkits.axisartist import SubplotHost
import matplotlib.gridspec as gridspec


def availabilities(path):

    '''Calculate data availabilities for each month and year from the dataset.'''

    # Read file
    df = pd.read_csv(path, sep='\t', header=46, index_col=0, parse_dates=True)
    
    # Generate full hourly index
    full_index = pd.date_range(start=f'{df.index.min().year}-01-01',
                               end=f'{df.index.max().year + 1}-01-01', freq='h')

    # Build a DataFrame with all expected timestamps
    expected_df = pd.DataFrame(index=full_index)
    expected_df['year'] = expected_df.index.year
    expected_df['month'] = expected_df.index.month

    # Count how many hours are expected in each (year, month)
    expected_counts = expected_df.groupby(['year', 'month']).size()
    
    availabilities = []
    
    for col in df.columns[0:1]:  # Only first numeric column

        # Skip non-numeric columns
        if df[col].dtype.kind not in 'fi':
            continue
    
        temp_df = df[[col]].copy()
        temp_df['year'] = temp_df.index.year
        temp_df['month'] = temp_df.index.month


        # Count how many non-NaN entries are available per (year, month)
        valid_counts = temp_df[col].notna().groupby([temp_df['year'], temp_df['month']]).sum()

        # Align valid counts with expected index and fill missing months with 0
        aligned_valid = valid_counts.reindex(expected_counts.index, fill_value=0)

        # Compute availability ratio (actual / expected)
        availability = (aligned_valid / expected_counts).unstack(level=0)
        availabilities.append(availability)
    
    avg_availability = sum(availabilities) / len(availabilities)
    
    # Remove years with only NaN or 0
    year_mask = (avg_availability.fillna(0) != 0).any(axis=0)
    avg_availability = avg_availability.loc[:, year_mask]

    # Calculate average availability per year
    yearly_avgs = avg_availability.mean(axis=0).dropna()
    return avg_availability, yearly_avgs


def availabilities_plots(paths):

    '''Plot data availabilities for multiple stations.'''

    # Initialize subplots
    fig, axes = plt.subplots(nrows=len(paths), ncols=1, figsize=(16, 14))
    axes = axes.flatten()
    
    for i, (station, path) in enumerate(paths.items()):
        try:
            
            avg_availability, yearly_avgs = availabilities(path)
            
            legend_labels = [f"{year} - {int(100 * yearly_avgs[year]):d} %" for year in yearly_avgs.index]
    
            # Define a color palette
            palette = sns.color_palette("mako", n_colors=len(yearly_avgs))
    
            # Plot
            avg_availability.plot(
                kind='bar',
                ax=axes[i],
                width=0.8,
                color=palette
            )
    
            axes[i].set_title(f'{station} Station', fontweight='bold')
            axes[i].set_ylabel('Fraction of Data Available')
            axes[i].set_xlabel('')
            axes[i].set_xticks(ticks=range(12))
            axes[i].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
            axes[i].legend(
                        title='Year',
                        labels=legend_labels,
                        loc='center left',
                        bbox_to_anchor=(1.01, 0.5),
                        borderaxespad=0.
                )
            
            axes[i].grid(True, axis='y')
            axes[i].set_ylim(0, 1)
    
        except Exception as e:
            print(f"Error processing station {station}: {e}")
    
    # Remove unused subplots if any
    if len(paths) < len(axes):
        for j in range(len(paths), len(axes)):
            fig.delaxes(axes[j])
    
    plt.tight_layout(pad=3.0)
    
    plt.savefig('data_availability.png', dpi=150)
    plt.show()
    return 







def preprocess_monthly_distributions(paths):
    
    '''Preprocess data to compute monthly median and interquartile ranges for size distributions.'''

    monthly_data_dict = {}
    diameters = None

    for station, path in paths.items():
        try:
            df = pd.read_csv(path, sep='\t', header=46, index_col=0, parse_dates=True)
            df['month'] = df.index.month

            size_bin_cols = [col for col in df.columns if 'dN/dlogDp' in col]

            # Extract diameters once
            if diameters is None:
                diameters = np.array([float(re.search(r'\((.*?)\)', col).group(1)) for col in size_bin_cols])

            station_data = {}
            for month in range(1, 13):
                monthly = df[df['month'] == month][size_bin_cols]
                station_data[month] = {
                    'median': monthly.median(),
                    'q25': monthly.quantile(0.25),
                    'q75': monthly.quantile(0.75)
                }

            monthly_data_dict[station] = station_data

        except Exception as e:
            print(f"Error processing {station}: {e}")

    return diameters, monthly_data_dict







def plot_monthly_distributions(diameters, monthly_data_dict):

    '''Plot monthly aerosol number size distributions for multiple stations.'''


    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten()
    xticks = [20, 50, 100, 200, 500]

    colors = {
        "Zeppelin": "black",
        "Nord": "blue",
        "Alert": "green",
        "Utqiagvik": "orange",
        "Tiksi": "red"
    }

    for station, station_data in monthly_data_dict.items():
        for month in range(1, 13):
            ax = axes[month - 1]
            try:
                med = station_data[month]['median']
                q25 = station_data[month]['q25']
                q75 = station_data[month]['q75']

                ax.plot(diameters, med, label=station, color=colors[station])
                ax.fill_between(diameters, q25, q75, color=colors[station], alpha=0.2)

            except Exception as e:
                print(f"Plotting error for {station} month {month}: {e}")

    for i, ax in enumerate(axes):
        month = i + 1
        ax.set_title(pd.to_datetime(str(month), format='%m').strftime('%B'), fontweight='bold', fontsize=25)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(20, 500)
        ax.set_ylim(1, None)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=15)

        ax.axvline(x=100, color='k', linestyle='-')
        ax.axhline(y=100, color='k', linestyle='-')

        if i % 3 == 0:
            ax.set_ylabel('dN/dlogDp [cm$^{-3}$]', fontsize=20)
        if i >= 9:
            ax.set_xlabel('Dry diameter [nm]', fontsize=20)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=station)
        for station, color in colors.items()
    ]
    legend = fig.legend(
        handles=legend_elements,
        loc='center',
        bbox_to_anchor=(1., 0.5),
        title='Station',
        title_fontsize=20,
        fontsize=20
    )
    legend.get_title().set_weight('bold')
    for text, handle in zip(legend.get_texts(), legend_elements):
        text.set_color(handle.get_color())

    # Horizontal line under legend title
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    inv = fig.transFigure.inverted()
    x0, y0 = inv.transform(bbox)[0]
    x1, y1 = inv.transform(bbox)[1]
    fig.lines.append(Line2D([x0, x1], [y1 - 0.03] * 2, transform=fig.transFigure, color='black', linewidth=1.5))

    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.93, wspace=0.3, hspace=0.4)
    # plt.suptitle("Monthly Aerosol Number Size Distributions", fontsize=30, fontweight='bold')
    plt.show()

    return 




def station_NSV(path):
    
    '''Calculate monthly median and interquartile ranges for total number, accumulation-mode number, surface area, and volume concentrations.'''

    df = pd.read_csv(path, sep='\t', header=46, index_col=0, parse_dates=True)
    df['month'] = df.index.month
    size_bin_cols = [col for col in df.columns if 'dN/dlogDp' in col]
    diameters = np.array([float(re.search(r'\((.*?)\)', col).group(1)) for col in size_bin_cols])
    data = df[size_bin_cols]
    
    r_cm = (diameters / 2) * 1e-7
    area_weights = 4 * np.pi * r_cm**2
    volume_weights = (4/3) * np.pi * r_cm**3

    df['N10'] = np.trapz(data, np.log10(diameters), axis=1)
    df['Nacc'] = np.trapz(data.loc[:, diameters >= 100], np.log10(diameters[diameters >= 100]), axis=1)
    df['S10'] = np.trapz(data * area_weights, np.log10(diameters), axis=1) * 1e6
    df['V10'] = np.trapz(data * volume_weights, np.log10(diameters), axis=1) * 1e6

    grouped = df.groupby('month')[['N10', 'Nacc', 'S10', 'V10']]
    median = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)
    iqr = q75 - q25

    return median, iqr



def plot_station_subplot(fig, gs, idx, station, median, iqr):

    '''Plot a subplot for a single station showing monthly median and interquartile ranges for number, surface area, and volume concentrations.'''

    months = np.arange(1, 13)

    host = SubplotHost(fig, gs[idx])
    fig.add_subplot(host)

    par1 = host.twinx()
    par2 = host.twinx()

    offset1 = -60
    offset2 = -120

    new_fixed_axis = par1.get_grid_helper().new_fixed_axis
    par1.axis["right"] = new_fixed_axis(loc="left", axes=par1, offset=(offset1, 0))
    par2.axis["right"] = new_fixed_axis(loc="left", axes=par2, offset=(offset2, 0))

    par1.axis["left"].set_visible(False)
    par2.axis["left"].set_visible(False)

    # Axis scales
    host.set_yscale('log')
    par1.set_yscale('log')
    par2.set_yscale('log')

    # Axis labels and colors
    host.set_ylabel("Number concentraion [cm$^{-3}$]", labelpad=35)
    par1.set_ylabel("Surface area [$cm^2  m^{-3}$]", labelpad=45)
    par2.set_ylabel("Total Volume [$cm^3  m^{-3}$]", labelpad=45)

    host.axis["left"].line.set_color("black")
    host.axis["left"].major_ticklabels.set_color("black")
    host.axis["left"].label.set_color("black")

    par1.axis["right"].line.set_color("orange")
    par1.axis["right"].major_ticklabels.set_color("orange")
    par1.axis["right"].label.set_color("orange")

    par2.axis["right"].line.set_color("steelblue")
    par2.axis["right"].major_ticklabels.set_color("steelblue")
    par2.axis["right"].label.set_color("steelblue")

    # X-axis settings
    host.set_xticks(months)
    host.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # host.set_xlabel("Month")

    # Plot data
    p1 = host.errorbar(months, median['N10'], yerr=iqr['N10'], color='gray', linestyle='--', capsize=3, label='N$_{10}$')
    p2 = host.errorbar(months, median['Nacc'], yerr=iqr['Nacc'], color='black', linestyle='-', capsize=3, label='N$_{acc}$')
    p3 = par1.errorbar(months, median['S10'], yerr=iqr['S10'], color='orange', capsize=3, label='S$_{10}$')
    p4 = par2.errorbar(months, median['V10'], yerr=iqr['V10'], color='steelblue', capsize=3, label='V$_{10}$')

    host.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Legend with station name
    station_label = Line2D([0], [0], color='none', label=f"$\\bf{{{station}}}$")

    lines = [station_label, p1, p2, p3, p4]
    labels = [f"$\\bf{{{station}}}$", "N$_{10}$", "N$_{acc}$", "S$_{10}$", "V$_{10}$"]
    
    legend = host.legend(lines, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=15, frameon=True)

    # --- Increase Y-axis label font sizes ---
    host.axis["left"].label.set_fontsize(14)
    par1.axis["right"].label.set_fontsize(14)
    par2.axis["right"].label.set_fontsize(14)

    # Optional: Also increase tick label sizes
    host.tick_params(axis='y', labelsize=12)
    par1.tick_params(axis='y', labelsize=12)
    par2.tick_params(axis='y', labelsize=12)



def total_NSV_plot(paths):
    
    '''Plot monthly median and interquartile ranges for total number, accumulation-mode number, surface area, and volume concentrations for multiple stations.'''
    n_stations = len(paths)
    ncols = 1
    nrows = n_stations
    
    fig = plt.figure(figsize=(15, 20), dpi=150)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.3)
    
    for i, (station, path) in enumerate(paths.items()):
        try:
            median, iqr = station_NSV(path)
            plot_station_subplot(fig, gs, i, station, median, iqr)
        except Exception as e:
            print(f"Error in {station}: {e}")
    
    # plt.tight_layout()
    # plt.suptitle(
    #     "The monthly median and interquartile ranges of the aerosol total and accumulation-mode number concentrations,\n"
    #     "aerosol total surface and the total volume",
    #     fontsize=20,
    #     weight='bold',
    #     y=0.99  # Bring it closer to subplots
    # )
    plt.subplots_adjust(top=0.94)
    # plt.tight_layout()  
    
    plt.show()



def aitken_VS_acc_geom_mean(path):

    '''Calculate geometric mean and standard deviation for Aitken and accumulation-mode particle concentrations.'''


    df = pd.read_csv(path, sep='\t', header=46, index_col=0, parse_dates=True)
    df['month'] = df.index.month
    size_bin_cols = [col for col in df.columns if 'dN/dlogDp' in col]
    diameters = np.array([float(re.search(r'\((.*?)\)', col).group(1)) for col in size_bin_cols])
    data = df[size_bin_cols]

    df['N_aitken'] = np.trapz(data.loc[:, (diameters >= 22) & (diameters<=90)], np.log10(diameters[(diameters >= 22) & (diameters<=90)]), axis=1)
    df['N_acc'] = np.trapz(data.loc[:, (diameters >= 90)], np.log10(diameters[(diameters >= 90)]), axis=1)


    df['date'] = df.index.date
    daily = df.groupby('date')[['N_aitken', 'N_acc']].mean()


    
    log_daily = np.log10(daily).rolling(window=7, center=True)
    geo_mean = 10**log_daily.mean()

    
    
    geo_std_upper = 10**(log_daily.mean() + log_daily.std())
    geo_std_lower = 10**(log_daily.mean() - log_daily.std())
    

    
    # Assign day-of-year and group
    for i in [geo_mean, geo_std_upper, geo_std_lower]:
        i['doy'] = pd.to_datetime(i.index).dayofyear

    mean_doy = geo_mean.groupby('doy').mean().loc[:]
    std_upper_doy = geo_std_upper.groupby('doy').mean().loc[:]
    std_lower_doy = geo_std_lower.groupby('doy').mean().loc[:]


    return mean_doy, std_upper_doy, std_lower_doy, df 




def plot_aitken_VS_acc(paths , scale, tran_period, save):

    '''Plot weekly moving average of Aitken and Accumulation-mode particle concentrations for multiple stations.'''

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(len(paths), hspace=0.1)
    axs = gs.subplots(sharex=True)

    
    for i, (station, path) in enumerate(paths.items()):

        mean_doy, std_upper_doy, std_lower_doy, df = aitken_VS_acc_geom_mean(path)
        
        # Plot
        ax1 = axs[i]
    
        # Plot data
        aitken_line = ax1.plot(mean_doy.index, mean_doy['N_aitken'], label=f'{station}', linestyle='-', color='black')[0]  #Aitken mode (22–90 nm) 
        accum_line  = ax1.plot(mean_doy.index, mean_doy['N_acc'], label=f'{station}', linestyle='-', linewidth=2, color='red')[0]  #Accumulation mode (90–500 nm) 
    
    
        # Shaded quantile bands for Aitken - geometric
        ax1.fill_between(
            mean_doy.index,
            std_lower_doy['N_aitken'],
            std_upper_doy['N_aitken'],
            color='black',
            alpha=0.1,
            #label='Aitken IQR'
        )
    
        # Shaded quantile bands for Accumulation - geometric
        ax1.fill_between(
            mean_doy.index,
            std_lower_doy['N_acc'],
            std_upper_doy['N_acc'],
            color='red',
            alpha=0.1,
            # label='Accumulation IQR'
        )
    
    
     
        # Colored ticks
        ax1.tick_params(axis='y', colors='black')
    
        # Colored axis spines 
        ax1.spines['left'].set_color('black')
    
        # Grid
        ax1.grid(True, linestyle='--', linewidth=0.5)
    
        # log scale
        if scale=='log':
            ax1.set_yscale('log')
            ax1.set_ylim(4, 1500)
        elif scale == 'linear': 
            ax1.set_ylim(0,800)
            
        if tran_period == True:
    
            # focus only between April-June
            ax1.set_xlim(90,180)
            # fig.suptitle('Weekly moving average of Aitken and Accumulation-mode particles\n(April-June)', fontsize=27, y=0.93, weight='bold')
            
        else: # plot the whole year
            ax1.set_xlim()
            # fig.suptitle('Annual weekly moving average of Aitken and Accumulation-mode particles', fontsize=27, y=0.93, weight='bold')
    
    
        # Add legend in each subplot
        bold_font = FontProperties(weight='bold', size=20)
        legend_label = Patch(color='none', label=station + f' ({df.index.year.min()} - {df.index.year.max()})')
        ax1.legend(handles=[legend_label], loc='upper right', frameon=False , prop=bold_font)
    
    
    # Set global y-axis label using fig.text
    fig.text(0.08, 0.5, 'Particle Number Concentration [cm$^{-3}$]', va='center', rotation='vertical', fontsize=27, color='black')
    
    # Set global x-axis label
    fig.text(0.5, 0.07, 'Day of Year', ha='center', fontsize=25)
    
    # Add bottom shared legend
    fig.legend(
        [aitken_line, accum_line],
        ['Aitken mode (22–90 nm)', 'Accumulation mode (90–500 nm)'],
        loc='lower center',
        ncol=2,
        fontsize=23,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005)
    )
    
    
    # plt.tight_layout()
    
    if save is True:
        if tran_period is True:
            plt.savefig(f'weekly_ma_apr_june_{scale}.png', dpi=150)
        else:
            plt.savefig(f'annual_ma_{scale}.png', dpi=150)
    plt.show()

    return





def plot_ratio(paths):

    '''Plot weekly running mean of the ratio of particle concentration between Aitken and accumulation mode for multiple stations.'''

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(len(paths), hspace=0.1)
    axs = gs.subplots(sharex=True)
    # fig.suptitle(r'Weekly running mean of the ratio of particle concentration between Aitken and accumulation mode' + '\n' + '(Apr–July)', fontsize = 20 , y=0.93)
    
    
    
    for i, (station, path) in enumerate(paths.items()):
    
        mean_doy, std_upper_doy, std_lower_doy, df = aitken_VS_acc_geom_mean(path)
    
        ratio_mean = mean_doy['N_aitken'] / mean_doy['N_acc']
        ratio_lower = std_lower_doy['N_aitken'] / std_lower_doy['N_acc']
        ratio_upper = std_upper_doy['N_aitken'] / std_upper_doy['N_acc']
    
    
        
        # Plot
        ax1 = axs[i]
        
    
        # Plot data
        ratio_line = ax1.plot(mean_doy.index, ratio_mean, label= '$R_{mean}$', linestyle='-', color='crimson')[0]  #Aitken mode (22–90 nm) - 
        # ratio_line2 = ax1.plot(mean_doy.index, mean_doy['ratio'], label= '$R_{mean2, c.b}$', linestyle=':', color='red')[0]  #Aitken mode (22–90 nm) - 
    
        
        # Shaded quantile bands for Aitken - geometric
        ratio_feeling = ax1.fill_between(
            mean_doy.index,
            ratio_lower,
            ratio_upper,
            color='crimson',
            alpha=0.2,
            label= r'$R_{mean} \pm \sigma $'
        )
    
        # ratio_feeling2 = ax1.fill_between(
        #     mean_doy.index,
        #     std_lower_doy['ratio'],
        #     std_upper_doy['ratio'],
        #     color='red',
        #     alpha=0.1,
        #     label= r'$R_{mean2} \pm \sigma $'
        # )
    
    
        # Colored ticks
        ax1.tick_params(axis='y', colors='black')
    
        # Colored axis spines (optional for visual clarity)
        ax1.spines['left'].set_color('black')
    
    
        ax1.grid(True, linestyle='--', linewidth=0.5)
    
    
        # ax1.set_xlim(90, 180)
        # ax1.set_ylim(0,10)
    
        # Add legend in each subplot
        bold_font = FontProperties(weight='bold', size=15)
        legend_label = Patch(color='none', label=station + f' ({df.index.year.min()} - {df.index.year.max()})')
        ax1.legend(handles=[legend_label], loc='upper right', frameon=False , prop=bold_font)
    
    
    
    fig.text(0.08, 0.5,
             r"$R_{\text{mean}} = \frac{N_{\text{ait}}}{N_{\text{acc}}}$",
             va='center', rotation='vertical', fontsize=20, color='black')
    
    
    
    # Set global x-axis label
    fig.text(0.5, 0.08, 'Day of Year', ha='center', fontsize=16)
    
    
    # Add bottom shared legend
    
    fig.legend(
        [ratio_line, ratio_feeling], #, ratio_line2, ratio_feeling2
        [ratio_line.get_label(), ratio_feeling.get_label()], #, ratio_line2.get_label(), ratio_feeling2.get_label()
        loc='lower center',
        ncol=2,
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005)
    )
    
    
    # plt.tight_layout()
    plt.savefig('ratios.png', dpi=150)
    plt.show()

    return 





def analyze_station_for_ati(path, threshold=0.4, min_streak=10):
    
    '''Analyze a station's data to compute the Aerosol Transition Index (ATI) and determine the summer start day.'''

    df = pd.read_csv(path, sep='\t', header=46, index_col=0, parse_dates=True)
    size_bin_cols = [col for col in df.columns if 'dN/dlogDp' in col]
    diameters = np.array([float(re.search(r'\((.*?)\)', col).group(1)) for col in size_bin_cols])
    data = df[size_bin_cols]

    df['N_aitken'] = np.trapz(data.loc[:, (diameters >= 22) & (diameters <= 90)],
                              np.log10(diameters[(diameters >= 22) & (diameters <= 90)]), axis=1)
    df['N_acc'] = np.trapz(data.loc[:, (diameters >= 90)],
                           np.log10(diameters[(diameters >= 90)]), axis=1)

    df.loc[df['N_aitken'] < 1, 'N_aitken'] = np.nan
    df.loc[df['N_acc'] < 1, 'N_acc'] = np.nan

    df['ratio'] = df['N_aitken'] / df['N_acc']
    df['above1'] = df['ratio'] > 1
    df['under1'] = df['ratio'] < 1

    rolling = df[['above1', 'under1']].rolling(window='7d', center=True).sum()
    rolling['ATI'] = rolling['above1'] / rolling['under1']
    rolling['ATI_week_avg'] = rolling['ATI'].rolling('7d', center=True).mean()

    ati_grouped = rolling.groupby(rolling.index.dayofyear)['ATI_week_avg']
    ati_mean = ati_grouped.mean()
    ati_q25 = ati_grouped.quantile(0.25)
    ati_q75 = ati_grouped.quantile(0.75)

    # Summer start day
    # slice the ati dataframe between 100-180 and not from 90th day because there are some spikes of the ati around that day
    ati_slice = ati_mean.loc[(ati_mean.index >= 100) & (ati_mean.index <= 180)] 
    above_threshold = ati_slice > threshold
    streak = 0
    summer_start_day = None

    for doy, is_above in above_threshold.items():
        if is_above:
            streak += 1
            if streak == min_streak:
                summer_start_day = doy - min_streak + 1
                break
        else:
            streak = 0

    year_range = (df.index.year.min(), df.index.year.max())

    return ati_mean, ati_q25, ati_q75, summer_start_day, year_range






def plot_ati(paths, label_fontsize=17, legend_fontsize=15, title_fontsize=14):

    '''Plot the Aerosol Transition Index (ATI) for multiple stations.'''

    
    figsize = (12, 3 * len(paths))
    fig, axes = plt.subplots(len(paths), 1, sharex=True, figsize=figsize)

    if len(paths) == 1:
        axes = [axes]

    for ax, (station_key, path) in zip(axes, paths.items()):
        ati_mean, ati_q25, ati_q75, summer_start_day, year_range = analyze_station_for_ati(path)

        # print(f"{station_key}: Summer starts at DOY {summer_start_day}" if summer_start_day else f"{station_key}: No 10-day streak above threshold")

        x = ati_mean.index
        ax.plot(x, ati_mean, color='crimson')
        ax.fill_between(x, ati_q25, ati_q75, alpha=0.2, color='crimson')

        ax.axhline(0.4, color='black', ls='--', lw=0.6)

        if summer_start_day is not None:
            ax.axvline(summer_start_day, color='blue', linestyle='--', alpha=0.7, lw=1)
            ax.text(summer_start_day - 0.8, 1, f"Start: {summer_start_day}", rotation=90,
                    fontsize=legend_fontsize, color='blue', va='bottom', ha='right')

        ax.set_ylabel('ATI', fontsize=label_fontsize)

        # Legend
        bold_font = FontProperties(weight='bold', size=legend_fontsize)
        legend_label = Patch(color='none', label=f"{station_key} ({year_range[0]} - {year_range[1]})")
        ax.legend(handles=[legend_label], loc='upper left', frameon=False, prop=bold_font)

        ax.set_xlim(90, 180)
        ax.set_ylim(-0.2, 10)

    axes[-1].set_xlabel('Day of Year', fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()
    return






















































