#  Loading in the Libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Loading in the Simulation Objects
from pyspin import *


# ------------
# Display properties
# ------------
# Set page layout to wide
st.set_page_config(page_title="Nano Particle Simulation", 
                   initial_sidebar_state="collapsed",
                   page_icon="img/NPP-icon-blueBG.png")
st.set_option('deprecation.showPyplotGlobalUse', False)


# makes the plots in line with the style of the application dark mode
rc = {'figure.figsize':(6,5),

        'axes.facecolor':'#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size' : 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16}
plt.rcParams.update(rc)

color_list = ['#4a678c','#9d0a5b', '#577c9b', '#bc3e47', '#679eb1', '#5d87a2', '#6292aa','#517193', '#ffd6ba', '#f7af95', '#e98975', '#d56459']


def get_file_names(folder_path : str, filter_pattern : str = None):
    """
    Get a list of filenames in the specified folder path.

    Args:
        folder_path (str): The path to the folder from which to retrieve filenames.
        filter_pattern (str): A pattern to filter the list of filenames by --> recommended to use file types

    Returns:
        list of str: A list of filenames in the specified folder.
    """

    file_names = os.listdir(folder_path)

    # filter out the list of filenames based on a pattern
    if filter_pattern is not None:
        file_names = [col for col in file_names if filter_pattern in col.lower()]

    return file_names

def load_data_from_txt(file_path : str, header : list[str] = None) -> pd.DataFrame:
    """
    Load data from a .txt file.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data from the .txt file.
            Returns None if the file is not found or an error occurs during loading.
    """
    try:
        data = pd.read_csv(file_path, sep='\t', header=header)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from .txt file: {str(e)}")
        return None

def open_data_file(basepath : str, field : str, file_filter : str = ".txt", col_formatter : str = ".") -> pd.DataFrame:
    """
    Loads and combines multiple text files from a specified directory into a single pandas DataFrame.
    
    Each text file should contain two columns: one for particle radii and another for particle concentration.
    The function merges these data files based on the particle size (radii) and uses the filenames as column headers
    for the concentration data.

    Args:
        basepath (str): The base directory path where the text files are located.
        field (str): The name of the first column representing the particle radii.
        file_filter (str, optional): The file extension filter to select files from the base directory (default is ".txt").
        col_formatter (str, optional): The delimiter used to format column names from filenames (default is ".").
                                       Only the part of the filename before the first occurrence of this delimiter
                                       will be used as the column name.

    Returns:
        pd.DataFrame: A merged DataFrame containing all the data from the text files, where the first column is the
                      particle size (radii) and subsequent columns represent the concentration data from each file.

    Raises:
        FileNotFoundError: If no files matching the filter are found in the base directory.
        ValueError: If any file cannot be loaded into a DataFrame or if merging fails.
    
    Example:
        >>> load_df_from_txts('/data/particles/', 'Radii', '.txt', '_norm')
        Returns a DataFrame where the first column is 'Radii' and subsequent columns are concentrations for different 
        files, named after their respective file prefixes.
    """
    # Retive the files names from base path that match the file_filter patten
    file_names = get_file_names(basepath, file_filter)

    # This list will hold the df for each of the files loaded
    df_list = []

    # Generate a df for each file in the basepath
    # and add it to the df_list --> will merge latter
    for name in file_names:
        df = load_data_from_txt(basepath + name, header = None)

        # Chnaging the column names
        # col_formatter will cut the stings off and take the first half
        # i.e "1.8W - 21-08 - 14Ks_norm.txt" col_formatter = "." --> "1.8W - 21-08 - 14Ks_norm"
        df.columns = [field,  name.split(col_formatter)[0]] 

        # Check if the fild is on the nano-scale with SI units
        # if so, convert to nm from SI units
        # Else assume alread in nm units
        if df[field].max() <= 1e-6:
            df[field] *= 1e9

        # Adding the df to the list
        df_list.append(df)

    # need an initial df to merge the others onto
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=field, how='outer')  # Outer join to ensure all radii are included

    return merged_df

def get_mass_precent(data_in : pd.DataFrame, norm_mode = 'L1') -> pd.DataFrame:
    """
    Calculate the mass percentage for each particle size in the given DataFrame, and optionally normalize the data.

    This function computes the mass of silicon particles based on their radii (given in nanometers) and the known 
    density of silicon (2330 kg/m^3). The mass is calculated for each size, and the input data is scaled accordingly.
    Optionally, the data can be normalized using the specified normalization mode.

    Args:
        data_in (pd.DataFrame): The input DataFrame containing the particle size data (column 'Radii (nm)') and 
                                corresponding raw measurements for each sample.
        norm_mode (str, optional): The normalization mode to use. If provided, the data will be normalized. 
                                   The default is 'L1'. If set to `None`, no normalization is applied.

    Returns:
        pd.DataFrame: A DataFrame containing the mass percentage of the samples. If normalization is applied, 
                      the '_norm' suffix is removed from the column names. The 'Radii (nm)' column is retained 
                      in the output.

    Steps:
        1. Compute the volume of each particle assuming spherical particles.
        2. Compute the mass of each particle using the volume and the density of silicon (2330 kg/m^3).
        3. Scale the sample data in the DataFrame by the calculated mass for each particle size.
        4. Optionally normalize the data using the specified `norm_mode`.
        5. Return the processed DataFrame with the mass percentage for each sample and the particle size ('Radii (nm)').

    Example:
        >>> data_in = pd.DataFrame({
        ...     'Radii (nm)': [10, 20, 30],
        ...     'Sample1': [0.1, 0.2, 0.3],
        ...     'Sample2': [0.4, 0.5, 0.6]
        ... })
        >>> result = get_mass_precent(data_in)
        >>> print(result)

    Notes:
        - The function assumes spherical particles for volume calculation.
        - The default normalization mode is 'L1'. Other modes can be specified, or set to `None` to skip normalization.
        - The output DataFrame retains the 'Radii (nm)' column and removes any temporary columns (e.g., 'Bins') if present.

    """

    data = data_in.copy()

    volume = (4/3) * np.pi * np.power((data_in['Radii (nm)'] * 1e-9), 3) # volume in SI --> m^3

    silicon_density = 2330 #kg/m^3

    mass = volume * silicon_density

    # Drops the Radius column -> not needed for the analysis
    data = data.drop('Radii (nm)', axis = 1)

    # Removes the Bins field if present
    if 'Bins' in data.columns:
        data = data.drop('Bins', axis = 1)
    

    for y_col in data.columns:
        data[y_col] = data[y_col] * mass


    data['Radii (nm)'] = data_in['Radii (nm)']

    # Will normalise the data if a mode is proided
    if norm_mode is not None:
        data = normalize_data(data, ['Radii (nm)'], mode = norm_mode)

        # Only return the norm values
        data = data.filter(like='_norm')

    # Remove the _norm* from each of the column names
    data.columns = data.columns.str.replace(r'_.*', '', regex=True)


    # Add the radii column back to the df
    data['Radii (nm)'] = data_in['Radii (nm)']

    return data

def get_df_bins(data: pd.DataFrame, interval: int = 20, bin_edges = None) -> pd.DataFrame:
    """
    Split the DataFrame into bins based on the 'Radii (nm)' column with the given interval, 
    and calculate the mean for each bin.

    Args:
        data (pd.DataFrame): The input DataFrame.
        interval (int): The interval size for binning.

    Returns:
        pd.DataFrame: A DataFrame containing the mean values for each bin.
    """
    
    if bin_edges is not None:
        bins = bin_edges
    else:
        # Define the bin edges (ranges from the min to max of 'Radii (nm)' with the given interval)
        bins = np.arange(data['Radii (nm)'].min(), data['Radii (nm)'].max() + interval, interval)
    
    print(bins)
    # Use pd.cut() to bin the 'Radii (nm)' column into intervals
    data['Bins'] = pd.cut(data['Radii (nm)'], bins=bins, right=False)
    
    # This will store the list of means for each bin
    mean_list = []

    # Finding the mean for each bin
    for bin_interval in data['Bins'].unique():
        # Get data for the current bin
        data_section = data[data['Bins'] == bin_interval]
        
        # If there's data in this bin, calculate the mean (excluding non-numeric columns like 'Bins')
        if not data_section.empty:
            bin_mean = data_section.drop(columns=['Bins']).sum()  # Exclude non-numeric columns
            # Store the bin mean
            mean_list.append(bin_mean)

    # Merging all bin means into a single DataFrame
    mean_df = pd.concat(mean_list, axis=1).T.reset_index(drop=True)
    
    # Force the first column (which is likely the 'Radii (nm)' mean) to be int
    mean_df.iloc[:, 0] = mean_df.iloc[:, 0].astype(int)

    # this will replace the radii names with the meddian values of each bin
    mean_df['Radii (nm)'] = bins[:-1] + np.round(interval / 2, 0)

    return mean_df

Centrifugation = pyspin(size=np.linspace(1,150,100) * 1e-9)

with st.sidebar:
    Centrifugation.particle_density = 2330 # density of the particles used
    Centrifugation.liquid_density = 997 # default density 
    Centrifugation.liquid_viscosity = 1 # default density iw water at 20C
    Centrifugation.arm_length = 9.5 * 1e-2 # in cm
    Centrifugation.length = 2 * 1e-2 # in cm

    duration = 10 # Duration of Centrifugation (min)
    rpm = 10000 # RPM of the centrifuge 

time = np.linspace(0,duration,100)

runs = 2 
prob = 1

# Resting the Centrifugation object
Centrifugation.inital_supernate = np.ones(100) * prob
Centrifugation.run_cycles([rpm, rpm], duration)

st.header("Centrifugation of Colloids")


with st.expander("How does the ratio of compasition over time change?"):
    text=r"""
    ### Determining the Amount of Supernatant Remaining Over Time

    The process of determining how much supernatant remains after a given amount of time during centrifugation can be described by the following equation:

    $$
    P(r, t) = \frac{\text{length} - (\text{sedimentation velocity} \times \text{time})}{\text{length}}
    $$

    Where:
    - \( P(r, t) \) is the ratio of the remaining supernatant to the starting composition after time \( t \) at a distance \( r \) from the center.
    - **length** refers to the total length of the container that holds the colloid.
    - **sedimentation velocity** is the velocity at which the particles settle out of the suspension due to the centrifugal force.
    - **time** is the duration for which the centrifugation process has been running.

    ### Explanation of the Ratio Changes Over Time

    The equation \( P(r, t) \) represents how much of the original supernatant remains in the container at any given time during centrifugation. Initially, at \( t = 0 \), the amount of supernatant is equal to the original amount, so \( P(r, 0) = 1 \) (or 100% of the original amount).

    As time progresses (\( t > 0 \)), particles in the colloid start to sediment due to the applied centrifugal force. This sedimentation reduces the amount of remaining supernatant, as indicated by the term \( \text{sedimentation velocity} \times \text{time} \) in the equation. The longer the centrifugation time, the more significant this term becomes, leading to a decrease in \( P(r, t) \).

    When the sedimentation velocity and time are large enough, the product \( \text{sedimentation velocity} \times \text{time} \) can approach the value of **length**. At this point, \( P(r, t) \) approaches zero, indicating that nearly all the supernatant has been removed, and the particles have sedimented completely.

    In summary, the ratio \( P(r, t) \) decreases over time as more particles settle out of the suspension, reducing the amount of remaining supernatant. The rate at which this decrease happens depends on the sedimentation velocity and the duration of the centrifugation process.
    """
    st.markdown(text)

st.divider()

text_col, plot_col = st.columns([1,1])

with text_col:
    st.subheader("Overall Composition of the Colloid")
    st.markdown(r"""
    The Supernate (Solid line) is the particle sizes the remain suspended in the water after the centrifugation process.
    
    The Pallets (dotted line) are the partiles that have reached the bottom of the container.
                
    The centrifugation settings can be edited in the side pannel.
    """)

with plot_col:

    fig, ax = plt.subplots()

    ax.plot(Centrifugation.size*1e9, Centrifugation.inital_supernate * 1e2, label=f"Inital state", linewidth=2, color = color_list[-1])

    index = 0 # Iditrator to run through each color when plotting --> helps make shit look nicer
    for supernate, pallet in zip(Centrifugation.supernate, Centrifugation.pallets):

        ax.plot(Centrifugation.size*1e9, supernate * 1e2, label=f"Cycle: {index +1}", alpha = 1, linewidth=2, color=color_list[index]) 
        ax.plot(Centrifugation.size*1e9, pallet * 1e2, alpha=1, linestyle='--', color=color_list[index])
        index += 1

    ax.set_xlabel("Particle Radius (nm)")
    ax.set_ylabel("Composition (%)")
    ax.set_title(f"Colloid Centrifuge")
    ax.legend()

    st.pyplot(fig)
    st.caption(f'Centrifugation Time: {duration}, Centrifugation Speed: {rpm *1e-3 :.0f}K')


st.divider()

text_col, settings_col = st.columns([1,1])

with text_col:
    st.subheader("Centrigugation with inital states.")
    st.markdown(r"""
    When using the inital concentration values experimentally measured from samples produced at 1.8W PLAL settings.
                
    Below is are the concentrations measured before and after the centrifugation cycles.
    """)

    data=open_data_file('src/', 'Radii (nm)', col_formatter='_')
    data.columns = ['Radii (nm)', 'Before Centrifugation', 'After Centrifugation']


    fig, ax = plt.subplots(1, figsize=(6,5)) # Set the figure size
    y_columns = data.columns[1:]  # Assuming first column is 'Wavelength (nm)'

    for column in y_columns:
        ax.plot(data['Radii (nm)'], np.log10(data[column]), label=column, linewidth=2)

    ax.legend()
    ax.set_ylabel("log10 [Concentration (a.u)]")
    ax.set_xlabel("Particle Radii(nm)")
    ax.set_title("Concentrations over centrifugation cycles")

    st.pyplot(fig)


with settings_col:
    Centrifugation.particle_density = st.number_input(r"Density of the colloids ($$kg/m^2$$)", 500, 3000, 2330) # density of the particles used
    Centrifugation.liquid_density = st.number_input(r"Density of liquid ($$kg/m^2$$)", 50, 3000, 997) # default density 
    Centrifugation.liquid_viscosity = st.number_input(r"Viscosity of liquid ($$m Pa.s$$)", 0.1, 2.0, 1.0) # default density iw water at 20C
    Centrifugation.arm_length = st.number_input(r"Centrifuge arm length ($$cm$$)", 1, 20, 10) * 1e-2
    Centrifugation.length = st.number_input(r"Length of the container ($$cm$$)", 1, 20, 1) * 1e-2

    duration = st.number_input(r"Duration ($$min$$)", 1, 120, 10) # Duration of Centrifugation
    rpm = st.number_input(r"Centrifuge speed ($$RPM$$)", 1, 40000, 4000) # RPM of the centrifuge 
    # where all the indivdual settings for the centrifugation can be re-0defined (multiple runs, speed and so on)

    # If the settings are changed, do the files get lost? how to keep them in memory (this may not be a problem at all.) -> link to google docs? could be better


st.subheader("Mass Percent Compositions")
measured_plot_col, modelled_plot_col = st.columns([1,1])

with measured_plot_col:
    mass = get_mass_precent(data, norm_mode=None)

    mass_percent = mass / mass.sum() * 100

    # Need to re-estabilish the correct radii values
    mass_percent['Radii (nm)'] = data['Radii (nm)']

    bins = [20, 35, 50, 60, 70, 85, 90, 100, 110]

    mass_percent_bins = get_df_bins(mass_percent, bin_edges=bins)

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots()

    # Plot the data on the axis
    mass_percent_bins.plot(x='Radii (nm)', kind='bar', ax=ax)

    ax.set_title("Experimental Data")
    ax.set_xlabel("Particle Radii(nm)")
    ax.set_ylabel("Mass Composition (%)")


    # Display the figure in Streamlit
    st.pyplot(fig)


# Running the Centrigugation model






