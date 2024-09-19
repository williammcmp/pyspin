#  Loading in the Libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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


Centrifugation = pyspin(size=np.linspace(1,120,100) * 1e-9)

with st.sidebar:
    Centrifugation.particle_density = st.number_input(r"Density of the colloids ($$kg/m^2$$)", 500, 3000, 2330) # density of the particles used
    Centrifugation.liquid_density = st.number_input(r"Density of liquid ($$kg/m^2$$)", 50, 3000, 997) # default density 
    Centrifugation.liquid_viscosity = st.number_input(r"Viscosity of liquid ($$m Pa.s$$)", 0.1, 2.0, 1.0) # default density iw water at 20C
    Centrifugation.arm_length = st.number_input(r"Centrifuge arm length ($$cm$$)", 1, 20, 10) * 1e-2
    Centrifugation.length = st.number_input(r"Length of the container ($$cm$$)", 1, 20, 1) * 1e-2

    duration = st.number_input(r"Duration ($$min$$)", 1, 120, 10) # Duration of Centrifugation
    rpm = st.number_input(r"Centrifuge speed ($$RPM$$)", 1, 40000, 4000) # RPM of the centrifuge 

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
    Upload your concentration data for the inital state and see how the centrifugation will effect the size distribution of the colloids
    """)
    # Add upload link for the file.


    # Add download link for an example file of the concentration --> with Template.

with settings_col:

    pass
    # where all the indivdual settings for the centrifugation can be re-0defined (multiple runs, speed and so on)

    # If the settings are changed, do the files get lost? how to keep them in memory (this may not be a problem at all.) -> link to google docs? could be better

# The plot the distributions after each cycles 
# each cycle gets its own row with a col for the standard plot (see above) and the weighted mass comps (see final excel sheet)

