#  Loading in the Libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Loading in the Simulation Objects
from pyspin import *


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

input_col, plot_col = st.columns([1,2])
with input_col:
    # Inputs using Streamlit number input widgets
    rpm = st.number_input('Centrifuge Speed (RPM)', min_value=1000, max_value=15000, value=5000, step=500)
    duration = st.number_input('Duration (minutes)', min_value=1, max_value=60, value=17)

# Constants
arm_length = 15 * 1e-2  # 15 cm
length = 4 * 1e-2  # 4 cm liquid depth

# Making the centrifuge object
cen = pyspin(np.linspace(1, 250, 100) * 1e-9, np.ones(100))
cen.arm_length = arm_length
cen.length = length
cen.liquid_viscosity = 1.5182 * 1e-3

cen.cycle(rpm, duration)

# Get results and plot
results = cen.results()
fig, ax = plt.subplots(1, figsize=(5, 4), sharex=True)
ax.set_title(f'{cen.rpms[0]} RPM - {duration} min')
ax.plot(cen.size * 1e9, cen.pallets[-1] * 100)

# Add text to the bottom of the figure
text = f"Arm length: {cen.arm_length * 1e2:.0f}cm, liquid depth: {cen.length * 1e2:.0f}cm"
fig.text(0.40, 0.80, text, ha='center', fontsize=10)

ax.axvline(x=100, color='gray', linestyle="--")
first_max_index = np.where(cen.pallets[-1] == 1)[0][0]
ax.axvline(x=cen.size[first_max_index] * 1e9, color='red', linestyle="--")

ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Palleted (%)")
# Show the plot in Streamlit
with plot_col:
    st.pyplot(fig)

st.divider()

input_col, plot_col = st.columns([1,2])
with input_col:
    # Inputs using Streamlit number input widgets
    rpm = st.number_input('(RPM)', min_value=1000, max_value=15000, value=5000, step=500)
    particle_size = st.number_input('Particle size (nm)', min_value=1, max_value=250, value=50)

# Run centrifugation simulation
sedimentation_rate = cen.cal_sedimentation_rate(rpm, particle_size * 1e-9)[1]
time_to_sedimentation = cen.length / sedimentation_rate

with plot_col:
    results_table = pd.DataFrame({
        "Parameter": ["Sedimentation Rate (m/s)", "Sedimentation Rate (m/min)", "Time to Sedimentation (min)"],
        "Value": [f"{sedimentation_rate:.3g}", f"{sedimentation_rate / 60:.3g}", f"{time_to_sedimentation:.3g}"]
    })
    
    # Display the results as a table
    st.table(results_table) 