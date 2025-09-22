import numpy as np
import matplotlib.pyplot as plt


class pyspin:
    """
    A class to simulate the centrifugation process for a colloids.

    Attributes:
        size (np.array): An array of particle sizes (in meters).
        inital_supernate (np.array): The initial supernatant distribution for each particle size.
        arm_length (float): The arm length of the centrifuge in meters (default: 0.1 m).
        length (float): The length of the centrifuge tube in meters (default: 0.01 m).
        liquid_density (float): The density of the liquid (kg/m^3) (default: 997 for water).
        liquid_viscosity (float): The viscosity of the liquid (Pa.s) (default: 1 for water).
        particle_density (float): The density of the particles (kg/m^3) (default: 2330 for silicon).
        supernate (list): List to store supernate distributions after each centrifugation cycle.
        pallets (list): List to store pallet distributions after each centrifugation cycle.
        rpms (list): List to store RPM values for each cycle.
        times (list): List to store time values for each cycle.

    Methods:
        __init__: Initializes the Centrifugation class with specified parameters.
        run_cycles: Runs multiple centrifugation cycles at specified RPMs and duration.
        cycle: Runs a single centrifugation cycle at a given RPM and duration.
        results: Returns the calculated results, including average particle sizes per cycle.
        cal_supernate_and_pallets: Calculates the remaining supernate and the resulting pallets after a centrifugation cycle.
        cal_centrifuge_change: Simulates the change in supernate and pallets across multiple centrifugation cycles.
        cal_sedimentation_rate: Calculates the sedimentation coefficient and rate for particles.
        _check_size: Checks that the size and initial supernate arrays are the same length.
        _clear_state: Clears the stored supernate and pallet data.
        _scale_check: Checks if the user input size scale is within the nanometer range.
        __str__: Returns a string representation of the centrifugation object.
    """

    def __init__(
        self,
        size: np.array = np.linspace(5, 250, 100) * 1e-9, # 5-250nm -> 5e-9 - 250e-9m
        inital_supernate: np.array = np.ones(100), # a.u
        arm_length=1e-1, # 10cm -> 0.1
        length=1e-2, # 1cm -> 0.01m
        liquid_density=997, # 997kg/m^3
        liquid_viscosity=0.899e-3, # at 25℃
        particle_density=2330, #233kg/m^3
    ):
        """
        Initializes the Centrifugation class with the given parameters.

        Args:
            size (np.array): An array of particle sizes (in meters).
            inital_supernate (np.array): The initial supernatant distribution.
            arm_length (float): Length of the centrifuge arm (default 0.1 m).
            length (float): Length of the centrifuge tube (default 0.01 m).
            liquid_density (float): The density of the liquid (default: 997 kg/m^3).
            liquid_viscosity (float): The viscosity of the liquid (default: 1 Pa.s).
            particle_density (float): The density of the particles (default: 2330 kg/m^3).
        """
        self.size = size
        self.inital_supernate = inital_supernate
        self.supernate = []
        self.pallets = []
        self._check_size()
        self._scale_check()
        self.count = len(self.size)
        self.mode = 'pal'

        self.accel_time = 20
        self.decel_time = 20

        # Centrifugation machine properties
        self.arm_length = arm_length  # length of centrifuge 10cm  (m)
        self.length = length  # tube length 1cm (m)
        self.liquid_density = liquid_density  # water (kg/m^2)
        self.liquid_viscosity = liquid_viscosity  # water (mPa.s)
        self.particle_density = particle_density  # Silicon (kg.m^2)
        self.rpms = []  # empty list to store the rpms
        self.times = [] # emptylist to store the times

    def info(self):
        """
        Basic information about the centrifugation object

        Returns:
            dict: details about the centrifuge and it's setup
        """
        text = {
            "Colloid Info": {
                "Particle Count": self.count,
                "Particle Radii Range (m)": [np.min(self.size), np.max(self.size)],
                "Average Inital Radii (m)": np.average(
                    self.size, weights=self.inital_supernate
                ),
                "Particle Density (kg/m^3)": self.particle_density,
                "Liquid Density (kg/m^3)": self.liquid_density,
                "Liquid Viscosity (Pa.s)": self.liquid_viscosity,
            },
            "Centrifuge Info": {
                "Arm Length (m)": self.arm_length,
                "Tube length (m)": self.length,
                "RPMS": self.rpms,
            },
        }
        return text

    def run_cycles(self, rpms: list, duration):
        """
        Runs multiple centrifugation cycles at specified RPMs and duration.

        Args:
            rpms (list): List of RPMs for each cycle.
            duration (float or list): If a float is provided, it applies to all cycles.
                                    If a list is provided, it must be the same length as `rpms`,
                                    specifying the duration for each corresponding RPM.

        Raises:
            ValueError: If the duration is a list and its length does not match the number of RPMs.
        """
        # Check if duration is a list or a single float
        if isinstance(duration, list):
            if len(duration) != len(rpms):
                raise ValueError(
                    "The length of the duration list must match the number of RPMs."
                )
            durations = duration  # Use the list as-is
        else:
            durations = [duration] * len(
                rpms
            )  # Repeat the single duration for each RPM

        # Run each cycle with the corresponding RPM and duration
        for rpm, dur in zip(rpms, durations):
            self.cycle(rpm, dur)

    def cycle(self, rpm, duration):
        """
        Runs a single centrifugation cycle at the specified RPM and duration.

        Args:
            rpm (int): The RPM for this cycle.
            duration (float): The duration of the cycle in seconds.
        """

        # Collected the most recent supernate data
        if not self.supernate:
            inital_supernate = self.inital_supernate.copy()
        else:
            if 'sup' in self.mode.lower():
                print('Using previous supernate')
                inital_supernate = self.supernate[-1].copy()
            else:
                print('Using previous Pallet')
                inital_supernate = self.pallets[-1].copy()

        supernate, pallets = self.cal_supernate_and_pallets(
            rpm, duration, inital_supernate
        )

        # Save data to state
        self.supernate.append(supernate)
        self.pallets.append(pallets)
        self.rpms.append(rpm)
        self.times.append(duration)

        print(f"Centrifuge cycle at {rpm/1000:.0f}K RPM over {duration}min completed")

    def results(self, avg = True, supernates = False):
        """
        Returns the results of the centrifugation, including average particle sizes per cycle.

        Args:
            avg (bool): Whether to include average particle sizes in the results (default: True).

        Returns:
            dict: A dictionary containing the particle radii, pallets, supernates, and (optionally) average sizes per cycle.
        """
        # returns the calcuated results in a dict with average particle size per cycle (as an option)
        results = {'Radii (nm)': self.size * 1e9} # converting size to nm scale 
        for i in range(len(self.rpms)):
            rpm = self.rpms[i]

            results[f'{rpm/1000:.0f}kp'] = self.pallets[i] # pallet stats
            if supernates:
                results[f'{rpm/1000:.0f}ks'] = self.supernate[i] # supernate states

        # only get the last one --> simular to experimental data
        if not supernates:
            results[f'{rpm/1000:.0f}ks'] = self.supernate[i] # supernate states


            if avg ==  True:
                results[f'{rpm/1000:.0f}kp_avg'] = np.average(self.size, weights=self.pallets[i]) # avg particle size
                results[f'{rpm/1000:.0f}ks_avg'] = np.average(self.size, weights=self.supernate[i]) # avg particle size

        return results

    def cal_supernate_and_pallets(
        self, rpm, duration, inital_supernate, normalise=False, size=None
    ):
        """
        Calculates the remaining supernate and the resulting pallets after a centrifugation cycle.

        Args:
            rpm (int): The RPM of the centrifugation cycle.
            duration (float): The duration of the cycle in seconds.
            inital_supernate (np.array): The initial distribution of particles in the supernate.
            normalise (bool): Whether to normalise the supernate and pallets distributions (default: True).
            size (np.array, optional): An array of particle sizes. If None, uses self.size.

        Returns:
            tuple: A tuple containing the supernate and pallets arrays after the centrifugation cycle.
        """
        # Cal sedmentaiton rates
        sed_coefficient, sed_rate = self.cal_sedimentation_rate(rpm, size)

        omega = rpm * 2 * np.pi / 60 

        # The smallest r to be palleted based at the bottom of the centrifige container.
        r_0 = self.arm_length * np.exp(-sed_coefficient * omega**2 * (duration - 2*(self.accel_time+self.decel_time)/3))

        # the position for each size that would be palleted during centrifugation
        pos_percent_pallet = (self.arm_length - r_0)/self.length


        # Sets any percentage position above 1 to be = 1 
        pos_percent_pallet = np.where(pos_percent_pallet > 1, 1, pos_percent_pallet)

        supernate = inital_supernate * (1 - pos_percent_pallet)


        # # Calculates the remaining % of supernate
        # supernate = inital_supernate * (
        #     (self.length - (sed_rate * duration)) / self.length
        # )

        # Sets all negative values to 0
        supernate = np.where(supernate < 0, 0, supernate)

        pallets = inital_supernate - supernate

        if normalise:
            # Normalising the Supernate and Pallets --> see centrifugation theory
            
            def min_max_norm(data):
                max_value = data.max()
                min_value = data.min()
                data = (data - min_value) / (max_value - min_value)

                return data

            supernate = min_max_norm(supernate)
            pallets = min_max_norm(pallets)


        return supernate, pallets

    def cal_centrifuge_change(self, size, rpms, duration=60, inital_supernate=1):
        """
        Simulates the change in supernate and pallets across multiple centrifugation cycles.

        Args:
            size (np.array): An array of particle sizes.
            rpms (list): A list of RPM values for the centrifugation cycles.
            duration (float): The duration of each cycle in seconds (default: 60 seconds).
            inital_supernate (float or np.array): The initial supernate percentage or distribution (default: 1, representing 100%).

        Returns:
            dict: A dictionary containing the resulting pallets ('kp') and supernate ('ks') for each RPM cycle.
        """
        time = np.linspace(0, duration, 100)

        results = {}
        for rpm in rpms:
            supernate, pallets = self.cal_supernate_and_pallets(
                rpm, time, inital_supernate, size=size
            )
            results[f"{rpm:.0f}kp"] = pallets
            results[f"{rpm:.0f}ks"] = supernate

            # updated the inital supernate percent start where the previous cycle ends
            inital_supernate = supernate[-1]

        return results

    def cal_sedimentation_rate(self, rpm, size=None):
        """
        Calculates the sedimentation coefficient and rate for the particles.

        Args:
            rpm (int): The RPM of the centrifugation cycle.
            size (np.array, optional): An array of particle sizes (default: uses self.size).

        Returns:
            tuple: The sedimentation coefficient and rate.
        """

        if size is None:
            size = self.size

        # Calculates the sedimentation rate and coefficent
        angular_velocity = rpm * 2 * np.pi / 60 # Convert RPM to rad/s

        sed_coefficient = (
            2 * (size**2) * (self.particle_density - self.liquid_density)
        ) / (
            9 * self.liquid_viscosity
        )  # s = (2r^2(ρ_s - ρ_w) / (9 * liquid_viscosity)
        sed_rate = (
            (angular_velocity**2) * self.arm_length * sed_coefficient
        )  # ⍵^2 * r * s --> in cm/s

        return sed_coefficient, sed_rate

    def plot_cycles(self, ax=None, fig=None):
        """
        Plots the supernate and pallet compositions across multiple centrifugation cycles.

        -- Its recommended the user generates plots using state or results() method --

        This method creates a plot showing the initial supernate distribution and the resulting
        supernate and pallet distributions after each centrifugation cycle. Each distribution is
        plotted as a function of particle radius.

        The plot uses different colors for each cycle, with a color list generated
        based on the number of particle sizes plus one for the initial state.

        The function iterates through the stored supernate and pallet data, directly plotting
        them rather than using the `results` method.

        Returns:
            tuple: A tuple containing the Matplotlib figure and axis objects (fig, ax).
        """

        print('-- Its recommended the user generates plots using state or results() method --')

        # Creates the Matplotlib fig objs if not parsed
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(5, 6),  sharex="col", sharey="row")

        # Creates a color list the size of number of particle sizes
        colors = self.generate_color_list(
            len(self.supernate) + 1
        )  # +1 to allow for inital state color

        ax.plot(
            self.size * 1e9,
            self.inital_supernate * 1e2,
            label="Inital Supernate",
            linewidth=2,
            color=colors[0],
        )

        # Use the pyspin state to plot, not the results method
        # itterate through the different particle sizes
        for ii in range(len(self.supernate)):

            # Supernate composition(%)
            ax.plot(
                self.size * 1e9,
                self.supernate[ii] * 1e2,
                label=f"{self.rpms[ii]/1000:.0f}ks",
                linewidth=2,
                linestyle="-.",
                color=colors[ii + 1],
            )

            # Pallet composition(%)
            ax.plot(
                self.size * 1e9,
                self.pallets[ii] * 1e2,
                label=f"{self.rpms[ii]/1000:.0f}kp",
                linewidth=2,
                linestyle="-.",
                color=colors[ii + 1],
            )

        # X-axis label
        ax.set_xlabel("Particle Radius (nm)")

        # Y-axis labels
        ax.set_ylabel("Composition (%)")

        # Shared legend below the plots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        return fig, ax

    def generate_color_list(self, n, cmap_name="viridis"):
        """
        Generates a list of n colors from a specified Matplotlib colormap.

        Args:
            n (int): The number of colors to generate.
            cmap_name (str): The name of the Matplotlib colormap to use (default: 'viridis').

        Returns:
            list: A list of RGBA color tuples.
        """
        cmap = plt.get_cmap(cmap_name)  # Get the colormap
        colors = [cmap(i / n) for i in range(n)]  # Generate n colors
        return colors

    def _check_size(self):
        """
        Checks that the size and initial supernate arrays have the same length.

        Raises:
            ValueError: If the size and inital_supernate arrays do not match in length.
        """
        if len(self.size) != len(self.inital_supernate):
            raise ValueError(
                f"Size mismatch: Size has size ({len(self.size)}), but inital_supernate has size ({len(self.inital_supernate)})"
            )
        return

    def _clear_state(self):
        """
        Clears the stored supernate and pallet data.
        """
        self.supernate = []
        self.pallets = []

    def _scale_check(self):
        """
        Checks if the user input size scale is within the nanometer range.

        The function will raise an error if any particle size is larger than or equal to 1 cm (1e-2 meters).
        It will issue a warning if any particle size is larger than or equal to 1 µm (1e-6 meters) but smaller than 1 cm.

        Raises:
            ValueError: If any particle size is larger than or equal to 1 cm.
            Warning: If any particle size is larger than or equal to 1 µm but smaller than 1 cm.
        """
        # Check if the size is larger than or equal to 1 cm
        if np.any(self.size >= 1e-2):
            raise ValueError(
                f"Invalid particle size found ({len(self.size[self.size >= 1e-2])}): {self.size[self.size >= 1e-2]}. Particle sizes must be smaller than 1 cm."
            )

        # Check if the size is larger than or equal to 1 µm but smaller than 1 cm
        elif np.any(self.size >= 1e-6):
            print(
                f"Warning: Large particle size found ({len(self.size[self.size >= 1e-6])}): {self.size[self.size >= 1e-6]}. Particles should ideally be smaller than 1 µm."
            )

    def __str__(self):
        return str(self.info())
