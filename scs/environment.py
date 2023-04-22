from __future__ import annotations

import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from scs.owl_scout import OSController, OwlScout
from scs.owls import Hoot, Owl

logger = logging.getLogger(__name__)


class Environment:
    """Environment to simulate the Owl Scout system

    The environment in which the Owl Scout system is simulated. Holds all required
    objects and controls the environments flow of time, the propagation of
    sound, and initialises the timstep updates for all dynamic elements of the
    system.

    Attributes:
        owls: A list containing all the Owl entities of the simulation
        hoots: A list containing all hoots that are audible in the local environment
        owl_scout_controller: The OSC instance contorlling the owl scout network
        timestep_duration: How much passes during one simulation timestep in seconds
        owl_update_frequency: How many simulation timesteps in between owl position
            updates as the owls do not need a time resolution as high as the
            sound propagation
        timestep_delta: Timedelta to add to the simulation time during each timestep
        time: The current time in the simulation
        sound_v: Velocity of sound in air in m/s
        sound_max_distance: The maximum distance from a hoots origin at which it
            is still audible
        timstep_count: How many timesteps have been simulated since the last reset
    """

    def __init__(self) -> None:
        """Initializes the environment"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.owls: dict[str, Owl] = {}
        self.hoots: list[Hoot] = []
        self.owl_scout_controller: OSController = OSController()
        self.timestep_duration: float = 1 / 10000
        self.owl_update_frequency: int = 10000
        self.timestep_delta: timedelta = timedelta(milliseconds=0.1)
        self.time: datetime = datetime.now()
        self.sound_v: int = 343
        self.sound_max_distance: int = 10000
        self.timestep_count: int = 0
        self.logger.debug(
            f"Initialized environment with parameters:\n"
            f"\tTimstep duration: {str(self.timestep_duration)} seconds\n"
            f"\tOwl update frequency: "
            f"{str(self.timestep_duration * self.owl_update_frequency)} seconds\n"
            f"\tSpeed of sound: {str(self.sound_v)} m/s\n"
            f"\tMaximum sound progression distance: {str(self.sound_max_distance)} "
            f"seconds"
        )

    def simulate(self, for_seconds: int) -> None:
        """
        Simulates the environment for a passed time in seconds. Runs through as
        many timesteps as required to match the passed time based on the
        timestep duration.

        Args:
            for_seconds: Number of seconds to be simulated
        """
        timesteps = for_seconds * 10000
        one_percent = timesteps / 100
        percent_count = 0
        for t in range(timesteps):
            self.timestep()
            percent_count += 1
            if percent_count == one_percent:
                percent_count = 0
                self.logger.debug(
                    f"Simulation rogress at {str(int((t + 1) // one_percent))}%"
                )

    def timestep(self) -> None:
        """
        Simulates one timestep in the environment. Updates the simulation time,
        the sound propagation, collects the microphon signals, and updates the
        owls states based on the owl update frequency.
        """
        self.timestep_count += 1
        self.time += self.timestep_delta
        self._propagate_hoot_sound()
        self.owl_scout_controller.collect_events(self.hoots, self.time)
        if self.timestep_count == self.owl_update_frequency:
            self.logger.debug("Update owls")
            self._update_owl_movement()
            self.timestep_count = 0

    def _propagate_hoot_sound(self) -> None:
        """
        Updates the radius the sound has traveled from its origin for each Hoot
        instance and removes the hoots that are no longer audible in the local
        environment.
        """
        keep_hoots = []
        for hoot in self.hoots:
            radius = hoot.extend_radius(self.sound_v * self.timestep_duration)
            if radius < self.sound_max_distance:
                keep_hoots.append(hoot)
            else:
                self.logger.debug(
                    f"Removed hoot '{hoot.hoot_id}' with sound progression radius "
                    f"{str(radius)} > max radius of {str(self.sound_max_distance)}"
                )
        self.hoots = keep_hoots

    def _update_owl_movement(self) -> None:
        """
        Initiates the state updates for all owls in the environment and places
        hoots, as occurring, into the simulation process.
        """
        for _, owl in self.owls.items():
            hoot = owl.timestep(self.owl_update_frequency * self.timestep_duration)
            if hoot:
                self.hoots.append(hoot)

    def add_owl(self, new_owl: Owl | tuple[str, tuple[float, float]]) -> None:
        """
        Adds a new owl to the environment. Takes an Owl instance to add or creates
        and adds a new Owl instance from passed arguments.
        """
        if isinstance(new_owl, Owl):
            self.owls[new_owl.id] = new_owl
        else:
            self.owls[new_owl[0]] = Owl(*new_owl)
        self.logger.debug("Owl added to environment")

    def add_owl_scout(
        self, os_scout: OwlScout | tuple[str, tuple[float, float], float]
    ) -> None:
        """
        Adds a new owl scout to the controllers network. Takes an OwlScout instance
        to add or creates and adds a new OwlScout instance from passed arguments.
        """
        if isinstance(os_scout, OwlScout):
            self.owl_scout_controller.add_owl_scout(os_scout)
        else:
            self.owl_scout_controller.add_owl_scout_from_args(*os_scout)
        self.logger.debug("Owl Scout added to environment")

    def plot_owl_path(
        self, owl: str, error: bool = True, individual_os_points: bool = False
    ) -> None:
        """
        Plots all owl data required for evaluation of the owl scout system.
        Includes the owls path, the positions at which hoots where initiated,
        and the triangulated trace.

        Args:
            owl: ID of the owl for which to plot the data
            error: Whether or not to draw the error circles around the triangulated
                hoot points
            individual_os_points: Whether or not to draw the individual triangulated
                positions as obtained by each of the owl scounts in the network as
                the controller stores the averaged position over all owl scouts
        """
        owl_x = [x[0] for x in self.owls[owl].pos_history]
        owl_y = [y[1] for y in self.owls[owl].pos_history]

        hoot_x = [x[0] for x in self.owls[owl].hoot_history]
        hoot_y = [y[1] for y in self.owls[owl].hoot_history]

        trace_x = [x.position[0] for x in self.owl_scout_controller.owl_trace[owl]]
        trace_y = [y.position[1] for y in self.owl_scout_controller.owl_trace[owl]]

        os_x = [x.position[0] for _, x in self.owl_scout_controller.owl_scouts.items()]
        os_y = [y.position[1] for _, y in self.owl_scout_controller.owl_scouts.items()]

        _, ax = plt.subplots()
        ax.scatter(os_x, os_y, color="k", marker="s")
        ax.plot(owl_x, owl_y, color="b")
        ax.scatter(hoot_x, hoot_y, color="g")
        if error:
            self._add_error_circles(ax, owl)
        if individual_os_points:
            self._add_individual_os_points(ax, owl)
        ax.scatter(trace_x, trace_y, color="r", marker="x")
        plt.show()

    def _add_error_circles(self, ax: plt.Axes, owl: str) -> None:
        """Draws the positional error circles around the triangulated positions."""
        for tp in self.owl_scout_controller.owl_trace[owl]:
            circle = plt.Circle(
                (tp.position[0], tp.position[1]),
                tp.pos_error,
                color="r",
                linewidth=1,
                fill=False,
            )
            ax.add_patch(circle)

    def _add_individual_os_points(self, ax: plt.Axes, owl: str) -> None:
        """
        Draws the triangulation points as stored by each of the individual
        owl scout instances in the controllers network.
        """
        for _, os in self.owl_scout_controller.owl_scouts.items():
            os_trace_x = [x.position[0] for x in os.os_trace[owl]]
            os_trace_y = [y.position[1] for y in os.os_trace[owl]]
            ax.scatter(os_trace_x, os_trace_y, marker="1")
