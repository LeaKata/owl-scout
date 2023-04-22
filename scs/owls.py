from __future__ import annotations

import copy
import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class Hoot:
    """Hoot for the simulated owl.

    A simulated hoot for the simulated owl which also contains information
    required to evaluate the performacne of the Owl Scout system.

    Attributes:
        owl_id: ID of the owl from which the hoot originates
        hoot_id: ID of the hoot, to simulate the specific waveform of the hoot
        time: The time at which the hoot was initiated (Just used for calibration
            of the owl scout)
        origin: Position where the hoot was initiated, used to evaluate the owl
            scout performance
        radius: Distance which the hoot sound has propagated from its origin in
            the current timestep in meters
    """

    def __init__(self, id: str, origin: np.ndarray) -> None:
        """
        Initializes the hoot with args.

        Args:
            id: ID of the owl from which the hoot originates
            origin: Position at which the hoot was initiated
        """
        self.owl_id: str = id
        self.hoot_id: str = f"{id}_{str(datetime.now())}"
        self.time: datetime = datetime.now()
        self.origin: np.ndarray = np.asarray(origin)
        self.radius: float = 0.0
        logger.debug(
            f"Initialized hoot '{self.hoot_id}' at position {str(self.origin)}"
        )

    def extend_radius(self, delta_r: float) -> float:
        """
        Extends the radius by the passed distance in meters and returns the new
        radius.
        """
        self.radius += delta_r
        return self.radius


class Owl:
    """Contains the logic of a simulated owl.

    A simulated owl for the Owl Scout simulation environment. Contains the
    logic of the owls movement as well as when it does or does not hoot.
    Spawns in a defined distance from the origin of the environment and randomly
    moves around the environment.

    Attributes:
        id: Identifier of the owl (can be unique or type)
        velocity: Current velocity vector of the owl in m/s
        position: Current position vector of the owl as distance from the origin
            of the environment in meters
        hoot_prob: Probability of the owl hooting during a timestep
        v_max: Highest velocity the owl can reach in m/s
        d_v_max: Highest change in velocity the owl can perform
        pos_history: Records the history of all positions which can be used to
            draw the owls path in the environment
        hoot_history: Records the history of all positions where the owl hooted
            can be used to check the accuracy of the Owl Scout system
    """

    def __init__(self, owl_id: str, spawn_distance: int = 1) -> None:
        """
        Initializes the owl with args.

        Args:
            owl_id: Identifier of the owl (can be unique or type)
            spawn_distance: Distance from the environments origin in meters.
        """
        self.id: str = owl_id
        self.velocity: np.ndarray = np.zeros(2)
        self.position: np.ndarray = self._spawn_position(spawn_distance)
        self.hoot_prob: float = 0.0
        self.v_max: float = 18.1
        self.d_v_max: float = 5.0
        self.pos_history: list[np.ndarray] = []
        self.hoot_history: list[np.ndarray] = []
        logger.debug(
            f"Initialized owl '{self.id}' at position {str(self.position)} "
            f"with velocity vector {str(np.round(self.velocity, 4))}"
        )

    def _spawn_position(self, spawn_distance: int) -> np.ndarray:
        """
        Calculates and returns a random spawn position that matches the passed
        spawn distance in meters from the origin.

        Args:
            spawn_distance: Distance from the origin in meters

        Returns:
            Numpy array conating the spawn position of the owl
        """
        angle = np.random.uniform(0, 360, 1)[0]
        x_pos = spawn_distance * np.cos(angle)
        y_pos = np.sqrt(spawn_distance ** 2 - x_pos ** 2)
        return np.array([x_pos, y_pos])

    def timestep(self, timestep_duration: float) -> None | Hoot:
        """
        Performs one simulation timestep for the owl. Updates the velocity, the
        position, and potentially returns a hoot.

        Args:
            timestep_duration: The duration of one timestep in seconds

        Returns:
            None or a Hoot object depending on whether or not the owl did hoot
            during the timestep
        """
        self.position += self.velocity * timestep_duration
        self.pos_history.append(copy.copy(self.position))
        self._update_velocity()
        self._update_hoot_prob()
        logger.debug(
            f"Owl '{self.id}' new position {str(self.position)} and updated "
            f"velocity vector {str(np.round(self.velocity, 4))}"
        )
        return self._hoot()

    def _update_velocity(self) -> None:
        """
        Updates the velocity of the owl by sampling a random direction in which
        to perform the velocity change as well as a drawing a velocity change
        magnitude. If the directional velocity change would exceed the maximum
        velocity of the owl, the new velocity vector is truncated such that its
        norm does not exceed the maximum velocity.
        """
        d_v_angle = np.random.random_sample() * 2 * np.pi
        d_v_magnitude = np.sqrt(np.random.random_sample()) * self.d_v_max
        d_v_x = d_v_magnitude * np.cos(d_v_angle)
        d_v_y = d_v_magnitude * np.sin(d_v_angle)
        d_v = np.array([d_v_x, d_v_y])
        self.velocity += d_v
        norm_v = np.linalg.norm(self.velocity)
        if norm_v > self.v_max:
            v_angle = np.arccos(self.velocity[0] / norm_v)
            self.velocity[0] = self.v_max * np.cos(v_angle)
            self.velocity[1] = self.v_max * np.sin(v_angle)

    def _update_hoot_prob(self) -> None:
        """Placeholder method - TODO: Find elegant way to make this dynamic"""
        self.hoot_prob = 0.2
        logger.debug(
            f"Owl '{self.id}' hoot probability updated to "
            f"{str(np.round(self.hoot_prob, 2))}"
        )

    def _hoot(self) -> None | Hoot:
        """
        Returns a Hoot from the current position if the owl randomly decides to hoot
        """
        if np.random.choice([True, False], p=[self.hoot_prob, 1 - self.hoot_prob]):
            self.hoot_history.append(copy.copy(self.position))
            logger.debug(
                f"With hoot probability {str(np.round(self.hoot_prob, 2))} owl "
                f"'{self.id}' decided to hoot"
            )
            return Hoot(self.id, self.position)
        else:
            logger.debug(
                f"With hoot probability {str(np.round(self.hoot_prob, 2))} owl "
                f"'{self.id}' decided not to hoot"
            )
            return None
