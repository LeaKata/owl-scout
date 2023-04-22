from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from scs.owls import Hoot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    """A signal as recorded by a microphone.

    Contains the signal as well as the metadata required to triangulate a direction
    based on a pair of Signals.

    Attributes:
        timestamp: The time at which the signal was recorded
        signal_id: ID of the hoot, to simulate the specific waveform of the hoot
        owl_id: ID of the owl from which the hoot originates (can be unique or type
            based, depending on the identification posseblities)
        microphone: The ID of the microphone that recorded the signal
        microphone_position: The position of the microphone that recorded the
            signal in the environment grid.
    """

    timestamp: datetime
    signal_id: str
    owl_id: str
    microphone: str
    microphone_position: np.ndarray


@dataclass
class MatchedSignals:
    """Collects all signals required for triangulation of a hoot origin.

    Collects the recorded signals of the microphones in all four directions
    [north, east, south, west] with the orientation being relative to the Owl Scout
    module. Checks whether or not all required signals, for triangulation of the
    origin, have been collected when adding a new signal.

    Attributes:
        signal_id: ID of the hoot, to simulate the specific waveform of the hoot
        owl_id: ID of the owl from which the hoot originates (can be unique or type
            based, depending on the identification posseblities)
        mics: A dictionary that holds the signals recorded from each microphone
        complete: Indicates whether or not all signals required for triangulation
            have been collected
    """

    signal_id: str
    owl_id: str
    mics: dict[str, None | Signal] = field(
        default_factory=lambda: {
            "mn": None,
            "me": None,
            "ms": None,
            "mw": None,
        }
    )
    complete: bool = False

    def add_signal(self, signal: Signal, mic: str) -> bool:
        """
        Adds a signal to the respective microphone and returns True if all
        required signals for triangulation have been collected.

        Args:
            signal: The Signal which to add to the collection
            mic: The directional position of the microphone
        """
        self.mics[mic] = signal
        logger.debug(
            f"Collected signal '{signal.signal_id}' for '{self.signal_id}'-match; "
            f"collected by microphone '{mic}'"
        )
        if not [signal for _, signal in self.mics.items() if signal is None]:
            self.complete = True
        return self.complete


@dataclass(frozen=True)
class HootEvent:
    """A localized Hoot origin.

    Stores a triangulated hoot event by storing the position, the time, and the
    owl of the hoot.

    Attributes:
        owl_id: ID of the owl from which the hoot originates (can be unique or type
            based, depending on the identification posseblities)
        timestamp: The estimated time at which the hoot was initiated
        position: The triangulated position vector in meters of the origin
        pos_error: The triangulated position accuracy error
    """

    owl_id: str
    timestamp: datetime
    position: np.ndarray
    pos_error: float = 0.0


class Microphone:
    """A simulated microphone.

    A microphone to be used on a Owl Scout module which records signals
    and assigns them the recorded timestamp.

    Attributes:
        m_id: The unique ID of the microphone
        position: The position vector of the microphone in meters from the
                origin of the environment.
        recorded_last_hoot: A set that contains all recorded hoot IDs to
                make sure no hoot is recorded twice

    TODO: Replace hoot sound front ring instead of radius which is a more
    realistic simulation from the microphones point of view and removes the
    need for the recorded_last_hoot history.
    """

    def __init__(self, m_id: str, position: np.ndarray) -> None:
        """
        Initializes the Microphone with args.

        Args:
            m_id: The unique ID of the microphone
            position: The position vector of the microphone in meters from the
                origin of the environment.
            recodred_last_hoot:
        """
        self.id: str = m_id
        self.position: np.ndarray = position
        self.recorded_last_hoot: set[str] = set()
        logger.debug(
            f"Microhpne '{self.id}' initialized at position {str(self.position)}"
        )

    def check_event(self, hoot: Hoot, time: datetime) -> None | Signal:
        """
        Returns a Signal if triggered by a Hoot instance that has not previously
        been recorded by the microphone.

        Args:
            hoot: The hoot instance for which to check whether or not Signal is
                triggered
            time: The timestamp of the current timestep
        """
        if hoot.hoot_id in self.recorded_last_hoot:
            return None
        distance = np.linalg.norm(hoot.origin - self.position)
        if distance < hoot.radius:
            self.recorded_last_hoot.add(hoot.hoot_id)
            # TODO: Temporary accurate time for testing
            distance = np.linalg.norm(hoot.origin - self.position)
            time = hoot.time + timedelta(seconds=distance / 343)
            return Signal(time, hoot.hoot_id, hoot.owl_id, self.id, self.position)
        else:
            return None

    def clear_hoot(self, signal_id: str) -> None:
        """
        Clears signals that are no longer in audible range from the recorded
        signals memory.

        Args:
            signal_id: The ID of the signal which to delete from the memory.
        """
        self.recorded_last_hoot.remove(signal_id)


class OwlScout:
    """An Owl Scout module.

    An individual owl scout module. Contains four microphones in the four directions
    [north, east, south, west] relative to the center of the module. Contains the
    logic of triangulating the position of a hoot signal based on the difference
    in times at which the microphones recorded the event.

    Attributes:
        id: Unique identifier of the module
        position: Position vector of the module in the grid of the environment in
            meters from the origin
        orientation: The orientation of the module in radians
        key_pairs: The keys pairs of microphones that, together, triangulate a
            direction angle of a signal
        microphone_distance: The distance of the microphones in meters from the
            center of the module
        microphones: A dictionary holding the modules microphone instances with
            their relative directions as keys
        angle_correction: Corrects the triangulated angles of each pair of
            microphones to the respective angle in the environments global grid
        mic_pair_centers: The position vectors of the centers of pair of
            microphones as setup to triangulate signal directions
        received_signals: A dictionary which collects the recordings from each
            microphone for each signal
        data_link: A data link which connects the module with a controller
        os_trace: Owl traces triangulated by this individual owl scout instance
    """

    def __init__(
        self,
        os_id: str,
        position: tuple[float, float],
        orientation: float = 0.0,
        data_link: None | dict[str, list[HootEvent]] = None,
    ) -> None:
        """
        Initializes the OwlScout with args.

        Args:
            os_id: Unique identifier of the module
            position: Position vector of the module in the grid of the environment
                in meters from the origin
            orientation: The orientation of the module in radians
            data_link: A data link which connects the module with a controller
        """
        self.id: str = os_id
        self.position: np.ndarray = np.asarray(position)
        self.orientation: float = orientation
        self.key_pairs: dict[str, str] = {
            "mn": "me",
            "me": "ms",
            "ms": "mw",
            "mw": "mn",
        }
        self.microphone_distance: int = np.sqrt(8)
        self.microphones: dict[str, Microphone] = self._add_microphones(4)
        if self.orientation != 0.0:
            self._rotate_mic_positions(self.orientation)
        self.angle_correction: dict[str, float] = self._get_angle_correction(
            orientation
        )
        self.mic_pair_centers: dict[str, np.ndarray] = self._get_mic_pair_centers()
        self.received_signals: dict[str, MatchedSignals] = {}
        self.data_link: None | dict[str, list[HootEvent]] = data_link
        self.os_trace: dict[str, HootEvent] = {}
        logger.debug(
            f"Initialized Owl Scout '{self.id}' at position {str(self.position)} "
            f" with orientation of {str(self.orientation)} radians"
        )

    def _add_microphones(self, spacing: int) -> dict[str, Microphone]:
        """
        Adds the four microphones in each direcion to the module.

        Args:
            spacing: The distance of each microphone from the center of the
                module in meters

        Returns:
            A dictionary containing the microphones at their respective positions
            in the global grid of the environment as well as the key of their
            position relative to the center of the module
        """
        return {
            "mn": Microphone(
                f"{self.id}_mn", np.array([0, spacing / 2]) + self.position
            ),
            "me": Microphone(
                f"{self.id}_me", np.array([spacing / 2, 0]) + self.position
            ),
            "ms": Microphone(
                f"{self.id}_ms", np.array([0, -spacing / 2]) + self.position
            ),
            "mw": Microphone(
                f"{self.id}_mw", np.array([-spacing / 2, 0]) + self.position
            ),
        }

    def _rotate_mic_positions(self, angle: float) -> None:
        """
        Roatates all microphone positions by the passed angle.

        Args:
            angle: Angle at which to rotate all microphones in radians
        """
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        )
        for _, mic in self.microphones.items():
            mic.position = rotation @ mic.position

    def _get_angle_correction(self, angle: float) -> dict[str, float]:
        """
        Corrects the triangulated direction of a signal for each pair of microphones.
        This is required as each pair of microphones obtains a signal angle relative
        to the line that connects both of them but for simplification reasons the
        intersections of lines are calculated relative to the environments global
        grid.

        Args:
            angle: Additional orientation angle for which to correct triangulated
                event directions in radians

        Returns:
            A dictionary containing the microphone pairs and their respective
            angle correction
        """
        return {
            "mn_me": angle + np.pi / 4,
            "me_ms": angle - np.pi / 4,
            "ms_mw": angle + np.pi / 4,
            "mw_mn": angle - np.pi / 4,
        }

    def _get_mic_pair_centers(self) -> dict[str, np.ndarray]:
        """
        Calculates and returns the centers of the connecting lines between the
        microphones of each pair.

        Returns:
            A dictionary containing the center point vectors for each pair of
            microphones
        """
        centers = {}
        for mic_1, mic_2 in self.key_pairs.items():
            centers[f"{mic_1}_{mic_2}"] = (
                self.microphones[mic_1].position + self.microphones[mic_2].position
            ) / 2
        return centers

    def add_data_link(self, data_link: dict[str, list[HootEvent]]) -> None:
        """Adds a controller data link to the module"""
        self.data_link = data_link

    def check_sensors(self, hoots: list[Hoot], time: datetime) -> None:
        """
        Checks whether or not a Signal is recorded for each micriphone and for
        each of the Hoots aspassed in the arguments and processes the recorded
        signals.

        Args:
            hoots: A list of all hoots for which to check the sensors
            time: The time at which the checks are performed
        """
        for hoot in hoots:
            for m_nr, m in self.microphones.items():
                signal = m.check_event(hoot, time)
                if signal:
                    logger.info(
                        f"Owl scout '{self.id}' captured signal '{signal.signal_id}' "
                        f"with microphone '{m.id}' at position {str(m.position)}"
                    )
                    self._process_signal(signal, m_nr)

    def _process_signal(self, signal: Signal, mic: str) -> None:
        """
        Processes a passed signal by creating a signal collection container, if
        one does not yet exist, and adding the signal to its respective container.
        Initializes the triangulation of the signals origin position if all
        required signals have been collected.

        Args:
            signal: The signal to process
            mic: The modules microphone which recorded the signal
        """
        if signal.signal_id not in self.received_signals:
            self.received_signals[signal.signal_id] = MatchedSignals(
                signal.signal_id, signal.owl_id
            )
        if self.received_signals[signal.signal_id].add_signal(signal, mic):
            logger.info(
                f"Signal '{signal.signal_id}' captured by all os '{self.id}' "
                f"microphones; calculating signal origin position"
            )
            signal_pair = self.received_signals.pop(signal.signal_id)
            self._get_position(signal_pair)

    def _get_position(self, signals: MatchedSignals) -> None:
        """
        Obtains the triangulated position of a signal and passes it through the
        data link.

        1. Calculates the signal direction angle from each of the mic pairs
        2. Correcteds the angles to be relative to the environments global grid
        3. Uses the angles of the two signal pairs most suitable for the
            triangulation to obtain the position of the signal.
        4. Passes the signals position and metadata to the data_link

        Args:
            signals: A container holding the signal recorded from each of the
                four microphones
        """
        angles = {}
        use_mics = self._get_quadrant_mics(signals)
        for mic, signal in signals.mics.items():
            mic_pair = f"{mic}_{self.key_pairs[mic]}"
            delta_t = (
                signals.mics[self.key_pairs[mic]].timestamp - signal.timestamp
            ).total_seconds()
            temp = delta_t * 343 / self.microphone_distance
            if temp > 1:
                temp = 1
            elif temp < -1:
                temp = -1
            angle = np.arcsin(temp)
            angles[mic_pair] = angle

        for mic_pair, angle in angles.items():
            if mic_pair in use_mics:
                angles[mic_pair] = angle * use_mics[mic_pair]
            else:
                # As blindspots of use_mics is 'behind' the other mic pairs
                angles[mic_pair] = angle * -1
            # Correct for angle deviation from global coordinates (see: localize_owl)
            angles[mic_pair] += self.angle_correction[mic_pair]
            logger.debug(
                f"OS '{self.id}' microphone pair '{mic_pair}' obtained angle of "
                f"{angle} radians for signal match '{signals.signal_id}'"
            )

        position = self.localize_owl(angles, use_mics)
        self._store_hoot_event(position, signals)

    def _get_quadrant_mics(self, signals: MatchedSignals) -> dict[str, int]:
        """
        Obtaines the signals origin quadrant and returns the best two signal
        pairs to use for triangulation based on that information.

        Args:
            signals: A container holding the signal recorded from each of the
                four microphones

        Returns:
            A dictionary containing the two microphone pairs to use for the
            triangulatio as well as the angle modifier indicating which side of
            the line, connecting each microphone pair, the signals position is
            located.
        """
        if signals.mics["mn"].timestamp < signals.mics["ms"].timestamp:
            north_south = "north"
        else:
            north_south = "south"
        if signals.mics["me"].timestamp < signals.mics["mw"].timestamp:
            east_west = "east"
        else:
            east_west = "west"
        return self._get_mic_pair(north_south, east_west)

    def _get_mic_pair(self, north_south: str, east_west: str) -> dict[str, int]:
        """
        Selects the best microphone pair to use based on the quadrant in which
        the signal originates. The angle modifiers are required as a microphone
        pair does not know from which side of its connecting line a signal
        originates as the perceived world is mirrored along this line. By default
        all angles are calculated in the 'front' facing directions of the array
        which is why some angles have to be inversed to account for signals on the
        side opposite to the 'front face'.

        Args:
            north_south: The signals origin direction in north-south direction
            east_west: The signals origin direction in east-west direction

        Returns:
            A dictionary containing the two microphone pairs to use for the
            triangulatio as well as the angle modifier indicating which side of
            the line, connecting each microphone pair, the signals position is
            located.
        """
        if north_south == "north":
            if east_west == "east":
                return {"mn_me": 1, "ms_mw": -1}
            else:
                return {"mw_mn": 1, "me_ms": -1}
        else:
            if east_west == "west":
                return {"ms_mw": 1, "mn_me": -1}
            else:
                return {"me_ms": 1, "mw_mn": -1}

    def localize_owl(
        self, angles: dict[str, float], use_mics: dict[str, int]
    ) -> np.ndarray:
        """
        Tries to obtain the signals position with the two optimal microphone pairs.
        If the signal lies to close to the pairs blindspot (the line running
        through the centers of both lines connecting each microphone pair) the
        other two microphone pairs are used to triangulate the signals position.
        For those two pairs, the signal originates from behind their 'front faces'
        and the angle is very steep resulting in a high sensitifiy to angle
        deviation.

        TODO: Find good way to incorporate that into the localization error.

        Args:
            angles: A dictionary containing the microphone pairs and their signal
                directions corrected to be relative to the environments global grid
            use_mics: The microphone pairs which to use optimally

        Returns:
            The best triangulated position of the signal
        """
        m_pairs = [m_pair for m_pair in use_mics]
        position = self._get_intersection(
            angles[m_pairs[0]],
            self.mic_pair_centers[m_pairs[0]],
            angles[m_pairs[1]],
            self.mic_pair_centers[m_pairs[1]],
        )
        if np.array_equal(np.zeros(2), position):
            m_pairs = [m_pair for m_pair in angles if m_pair not in use_mics]
            position = self._get_intersection(
                angles[m_pairs[0]],
                self.mic_pair_centers[m_pairs[0]],
                angles[m_pairs[1]],
                self.mic_pair_centers[m_pairs[1]],
            )
        return position

    def _get_intersection(
        self,
        angle_1: float,
        position_1: np.ndarray,
        angle_2: float,
        position_2: np.ndarray,
    ) -> np.ndarray | None:
        """
        Triangulates the position based on the intersection of the two lines as
        obtained from the two angles and origin positions. Returns a zero vector
        if the angle is in a blindspot which can either be a pi/2 angle or if
        the signal originates somewhere on the line running through the centers
        of the two lines connecting each of the microphone pairs.

        Args:
            angle_1: The first angle to use for the triangulation
            position_1: The origin position of the first directional line
            angle_2: The second angle to use for the triangulation
            position_2: The origin position of the second directinal line

        Returns:
            The triangulated position vector in meters from the environemts origin

        TODO: Better way to calculate intersection based on origin and angle
        """
        if (
            abs(angle_1 - np.pi / 2) < 1e-10
            or abs(angle_2 - np.pi / 2) < 1e-10
            or abs(angle_1 - angle_2) < 1e-10
        ):
            # TODO: Better way to handle this case!
            return np.zeros(2)
        tan_1 = np.tan(angle_1)
        tan_2 = np.tan(angle_2)
        d1 = position_1[1] - tan_1 * position_1[0]
        d2 = position_2[1] - tan_2 * position_2[0]
        x = (d1 - d2) / (tan_2 - tan_1)
        y = d1 + tan_1 * x
        return np.asarray([x, y])

    def _store_hoot_event(self, position: np.ndarray, signals: MatchedSignals) -> None:
        """
        Passes the localized hoot event to the controllers data link. Creates an
        entry for the signal of no corresponding entry exists yet.

        Args:
            position: The triangulated position of the signal
            signals: The container holding the recorded signals metadata
        """
        timestamp = self._get_hoot_time(signals.mics, position)
        logger.debug(
            f"OS '{self.id}' estimated hoot origin time to be: {str(timestamp)}"
        )
        hoot_event = HootEvent(
            owl_id=signals.owl_id,
            timestamp=timestamp,
            position=position,
        )
        logger.info(
            f"OS '{self.id}' localized and stored hoot event: {str(hoot_event)}"
        )
        if signals.signal_id in self.data_link:
            self.data_link[signals.signal_id].append(hoot_event)
        else:
            self.data_link[signals.signal_id] = [hoot_event]
        if signals.owl_id in self.os_trace:
            self.os_trace[signals.owl_id].append(hoot_event)
        else:
            self.os_trace[signals.owl_id] = [hoot_event]

    def _get_hoot_time(self, mics: dict[str, Signal], position: np.ndarray) -> datetime:
        """
        Estimates the time of the signals origination and returns the estimate.

        Args:
            mics: A dictionary mapping all modules microphones to their records
                of the signal to obtain the recoding times
            position: The triangulated position of the signal
        Returns:
            The time closest to the time of the signals origin
        """
        timestamp = datetime.now() + timedelta(hours=1)
        m_pos = np.zeros(2)
        for _, mic in mics.items():
            if mic.timestamp < timestamp:
                timestamp = mic.timestamp
                m_pos = mic.microphone_position
        distance = np.linalg.norm(position - m_pos)
        return timestamp - timedelta(seconds=distance / 343)


class OSController:
    """Controls the network of owl scouts.

    Controls multiple instances of owl scouts to cover a larger area or improve
    triangulation accuracy by averaging over positions. Also keeps track of the
    points through time at which the owl was traced.

    Arguments:
        owl_scouts: A list holding all owl scout instances of the network
        events: A dictionary to which the owl scouts pass the triangulated data
        owl_trace: A dictionary containing owls and their respectively triangulated
            paths through the environment
    """

    def __init__(self) -> None:
        """Initializes the controller"""
        self.owl_scouts: dict[str, OwlScout] = {}
        self.events: dict[str, list[HootEvent]] = {}
        self.owl_trace: dict[str, list[HootEvent]] = {}

    def add_owl_scout(self, owl_scout: OwlScout):
        """Adds a passed OwlScout instance to the network and sets its data link"""
        owl_scout.add_data_link(self.events)
        self.owl_scouts[owl_scout.id] = owl_scout
        logger.info(f"OS '{owl_scout.id}' connected to owl scout network")

    def add_owl_scout_from_args(
        self, os_id: str, position: tuple[float, float], orientation: float = 0.0
    ) -> None:
        """
        Creates a new OwlScout instance and adds it to the network.

        Args:
            os_id: Unique identifier of the module
            position: Position vector of the module in the grid of the environment
                in meters from the origin
            orientation: The orientation of the module in radians
        """
        self.owl_scouts[os_id] = OwlScout(os_id, position, orientation, self.events)
        logger.info(f"OS '{os_id}' connected to owl scout network")

    def collect_events(self, hoots: list[Hoot], time: datetime) -> None:
        """
        Collects the triangulated positions of all OwlScouts in the network and
        initializes the position averaging and storing process.

        Args:
            hoots: A list of all hoots to consider for the event collection
            time: The timestamp of the collection
        """
        for _, owl_scout in self.owl_scouts.items():
            owl_scout.check_sensors(hoots, time)
        self.average_and_store_position()

    def average_and_store_position(self) -> None:
        """
        Averages the positions of multiple triangulated positions exist for a
        localized signal to improve the accuracy. Stores the (averaged or not)
        position of all signals for which all modules have provided a triangulation,
        together with respective metadata, in the trace of the corresponding owl
        (or owl type). Clears the finalized signals from the data_link to reduce
        memory usage.
        """
        clear_signals = []
        for signal_id, events in self.events.items():
            if len(events) < len(self.owl_scouts):
                continue
            elif len(self.owl_scouts) == 1:
                self._store_localization(events[0])
                clear_signals.append(signal_id)
                continue
            avg_pos = sum([e.position for e in events]) / len(events)
            error = self._get_pos_error(events, avg_pos)
            logger.debug(
                f"Owl scout network averaged os hoot event positions and "
                f"obtained posituin vector {str(avg_pos)} with an estimated "
                f"position error of {str(np.round(error, 4))}"
            )
            timestamp = self._get_hoot_time(events)
            hoot_event = HootEvent(
                owl_id=events[0].owl_id,
                timestamp=timestamp,
                position=avg_pos,
                pos_error=error,
            )
            logger.info(f"Owl scout network captured hoot event: {str(hoot_event)}")
            self._store_localization(hoot_event)
            clear_signals.append(signal_id)

        for signal_id in clear_signals:
            del self.events[signal_id]

    def _store_localization(self, event: HootEvent) -> None:
        """
        Creates and/or stores the passed event in the corresponding owls trace
        """
        if event.owl_id not in self.owl_trace:
            self.owl_trace[event.owl_id] = [event]
            logger.info(f"Owl scout network encountered new owl: '{str(event.owl_id)}'")
        else:
            self.owl_trace[event.owl_id].append(event)
            logger.info(
                f"Owl scout network added position of trace belonging to owl "
                f"'{str(event.owl_id)}'"
            )

    def _get_pos_error(self, events: list[HootEvent], avg_pos: np.ndarray) -> float:
        """
        Calculates and returns the positional error as the average distance of all
        triangulates positions from the average position.

        Args:
            events: A list containing the triangulated positions from all owl scout
                modules in the network
            avg_pos: The position averaged over all triangulated positions as provided
                by the owl scout modules in the network

        Returns:
            The positional error in form of the average position deviations from
            the averaged position.

        TODO: Find better way to calculate position errors
        """
        return sum([np.linalg.norm(event.position - avg_pos) for event in events]) / 2

    def _get_hoot_time(self, events: list[HootEvent]) -> datetime:
        """
        Gets anr returns the time closest to the time at which the hoot originated.

        Args:
            events: A list containing all metadata of positions triangulated by
                owl scout modules in the network
        """
        timestamp = datetime.now() + timedelta(hours=1)
        for event in events:
            if event.timestamp < timestamp:
                timestamp = event.timestamp
        return timestamp

    def plot_trace(self, owl: str) -> None:
        """
        Plots the trace as prediced by the triangulation.

        Args:
            owl: ID of the owl for which to plot the trace
        """
        trace_x = [x.position[0] for x in self.owl_trace[owl]]
        trace_y = [y.position[1] for y in self.owl_trace[owl]]

        os_x = [x.position[0] for _, x in self.owl_scouts.items()]
        os_y = [y.position[1] for _, y in self.owl_scouts.items()]

        _, ax = plt.subplots()
        ax.scatter(os_x, os_y, color="k", marker="s")
        ax.plot(trace_x, trace_y, color="g")
        for tp in self.owl_trace[owl]:
            circle = plt.Circle(
                (tp.position[0], tp.position[1]),
                tp.pos_error,
                color="r",
                linewidth=1,
                fill=False,
            )
            ax.add_patch(circle)
        ax.scatter(trace_x, trace_y, color="r", marker="x")
        plt.show()
