import argparse
import time
import msgpack
from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np

from planning_utils import a_star, heuristic, create_grid, simple_prune, bresenham_prune
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection, debug, target_alt, target_lat, target_lon, prune):
        super().__init__(connection)

        # New variables that are being added
        # Set the debug Variable
        self.debug = debug
        self.prune = prune
        # Set the Target Altitude for the Drone
        self.target_alt = target_alt
        self.target_lat = target_lat
        self.target_lon = target_lon
        if self.debug:
            print("Target Altitude : {}".format(self.target_alt))
            print("Target Latitude: {0:5.2f}".format(self.target_lat))
            print("Target Longitude: {0:5.2f}".format(self.target_lon))

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = self.target_alt
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        with open("colliders.csv", 'r') as f:
            co_ords = f.readline().split(",")
            lat = float(co_ords[0].split(" ")[1])
            lon = float(co_ords[1].split(" ")[2])
            if self.debug:
                print("Reading from file...")
                print("Lat:{0:5.2f}     Lon: {1:5.2f}".format(lat, lon))

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon, lat, 0.0)

        # TODO: retrieve current global position - It should be (0,0,0)
        current_global_position = self.global_position
        if self.debug:
            print("Global Position: Lon :{0:5.2f}, Lat:{1:5.2f}, Alt:{2:5.2f}".format(current_global_position[0],
                    current_global_position[1], current_global_position[2]))

        # TODO: convert to current local position using global_to_local()
        current_local_position = global_to_local(current_global_position, self.global_home)
        if self.debug:
            print("Local Position: North :{0:5.2f}, East:{1:5.2f}, Down:{2:5.2f}".format(current_local_position[0],
                    current_local_position[1], current_local_position[2]))

        if self.debug:
            # Print the GPS Home position
            print("Global Home: Lon :{0:5.2f}, Lat:{1:5.2f}, Alt:{2:5.2f}".format(self.global_home[0],
                    self.global_home[1], self.global_home[2]))
            # Print the Global Position
            print("Global Position: Lon :{0:5.2f}, Lat:{1:5.2f}, Alt:{2:5.2f}".format(self.global_position[0],
                    self.global_position[1], self.global_position[2]))
            # Print the Local Position
            print("Local Position: North :{0:5.2f}, East:{1:5.2f}, Down:{2:5.2f}".format(self.local_position[0],
                    self.local_position[1], self.local_position[2]))

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE, self.debug)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))


        # TODO: convert start position to current position rather than map center
        north = self.local_position[0]
        east = self.local_position[1]
        grid_start = (int(north - north_offset), int(east - east_offset))

        # Adapt to set goal as latitude / longitude position and convert
        target_north, target_east, _ = global_to_local((self.target_lon, self.target_lat, 0.0), self.global_home)
        grid_goal = (int(target_north - north_offset), int( target_east - east_offset))

        grid_goal = (500, 500)  # Make sure the alt is 100 for this case.

        # Run A* to find a path from start to goal
        # Add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)

        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal, self.debug)

        # TODO: prune path to minimize number of waypoints
        if(self.prune == "simple_prune"):
            pruned_path = simple_prune(grid, path, self.debug)
        else:
            pruned_path = bresenham_prune(grid, path, self.debug)

        ########### PLACEHOLDER FOR RANDOM SAMPLING ####################################
        # TODO: Add random sampling methodology
        ##################END OF RANDOM SAMPLING #######################################

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in pruned_path]
        # Set self.waypoints
        self.waypoints = waypoints

        self.send_waypoints()

        if self.debug:
            # Plot the execution path to debug
            plt.imshow(grid, cmap='Greys', origin='lower')
            plt.plot(grid_start[1], grid_start[0], 'x')
            plt.plot(grid_goal[1], grid_goal[0], 'o')
            pruned_path = np.array(pruned_path)
            plt.plot(pruned_path[:, 1], pruned_path[:, 0], 'g')
            plt.scatter(pruned_path[:, 1], pruned_path[:, 0])
            plt.grid(True)
            plt.xlabel('EAST')
            plt.ylabel('NORTH')
            plt.show()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                                    help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--debug', type=str, default="false",
                        help="Prints Additional Debug Information to the Console")
    parser.add_argument('--target_alt', type=int, default=20,
                        help='Sets the target Altitude for the Drone.')
    parser.add_argument('--target_lat', type=float, default=37.797110,
                        help='Final target latitude for the Drone.')
    parser.add_argument('--target_lon', type=float, default=-122.393915,
                        help='Final target longitude for the Drone.')
    parser.add_argument('--prune', type=str, default='simple_prune',
                        help='Prune Mechanism. Can be either simple_prune or bresenham_prune')
    # Parsing all the arguments
    args = parser.parse_args()

    # Parsing debug
    debug = False
    if args.debug.lower() == "true":
        debug = True

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn, debug, args.target_alt, args.target_lat, args.target_lon, args.prune)
    time.sleep(1)

    drone.start()
