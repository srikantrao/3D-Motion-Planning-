import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, heuristic, create_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

GLOBAL_HOME = (-122.40, 37.79, 0.00)

pos = (-122.393915, 37.797110, 10.0)

north, east, down = global_to_local(pos,GLOBAL_HOME)

print(north, east, down)