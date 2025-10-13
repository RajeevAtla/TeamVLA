"""Scripted control policies for TeamVLA benchmark tasks."""

from .drawer_demo import scripted_policy as scripted_drawer_policy, waypoint_library as drawer_waypoints
from .handoff_demo import scripted_policy as scripted_handoff_policy, waypoint_library as handoff_waypoints
from .lift_demo import scripted_policy as scripted_lift_policy, waypoint_library as lift_waypoints

__all__ = [
    "drawer_waypoints",
    "handoff_waypoints",
    "lift_waypoints",
    "scripted_drawer_policy",
    "scripted_handoff_policy",
    "scripted_lift_policy",
]
