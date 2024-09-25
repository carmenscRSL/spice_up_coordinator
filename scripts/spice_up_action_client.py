#! /usr/bin/env python

import rospy
import sys
import actionlib

from cybathlon_bt_msgs.msg import SpiceUpBottlePickAction,SpiceUpBottlePickGoal


def spice_up_action_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (SpiceUpBottlePickAction) to the constructor.
    client = actionlib.SimpleActionClient('spice_up_action_server', SpiceUpBottlePickAction)

    client.wait_for_server()

    # Creates a goal to send to the action server.
    goal = SpiceUpBottlePickGoal()
    goal.activation = True

    # Sends the goal to the action server.
    client.send_goal(goal)
    print("[SpiceUpActionClient] :  Goal sent")

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # A SpiceUpBottlePickActionResult


if __name__ == '__main__':
    try:
        rospy.init_node('spice_up_action_client')
        print("[SpiceUpActionClient] :  Initialized")
        result = spice_up_action_client()
        print("ee_pickup_target: "+str(result.ee_pickup_target))
        print("ee_dropoff_target: "+str(result.ee_dropoff_target))
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)