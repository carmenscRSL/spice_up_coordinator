cmake_minimum_required(VERSION 3.0.2)
project(spice_up_coordinator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  geometry_msgs
  sensor_msgs
  message_generation
  actionlib
  actionlib_msgs
  cybathlon_bt_msgs
)

## Find Boost
find_package(Boost REQUIRED COMPONENTS system thread)


add_service_files(
   FILES
   GetSpiceName.srv
   EstimatePose.srv
)

generate_messages(
  DEPENDENCIES std_msgs actionlib_msgs geometry_msgs sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs actionlib roscpp actionlib_msgs
  DEPENDS Boost
  )

include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

catkin_install_python(PROGRAMS 
  scripts/spice_up_action_server.py
  scripts/spice_up_action_client.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/spice_up_coordinator
)