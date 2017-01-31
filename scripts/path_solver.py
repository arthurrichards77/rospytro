#!/usr/bin/env python
import rospy
import roslib
roslib.load_manifest('rospytro')
import pytro.ltraj
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PointStamped
from visualization_msgs.msg import Marker

class PathSolver:

  def __init__(self):
    # optimization setup
    A = np.eye(2)
    B = np.eye(2)
    self.lt = pytro.ltraj.LTraj2DAvoid(A,B,5)
    self.lt.add2NormStageCost(np.zeros((2,2)),np.eye(2),Nc=11)
    self.lt.addStatic2DObst(-2.5,-1.5,-2.5,1.5)
    self.lt.addStatic2DObst(1.5,2.5,-1.5,2.5)
    # default boundary conditions
    self.lt.setInitialState([0.0,0.0])
    self.lt.setTerminalState([1.0,1.0])
    # setup ROS node and topics
    rospy.init_node('traj_test', anonymous=True)
    self.path_pub = rospy.Publisher('path',Path,queue_size=1)
    self.marker_pub = rospy.Publisher('marker',Marker,queue_size=10) # needs longer queue as msgs go in quick succession
    self.goal_sub = rospy.Subscriber('move_base_simple/goal',PoseStamped,self.new_goal)
    self.start_sub = rospy.Subscriber('initialpose',PoseWithCovarianceStamped,self.new_start)
    self.click_sub = rospy.Subscriber('clicked_point',PointStamped,self.click_point)

  def solve(self):  
    # solve by PuLP default built-in solver
    self.lt.solveByBranchBound()
    x_vals = [x[self.lt.ind_x].varValue for x in self.lt.var_x]
    y_vals = [x[self.lt.ind_y].varValue for x in self.lt.var_x]
    # convert result to Path message
    output_msg = Path()
    output_msg.header.frame_id = 'world'
    for kk in range(len(x_vals)):
      p = PoseStamped()
      p.pose.position.x = x_vals[kk]
      p.pose.position.y = y_vals[kk]
      output_msg.poses.append(p)
    self.path_pub.publish(output_msg)
    
  def new_goal(self,data):
    # callback for PoseStamped target
    self.lt.changeTermState([data.pose.position.x,data.pose.position.y])
    self.pub_problem()
    self.solve()

  def new_start(self,data):
    # callback for PoseWithCovarianceStamped start point
    self.lt.changeInitState([data.pose.pose.position.x,data.pose.pose.position.y])
    self.pub_problem()

  def click_point(self,data):
    # callback for PointStamped click point, for add or delete obstacle
    del_box = self.lt.deleteObstByPoint([data.point.x, data.point.y])
    if del_box>=0:
        # need to clear the marker
    	self.del_obst_marker(del_box)
    self.pub_problem()

  def del_obst_marker(self,box_id):
    marker_msg = Marker()
    marker_msg.header.frame_id = 'world'
    marker_msg.ns = 'obstacles'
    marker_msg.id = box_id
    marker_msg.action = Marker.DELETE # delete this marker
    self.marker_pub.publish(marker_msg)

  def pub_boxes(self):
    marker_msg = Marker()
    marker_msg.header.frame_id = 'world'
    marker_msg.ns = 'obstacles'
    marker_msg.id = 0
    marker_msg.type = Marker.CUBE
    marker_msg.color.r = 1.0
    marker_msg.color.g = 0.0
    marker_msg.color.b = 0.0
    marker_msg.color.a = 1.0
    marker_msg.scale.z = 1.0
    for bb in self.lt.boxes:
      marker_msg.id += 1
      self.del_obst_marker(marker_msg.id)
      marker_msg.pose.position.x = 0.5*(bb[0]+bb[1])
      marker_msg.pose.position.y = 0.5*(bb[2]+bb[3])
      marker_msg.scale.x = bb[1]-bb[0]
      marker_msg.scale.y = bb[3]-bb[2]
      self.marker_pub.publish(marker_msg)
    self.del_obst_marker(marker_msg.id+1)

  def pub_init(self):
    marker_msg = Marker()
    marker_msg.header.frame_id = 'world'
    marker_msg.ns = 'init'
    marker_msg.id = 0
    marker_msg.type = Marker.SPHERE
    marker_msg.color.r = 0.0
    marker_msg.color.g = 0.0
    marker_msg.color.b = 1.0
    marker_msg.color.a = 1.0
    marker_msg.scale.x = 0.1
    marker_msg.scale.y = 0.1
    marker_msg.scale.z = 0.1
    marker_msg.pose.position.x = self.lt.init_x[self.lt.ind_x]
    marker_msg.pose.position.y = self.lt.init_x[self.lt.ind_y]
    self.marker_pub.publish(marker_msg)

  def pub_term(self):
    marker_msg = Marker()
    marker_msg.header.frame_id = 'world'
    marker_msg.ns = 'term'
    marker_msg.id = 0
    marker_msg.type = Marker.SPHERE
    marker_msg.color.r = 0.0
    marker_msg.color.g = 1.0
    marker_msg.color.b = 0.0
    marker_msg.color.a = 1.0
    marker_msg.scale.x = 0.1
    marker_msg.scale.y = 0.1
    marker_msg.scale.z = 0.1
    marker_msg.pose.position.x = self.lt.term_x[self.lt.ind_x]
    marker_msg.pose.position.y = self.lt.term_x[self.lt.ind_y]
    self.marker_pub.publish(marker_msg)

  def pub_problem(self):
    self.pub_boxes()
    self.pub_init()
    self.pub_term()

if __name__=="__main__":
    ps = PathSolver()
    rospy.spin()
