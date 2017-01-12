#!/usr/bin/env python
import rospy
import roslib
roslib.load_manifest('rospytro')
import pytro.ltraj
import numpy as np

class PathSolver:

  def _init_(self):
    rospy.init_node('traj_test', anonymous=True)
    A = np.eye(2)
    B = np.eye(2)
    self.lt = pytro.ltraj.LTraj2DAvoid(A,B,5)
    self.lt.setInitialState([2.0,3.0])
    self.lt.setTerminalState([8.0,4.0])
    self.lt.add2NormStageCost(np.zeros((2,2)),np.eye(2),Nc=11)
    self.lt.addStatic2DObst(2.5,3.5,1.5,4.5)
    self.lt.addStatic2DObst(5.5,6.5,3.5,7.5)

  
    # or solve by PuLP default built-in solver
    lt.solveByBranchBound()
    return lt

if __name__=="__main__":
    lt = bbTest()


