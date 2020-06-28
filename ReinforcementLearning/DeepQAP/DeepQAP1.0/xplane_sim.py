## xplane env sim
## Q-Learning Object
## Deep Q Learn AP (Auto Pilot)
## xplane environment for training the RL agent
## xPLANE and Reinforcement Learning - AI/RL based AutoPilot
## Deep Q AP
## Book -> Getting Started with Deep Learning: Programming and Methodologies using Python
## By Ricardo Calix
## www.rcalix.com
## Copyright (c) 2020, Ricardo A. Calix, Ph.D.

##########################################################################

import imp
import numpy as np

##########################################################################

xpc = imp.load_source('xpc','xpc.py')

##########################################################################

pi_d = 0.05
ro_d = 0.05
ru_d = 0.05

actions_n = 6

##########################################################################

drefs_position = ["sim/flightmodel/position/latitude",
                  "sim/flightmodel/position/longitude",
                  "sim/flightmodel/position/elevation",
                  "sim/flightmodel/position/theta",
                  "sim/flightmodel/position/phi",
                  "sim/flightmodel/position/psi",
                  "sim/cockpit/switches/gear_handle_status"]

###########################################################################

drefs_controls = ["sim/cockpit2/controls/yoke_pitch_ratio",
                  "sim/cockpit2/controls/yoke_roll_ratio",
                  "sim/cockpit2/controls/yoke_heading_ratio",
                  "sim/flightmodel/engine/ENGN_thro",
                  "sim/cockpit/switches/gear_handle_status",
                  "sim/flightmodel/controls/flaprqst"]

############################################################################

def send_posi(posi):
    client = xpc.XPlaneConnect()
    client.sendPOSI(posi)
    client.close()
    
##############################################################################

def send_ctrl(ctrl):
    client = xpc.XPlaneConnect()
    client.sendCTRL(ctrl)
    client.close()
 
##############################################################################

def get_posi():
    client = xpc.XPlaneConnect()
    #r = client.getDREFs(drefs_position)
    r = client.getPOSI(0)
    client.close()
    return r
   
##############################################################################

def get_ctrl():
    client = xpc.XPlaneConnect()
    #r = client.getDREFs(drefs_controls)
    r = client.getCTRL(0)
    client.close()
    return r
    
##############################################################################

def reset(posi):
    send_posi(posi)
    new_posi = get_posi()
    return new_posi
 
##############################################################################

def convert_action_to_control(ctrl, action, reward):
    ## action is the selected index of the one-hot encoded vector
    ## ctrl = [-1.0, 0.8, -998, -998, 0, 0] # [pitch, roll, rudder, throttle, gear, flaps]
    ## actions is a one_hot encoded vector
    ## actions = [] ## 2**6 = 64
    ## action is the decimal representation of actions_binary
    ## actions_binary = [pi+, pi-, ro+, ro-, ru+, ru-]
    ## actions_binary = [1, 0, 0, 1, 0, 0] -> (up pitch, left roll, no rudder)
  
    pitch = ctrl[0]
    roll =  ctrl[1]
    rudder = ctrl[2]
    
    actions_binary = np.zeros(actions_n, dtype=int)
    
    actions_binary[action] = 1
    
    pitch = pitch + actions_binary[0] * pi_d - actions_binary[1] * pi_d 
    roll = roll + actions_binary[2] * ro_d - actions_binary[3] * ro_d
    rudder = rudder + actions_binary[4] * ru_d - actions_binary[5] * ru_d
    
    '''
    pitch  = np.clip(pitch, -1.0, 1.0)
    roll   = np.clip(roll, -1.0, 1.0)
    rudder = np.clip(rudder, -1.0, 1.0)
    '''
    
    pitch  = np.clip(pitch, -0.10, 0.10)
    roll   = np.clip(roll, -0.15, 0.15)
    rudder = np.clip(rudder, -0.20, 0.20)
    
    ctrl = [pitch, roll, rudder, -998, 0, 0]
    return ctrl, actions_binary

##############################################################################
 
def update(action, reward):
    old_ctrl = get_ctrl()
    new_ctrl, actions_binary = convert_action_to_control(old_ctrl, action , reward)
    send_ctrl(new_ctrl) ## set control surfaces e.g. pilot the plane
    posi = get_posi()
    return posi, actions_binary, new_ctrl
 
#############################################################################


