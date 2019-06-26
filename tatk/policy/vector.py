# -*- coding: utf-8 -*-
"""Vector Interface"""

class Vector():
    
    def __init__(self):
        pass
    
    def state_vectorize(state):
        """
        vectorize a state
        Args:
            state (tuple): Dialog state
        Returns:
            state_vec (np.array): Dialog state vector
        """
        raise NotImplementedError
        
    def action_vectorize(action):
        """
        vectorize a action
        Args:
            action (tuple): Dialog act
        Returns:
            action_vec (np.array): Dialog act vector
        """
        raise NotImplementedError
        
    def state_devectorize(state_vec):
        """
        recover a state
        Args:
            state_vec (np.array): Dialog state vector
        Returns:
            state (tuple): Dialog state
        """
        raise NotImplementedError
        
    def action_devectorize(action_vec):
        """
        recover a action
        Args:
            action_vec (np.array): Dialog act vector
        Returns:
            action (tuple): Dialog act
        """
        raise NotImplementedError
    