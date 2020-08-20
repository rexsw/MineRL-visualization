# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:45:00 2020

@author: Scott
"""

import minerl
data = minerl.data.make(
    'MineRLNavigate-v0',
    data_dir='.')
    
for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=32):

        # Print the POV @ the first step of the sequence
#        print(current_state['pov'][0])
#
#        # Print the final reward pf the sequence!
#        print(reward[-1])
        
        print(action[-1])

#        # Check if final (next_state) is terminal.
#        print(done[-1])