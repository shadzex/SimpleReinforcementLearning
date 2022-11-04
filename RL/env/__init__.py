import gym

gym.envs.register(id='Insert-v0',
                  entry_point='RL.env.bullet.Insert:Insert')

gym.envs.register(id='InsertSparse-v0',
                  entry_point='RL.env.bullet.Insert:Insert', kwargs={'sparse_reward': True})

gym.envs.register(id='Push-v0',
                  entry_point='RL.env.bullet.Push:Push')

gym.envs.register(id='Place-v0',
                  entry_point='RL.env.bullet.Place:Place')

gym.envs.register(id='Place-v1',
                  entry_point='RL.env.bullet.Place:Place', kwargs={'use_goal' : True})

gym.envs.register(id='PlaceVisual-v0',
                  entry_point='RL.env.bullet.Place:Place', kwargs={'visual_obs' : True})

gym.envs.register(id='PlaceSoftbody-v0',
                  entry_point='RL.env.bullet.Place:Place', kwargs={'softbody' : True})

gym.envs.register(id='PlaceVisualSoftbody-v0',
                  entry_point='RL.env.bullet.Place:Place', kwargs={'visual_obs' : True, 'softbody' : True})

gym.envs.register(id='PickAndPlace-v0',
                  entry_point='RL.env.bullet.PickAndPlace:PickAndPlace')

# For test: deprecated in future
gym.envs.register(id='AntMaze-v0',
                  entry_point='RL.env.external.AntMaze:AntMaze')

gym.envs.register(id='NachiInsertion-v0',
                  entry_point='RL.env.custom.NachiInsertion:NachiInsertion')

gym.envs.register(id='NachiInsertionPixel-v0',
                  entry_point='RL.env.custom.NachiInsertion:NachiInsertion', kwargs={'use_pixel_data': True})

gym.envs.register(id='NachiInsertionSparse-v0',
                  entry_point='RL.env.custom.NachiInsertion:NachiInsertion', kwargs={'sparse_reward': True})

gym.envs.register(id='NachiInsertionPixelSparse-v0',
                  entry_point='RL.env.custom.NachiInsertion:NachiInsertion', kwargs={'use_pixel_data': True, 'sparse_reward': True})