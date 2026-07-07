import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    """
    Takes an array x and repeats it n times along a new dimension
    """
    # add new axis at the front -> row vector inside batch
    row = np.expand_dims(x, axis=0)
    # Creates a batch of n identical copies of x
    batch = np.repeat(row, n, axis=0)
    return batch


def repeated_box(box_space, n):
    """
    Converts a single-step continuous data space (Box) into a multi-step space
    """
    return spaces.Box(
        low=stack_repeated(box_space.low, n),   # stacks lowest possible value row n times 
        high=stack_repeated(box_space.high, n), # stacks highest possible value row n times 
        shape=(n,) + box_space.shape,   # If the old shape was (7,), the new shape is registered as (n, 7)
        dtype=box_space.dtype
    )


def repeated_space(space, n):
    """
    Router that figures out what kind of Gym space it is looking at and applies the repetition correctly
    """
    # if its a box, it hands it directly to repeated_box func
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    # if its a dict, loops through every single item in the dictionary to apply chunking to all type of values
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')


def take_last_n(x, n):
    """
    Slices n last items x
    """
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    """
    Loops through all the keys and values in dictionary x, and takes last n items of the value list.
    """
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method='max'):
    """
    Take a list of numbers and crush them down into a single value based on a method.
    """
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    # blank NumPy array in shape of the obs data
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)

    # take n obs from end of history
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])

    # pads if there are less obs in the history than asked for
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result



class MultiStepWrapper(gym.Wrapper):
    """
    Implements action chunking in simulation.

    args: 
        env: the simulation
        n_obs_steps: The number of past observations the agent gets to see at once.
        n_action_steps: The number of actions the agent will predict in a single chunk.
        max_episode_steps: how long the episode can run before being forcibly ended.
        reward_agg_method: How to combine the rewards from the multiple steps (default is 'max').

    """
    def __init__(
        self,
        env, 
        n_obs_steps, 
        n_action_steps,
        max_episode_steps,
        reward_agg_method='max'
    ):  
        # init underlying env
        super().__init__(env) 

        # space to create batch of actions
        self._action_space = repeated_space(env.action_space, n_action_steps)
        # space to create batch of obs
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)

        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps
    
    def reset(self):
        """Resets the environment using kwargs."""

        # reset env and give back initial obs
        obs = super().reset()

        # create emtpy obs history and place intit obs inside it
        self.obs = deque([obs], maxlen=self.n_obs_steps+1)

        # reset the rest
        self.reward = list()
        self.done = list()
        self.action_history = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        # duplicate init obs to fill empty space of n obs
        obs = self._get_obs(self.n_obs_steps)
        return obs
    
    def step(self, action):

        # action chunk
        for act in action:
            # Before taking a step, check if the epsiode already ended on the last step
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            
            # execute action of action chunk
            observation, reward, done, info = super().step(act)

            self.action_history.append(np.array(act, copy=True))
            self.obs.append(observation)
            self.reward.append(reward)
            # if max episode steps reached, then stop
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        # aggregate
        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info


    
        
    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')
    

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result

    def get_episode_metrics(self):
        base_env = self._get_base_env()
        metrics = dict()
        if hasattr(base_env, "get_episode_metrics"):
            metrics.update(base_env.get_episode_metrics())

        metrics["reward_history"] = list(self.reward)
        metrics["done_history"] = list(self.done)
        metrics["action_history"] = np.asarray(self.action_history, dtype=np.float32)
        metrics["episode_length"] = int(len(self.reward))
        return metrics

    def _get_base_env(self):
        env = self.env
        while hasattr(env, "env"):
            metric_fn = getattr(type(env), "get_episode_metrics", None)
            if callable(metric_fn):
                return env
            env = env.env
        return env

        
