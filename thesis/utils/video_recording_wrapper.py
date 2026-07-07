from utils.video_recoder import VideoRecorder
import gym
import numpy as np

class VideoRecordingWrapper(gym.Wrapper):
    """
    Compresses and write frames to the hard drive in the background. 
    It captures frames during `step()` based on a specified frequency.

    Note: This wrapper overrides the default behavior of `env.render()`. Calling 
    `render()` will stop the active recording and return the string path to the 
    saved video file, rather than returning an image array.

    Args:
        env (gym.Env): The underlying Gym environment to wrap.
        video_recoder (VideoRecorder): The external recording utility responsible for encoding and writing the video file.
        mode (str, optional): The render mode passed to the underlying environment. 
            Defaults to 'rgb_array'.
        file_path (str, optional): The absolute or relative path where the video mp4 file should be saved. 
            If `None`, recording is entirely disabled.
        steps_per_render (int, optional): The frequency of frame capture. For 
            example, 1 captures every step, 2 captures every other step. Defaults to 1.
        **kwargs: Additional keyword arguments to pass directly to the underlying 
            `env.render()` function.
    """
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder

        self.step_count = 0


    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.has_rendered_frame = False
        self.video_recoder.stop()
        return obs
    

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            # records frame of obs after the action is executed
            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            if frame is None:
                return result
            assert frame.dtype == np.uint8
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)
            # write frame to recorder
            self.video_recoder.write_frame(frame)
            self.has_rendered_frame = True
        return result


    def render(self, mode='rgb_array', **kwargs):
        """
        Returns the recorded video
        """
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        if not self.has_rendered_frame:
            return None
        return self.file_path
