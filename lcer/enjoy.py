import argparse
import os
import json
import gym
import torch
import numpy as np
import imageio

from types import SimpleNamespace
from lcer import HER, SAC


def process_inputs(obs, goal, norm_stats):
    obs_norm = (obs - norm_stats["obs_mean"]) / norm_stats["obs_std"]
    goal_norm = (goal - norm_stats["goal_mean"]) / norm_stats["goal_std"]
    return np.concatenate([obs_norm, goal_norm])


def enjoy() -> None:
    # ==================== Arguments ====================
    parser = argparse.ArgumentParser(description="Local Cluster Experience Replay (LCER) Enjoy Script - Arguments")

    parser.add_argument("--checkpoint", help="Checkpoint-file, file path", type=str)
    parser.add_argument("--config", help="Config-file file path", default="", type=str)
    parser.add_argument("--random-actions", help="Use random actions", action="store_true")
    parser.add_argument("--make-gif", help="Whether to make a gif or not", action="store_true")
    parser.add_argument("--add-episodes", help="Whether to add episodes to the gif or not", action="store_true")
    parser.add_argument("--n-enjoy-episodes", help="How many episodes to enjoy", default=10, type=int)

    arguments = parser.parse_args()

    assert os.path.exists(arguments.checkpoint), f'Checkpoint-file does not exist: "{arguments.checkpoint}"'
    
    if len(arguments.config) == 0:
        arguments.config = os.path.join(os.path.dirname(arguments.checkpoint), "config.json")

    assert os.path.exists(arguments.config), f'Config-file does not exist: "{arguments.config}"'

    if arguments.add_episodes:
        assert arguments.make_gif, '"--make-gif" needs to be set to add episodes'

    with open(arguments.config, "rb") as f:
        args = SimpleNamespace(**json.load(f))
    args.cuda = True if torch.cuda.is_available() else False
    args.checkpoint = arguments.checkpoint
    args.config = arguments.config
    args.random_actions = arguments.random_actions
    args.make_gif = arguments.make_gif
    args.add_episodes = arguments.add_episodes
    args.n_enjoy_episodes = arguments.n_enjoy_episodes

    # ==================== Environments ====================
    if "ShadowHandReach" in args.env_name or "ShadowHandBlock" in args.env_name:
        import shadowhand_gym
        
        env = gym.make(args.env_name, render=True)
    else:
        env = gym.make(args.env_name)

    # ==================== Agent ====================
    if "ShadowHandReach" in args.env_name or "ShadowHandBlock" in args.env_name:
        state_dim = env.observation_space["observation"].shape[0]
        goal_dim = env.observation_space["desired_goal"].shape[0]
        agent = HER(state_dim + goal_dim, env.action_space, args)

        if hasattr(args, "her_normalize") and args.her_normalize:
            norm_stats = agent.load_checkpoint(args.checkpoint, load_norm_stats=True)
        else:
            agent.load_checkpoint(args.checkpoint)
    else:
        state_dim = env.observation_space.shape[0]
        agent = SAC(state_dim, env.action_space, args)
        agent.load_checkpoint(args.checkpoint)

    # ==================== GIF ====================
    if args.make_gif:
        images = []
        if args.add_episodes:
            from PIL import Image, ImageDraw, ImageFont

            image_font = ImageFont.truetype("arial.ttf", 25)

        if "ShadowHandReach" not in args.env_name and "ShadowHandBlock" not in args.env_name:
            from gym.wrappers.monitoring.video_recorder import VideoRecorder

            video_path = os.path.join(os.path.dirname(args.checkpoint), "video.mp4")
            video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
    
    # ==================== Enjoy ====================
    for n in range(args.n_enjoy_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            if args.random_actions:
                action = env.action_space.sample()
            else:
                if hasattr(args, "her_normalize") and args.her_normalize:
                    state, goal = state["observation"], state["desired_goal"]
                    state = process_inputs(state, goal, norm_stats=norm_stats)
                action = agent.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)
            
            if args.make_gif: 
                if "ShadowHandReach" in args.env_name or "ShadowHandBlock" in args.env_name:
                    image = env.render(mode="rgb_array")
                else:
                    env.render(mode="human")
                    video_recorder.capture_frame()
                    image = video_recorder.last_frame

                if args.add_episodes:
                    image = Image.fromarray(image)
                    image_draw = ImageDraw.Draw(image)
                    image_draw.text((5, 5), f"Episode: {n+1}", font=image_font, fill=(255, 0, 0))
                    image = np.array(image)
                images.append(image)
            else:
                env.render()
            
            total_reward += reward
            step += 1
            
            if "is_success" in info:
                print(f"Episode: {n+1} \t - \t Step: {step} \t - \t Reward: {reward} \t - \t Success: {info['is_success']}")
            else:
                print(f"Episode: {n+1} \t - \t Step: {step} \t - \t Reward: {reward}")
        print(f"Total reward for episode {n+1}: {total_reward}")
    env.close()

    if args.make_gif:
        if "ShadowHandReach" not in args.env_name and "ShadowHandBlock" not in args.env_name:
            video_recorder.close()
            video_recorder.enabled = False

            if os.path.exists(video_path):
                os.remove(video_path)

            video_meta_path = os.path.join(os.path.dirname(args.checkpoint), "video.meta.json")
            if os.path.exists(video_meta_path):
                os.remove(video_meta_path)

        gif_path = os.path.join(os.path.dirname(args.checkpoint), f"{args.env_name}.gif")
        imageio.mimsave(gif_path, images, duration=0.04)


if __name__ == "__main__":
    enjoy()
