# Cooperative-Search-Gym
OpenAI Gym Environment for Multi-Agent Cooperative Search  

This is a multi-agent search environment designed to run in OpenAI Gym.
It enables cooperative search with multiple agents on a single, hidden map.
The goal is to search as much of the map as possible.
When agents 'meet' within the map, they can share information on what they've explored.
Each agents reward is the percent of the map it has explored.  

This environment is intended to run with a modified version of OpenAI's A2C baseline and MADDPG.
