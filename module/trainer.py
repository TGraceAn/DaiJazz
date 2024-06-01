from v1 import DaiJazz
import torch
import torch.nn as nn

import torch
from torch import nn, optim
from typing import Callable, Tuple
from torch.utils.data import DataLoader
import os
import time

model = DaiJazz()



def single_track(track):
    n = int(0.9*len(data)) # first 90% will be train, rest val
    data = track
    train_data = data[:n]
    val_data = data[n:]


#All tracks is a piece of music with multiple tracks
#This function trains a model for each track

def train_all_tracks():
    models = []
    for i in range(track_heads):
        print(f"Training model for track {i}")
        model = train_1_track()
        models.append(model)
    return models