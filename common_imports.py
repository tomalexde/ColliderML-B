import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl

import torch
from torch.nn import nn
import torch.nn.functional as F

import utils # This is some scripts that will help us with the project. You can find them in the /utils folder.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")