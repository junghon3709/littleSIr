%matplotlib inline
import numpy as np
import pandas as pd
import sys
import scipy
from scipy import integrate
import matplotlib.pyplot as plt

def SIR_model(y, t, k, b):
  S, I, R= y
  dS = -k*S*I 
  dI = k*S*I - b*I
  dR = b*I
  return ([dS, dI, dR])

class Country():
  def __init__(self, s0, i0, r0):
    self.susceptible = s0
    self.infected = i0
    self.recovered = r0
    self.time_period = {}
    self.time = 0
    self.start = 0
    self.results = 0
    self.mem = {}
    # self.time_ticks = np.linspace(0, 100, 5000)

  def add_time_period(self, time, beta, gamma):
    self.time = self.time + time
    self.time_period.update({self.time: [beta, gamma]})

  def get_time_period(self):
    return self.time_period

  def run_simulation(self):
    results = [] 
    for i in self.time_period.keys():
      self.end = i
      t = np.linspace(self.start,self.end, 50*(self.end-self.start))
      values_at_time = [self.susceptible, self.infected, self.recovered]
      parameters = tuple(self.time_period[i])
      vals = scipy.integrate.odeint(SIR_model, values_at_time, t, args=parameters)
      results.append(vals)
      self.susceptible = vals[-1][0]
      self.infected = vals[-1][1]
      self.recovered = vals[-1][2]
      self.start = self.end
    self.results = np.concatenate(results)
    last_time_period = list(self.get_time_period().keys())[-1]
    time_plots = np.linspace(0,last_time_period, 50*(last_time_period))
    return time_plots, self.results

  def plot_data(self):
    plt.figure(figsize=[12,8])
    t = np.linspace(0,list(self.time_period.keys())[-1], self.results.shape[0])
    plt.plot(t, self.results[:, 0], label="$S(t)$")
    plt.plot(t, self.results[:, 1], label="$I(t)$")
    plt.plot(t, self.results[:, 2], label="$R(t)$")
    plt.grid()
    plt.legend(prop={'size': 18})
    plt.xlabel("Time", size=18)
    plt.ylabel("Proportions", size=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("SIR model", size=18)
    plt.show()

  def return_results(self):
    return self.results
    
  def final_healthy(self, results):
    return self.results[-1][0]

def compare_graph(data, labels):
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
  for i in range(len(data)):
    axes[i].plot(time, data[i][:, 0], label="$S(t)$")
    axes[i].plot(time, data[i][:, 1], label="$I(t)$")
    axes[i].plot(time, data[i][:, 2], label="$R(t)$")
    axes[i].grid()
    axes[i].legend(prop={'size': 18})
    axes[i].tick_params(axis='both', which='major', labelsize=18)
    axes[i].set_title(labels[i], size=30)
  plt.xlabel("Time", size=18)
  plt.ylabel("Proportions", size=18)
  plt.tight_layout()
  plt.show()