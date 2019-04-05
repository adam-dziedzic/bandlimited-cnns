load officetemp.mat
Fs = 1/(60*30);   % Sample rate is 1 sample every 30 minutes
t = (0:length(temp)-1)/Fs;

helperFrequencyAnalysisPlot2(t/(60*60*24*7),temp,...
  'Time in weeks','Temperature (Fahrenheit)')