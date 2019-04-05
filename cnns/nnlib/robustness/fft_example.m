
p = [    0
     -1.5728
     -1.5747
     -1.5772
     -1.5790
     -1.5816
     -1.5852
     -1.5877
     -1.5922
     -1.5976
     -1.6044
     -1.6129
     -1.6269
     -1.6512
     -1.6998
     -1.8621
      1.7252
      1.6124
      1.5930
      1.5916
      1.5708
      1.5708
      1.5708 ];
  
 % disp(p)
 % disp(unwrap(p))
 
Fs = 44100;  % 44.1 kHz
y = audioread('guitartune.wav');
hplayer = audioplayer(y, Fs);
% play(hplayer);

NFFT = length(y);
disp(NFFT)
% disp(0:1/NFFT:1-1/NFFT) - range from 0 to 1-1/NFFT with 1/NFFT increment
Y = fft(y,NFFT);
F = ((0:1/NFFT:1-1/NFFT)*Fs).';  % define the correponding frequency coefficients (from the 2pi standard ones)
% disp(F);

magnitudeY = abs(Y);        % Magnitude of the FFT
phaseY = unwrap(angle(Y));  % Phase of the FFT

helperFrequencyAnalysisPlot1(F,magnitudeY,phaseY,NFFT);
% pause;

y1 = ifft(Y,NFFT,'symmetric');
norm(y-y1)

hplayer2 = audioplayer(y1, Fs);
% play(hplayer2);

Ylp = Y;
% Ylp(F>=1000 & F<=Fs-1000) = 0;
Ylp(F>=1000) = 0;

helperFrequencyAnalysisPlot1(F,abs(Ylp),unwrap(angle(Ylp)),NFFT)
% 'Frequency components above 1 kHz have been zeroed')

ylp = ifft(Ylp,'symmetric');
hplayer3 = audioplayer(ylp, Fs);
% play(hplayer3);

% Take the magnitude of each FFT component of the signal
Yzp = abs(Y);
helperFrequencyAnalysisPlot1(F,abs(Yzp),unwrap(angle(Yzp)),NFFT)
% 'Phase has been set to zero'

yzp = ifft(Yzp,'symmetric');
hplayer = audioplayer(yzp, Fs);
play(hplayer);
 
 