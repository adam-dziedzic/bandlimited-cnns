function [] = helperFrequencyAnalysisPlot1(F, magnitudeY, phaseY, NFFT)
    dB_mag=mag2db(magnitudeY);
    subplot(2,1,1);plot(F(1:NFFT/2),dB_mag(1:NFFT/2));title('Magnitude response of signal');
    ylabel('Magnitude(dB)');
    subplot(2,1,2);plot(F(1:NFFT/2),phaseY(1:NFFT/2));title('Phase response of signal');
    xlabel('Frequency in kHz')
    ylabel('radians');
end