%
% Embry-Riddle Aeronautical University
% MA490 - Capstone Project
% Jose Nicolas Gachancipa

DF = csvread("data\\wavelet_csv\\wavelet_4_109.csv");
[wt, f] = cwt(DF(:, 2), 1000);

% Full figure.
fh = figure;
ah = axes(fh);
colormap(turbo(150))
imagesc(ah, (1:length(wt))./1000, log10(f), abs(wt));
set(gca,'YDir','normal')
xlabel('Time (seconds)')
ylabel('Frequency (Hz) log10')
title('Magnitude Scalogram')
colorbar