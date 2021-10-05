%
% Embry-Riddle Aeronautical University
% MA490 - Capstone Project
% Jose Nicolas Gachancipa

DF = csvread("data\\wavelet_csv\\107309.csv");
[wt, f] = cwt(DF(:, 2), 1000);

% Obtain the final time.
s = size(DF);
final_time = DF(s(1), 1);

% Full figure.
fh = figure;
ah = axes(fh);
colormap(turbo(100))
imagesc(ah, (1:length(wt)).*(final_time/length(wt)), f, abs(wt));
set(gca,'YDir','normal', 'YScale', 'log', 'YTickLabel', [0.1, 1, 10, 100])
xlabel('Time (seconds)')
ylabel('Frequency (Hz)')
title('Magnitude Scalogram')
colorbar
caxis([0 200]);
