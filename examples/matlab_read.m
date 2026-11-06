%% Read a flir-csq temperature cube in MATLAB.
%
% HDF5 (recommended - no conversion needed, works for long recordings):
%   MATLAB reads the cube lazily through the HDF5 library; transpose because
%   h5read returns arrays in (columns, rows, frames) order.

file = "experiment_001_temperature.h5";

info = h5info(file);
disp({info.Datasets.Name});

T = h5read(file, "/temperature_C");      % float32, [width height frames]
T = permute(T, [2 1 3]);                 % -> [height width frames]
t = h5read(file, "/timestamps");         % seconds since the first frame

fprintf("shape: %d x %d x %d\n", size(T, 1), size(T, 2), size(T, 3));
frame = double(T(:, :, 1));
fprintf("frame 1: min %.2f  max %.2f  mean %.2f degC\n", ...
        min(frame(:)), max(frame(:)), mean(frame(:)));

%% Display one frame with a temperature colour bar
figure;
imagesc(frame);
axis image;
colormap hot;
cb = colorbar;
cb.Label.String = "temperature (degC)";
title("Frame 1");

%% Temperature of one pixel versus time
r = round(size(T, 1) / 2);
c = round(size(T, 2) / 2);
figure;
plot(t, squeeze(T(r, c, :)));
xlabel("time (s)");
ylabel("temperature (degC)");
title(sprintf("Pixel (%d, %d)", r, c));
grid on;

%% Mean frame temperature versus time
flat = reshape(T, [], size(T, 3));
figure;
plot(t, mean(flat, 1, "omitnan"));
xlabel("time (s)");
ylabel("mean temperature (degC)");
grid on;

%% If you exported the .mat file instead:
%   S = load("experiment_001_temperature.mat");
%   T = S.temperature_C;          % [frames height width], float32
%   t = S.timestamps_s;
%
% Note: the .mat export needs the whole cube in RAM; for long recordings use
% the HDF5 file, which MATLAB reads directly and without loading everything.
