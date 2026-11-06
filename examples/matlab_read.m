%% Read a flir-csq temperature cube in MATLAB.
%
% Both containers written by the converter load with the same command:
%   * v7.3 (--mat-version 7.3): HDF5-based, no 2 GB limit, streamed on write;
%   * v7   (--mat-version 7):   classic format for small cubes.
% MATLAB sees temperature_C as (height, width, frames), already in the order
% MATLAB uses - no permute needed.

S = load("experiment_001_temperature.mat");
T = S.temperature_C;                 % [height width frames], single, degC
t = S.timestamps_s;                  % [frames x 1], seconds from first frame

fprintf("shape: %d x %d x %d, class %s\n", size(T,1), size(T,2), size(T,3), class(T));
frame = double(T(:,:,1));
fprintf("frame 1: min %.2f  max %.2f  mean %.2f degC\n", ...
        min(frame(:)), max(frame(:)), mean(frame(:)));

%% Metadata (camera, calibration, conversion parameters, warnings)
meta = jsondecode(char(S.metadata_json));
disp(meta.camera.camera_model);
disp(meta.conversion.mode_effective);
disp(meta.conversion.parameters);

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

%% HDF5 alternative (same numbers, no conversion needed)
%   T = h5read("experiment_001_temperature.h5", "/temperature_C");  % [W H frames]
%   T = permute(T, [2 1 3]);                                        % [H W frames]
%   t = h5read("experiment_001_temperature.h5", "/timestamps");
%
% Note: the v7 format cannot exceed 2 GB, so the converter switches to v7.3
% automatically for large cubes (or force it with --mat-version 7.3).
