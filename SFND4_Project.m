% Automotive Engineer Lucas Raimundo
% Udacity - Sensor Fusion NanoDegree Program

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Radar Target Generation and Detection %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Radar Specifications 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz   %
% Max Range = 200m                 %
% Range Resolution = 1 m           %
% Max Velocity = 100 m/s           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%speed of light = 3e8
%% User Defined Range and Velocity of target

% define the target's initial position and velocity. Note : Velocity
% remains contant

R = 110; % car position
v = -20; % car speed
%% FMCW Waveform Generation

% Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the
% FMCW chirp using the requirements above.

% Constant Values
c = 3e8; % speed of light
delta_r = 1; % range resolution
fc= 77e9; % operating carrier frequency of Radar
range_max = 200; % maximum range

% The number of chirps in one sequence. Its ideal to have 2^ value for the
%ease of running the FFT for Doppler Estimation. 
Nd=128;          % #of doppler cells OR #of sent periods % number of chirps
% The number of samples on each chirp. 
Nr=1024;         %for length of time OR # of range cells

%%%%%%%%%%%%%%%
%%%% FCMW  %%%%
%%%%%%%%%%%%%%%

% Wavelenght
lambda = c/fc; 
% Chirp time 
%tm = 5.5*range2time(range_max,c);
tm = 5.5*2*range_max/c; 
% Bandwidth
%bw = range2bw(delta_r,c);
bw = c/2*delta_r; %Bsweep calculation
% Slope
sweep_slope = bw/tm;
% Alpha
alpha = sweep_slope;

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*tm,Nr*Nd); %total time for samples 1x131072
%% FCMW plots

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FMCW WAVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fr_max = range2beat(range_max,sweep_slope,c); % Frequency for the maximum
% range
% v_max = 230*1000/3600; % Maximum speed of a traveling car
% fd_max = speed2dop(2*v_max,lambda); % Maximum Doppler shift 
% fb_max = fr_max+fd_max; % Maximum beat frequency
% 
% fs = max(2*fb_max,bw); % Sample rate
% waveform = phased.FMCWWaveform('SweepTime',tm,'SweepBandwidth',bw,...,
%'SampleRate',fs);
% 
% sig = waveform();
% figure(1)
% subplot(211); plot(0:1/fs:tm-1/fs,real(sig));
% xlabel('Time (s)'); ylabel('Amplitude (v)');
% title('FMCW signal'); axis tight;
% subplot(212); spectrogram(sig,32,16,32,fs,'yaxis');
% title('FMCW signal spectrogram');
%% Signal generation and Moving Target simulation

% Running the radar scenario over the time. 

% Vectorization approach
% https://www.mathworks.com/help/matlab/matlab_prog/vectorization.html
r_t = R + v*t;
td = 2*r_t/c;
Tx = cos(2*pi*(fc*t + (alpha*(t.^2)/2)));
Rx = cos(2*pi*(fc*(t - td) + (alpha*((t - td).^2)/2)));
Mix = Tx .* Rx;
%% RANGE MEASUREMENT

% reshape the vector into Nr*Nd array. Nr and Nd here would also define the
% size of Range and Doppler FFT respectively.
Mix_matrix = reshape(Mix, [Nr, Nd]);

%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
Mix_fft = fft(Mix_matrix,[],1)/Nr;

% Take the absolute value of FFT output
Mix_fft_abs = abs(Mix_fft);

% Output of FFT is double sided signal, but we are interested in only one 
% side of the spectrum.
% Hence we throw out half of the samples.
Mix_fft_abs = Mix_fft_abs(1:Nr/2);

% Maximum value -> vehicle position
Mix_fft_max = max (Mix_fft_abs);

%plotting the range
figure(1)
plot(Mix_fft_abs,'b');
hold on
scatter(R+1,Mix_fft_max,'filled','r');
hold off
title('1D FFT');
xlabel('Range (m)');
ylabel('FFT Magnitude');
axis ([1 201 0 1]);
grid;

%% RANGE DOPPLER RESPONSE

% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map. CFAR will be implemented on the generated RDM

% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.
Mix = reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure(2)
surf(doppler_axis,range_axis,RDM);
colormap default;
colorbar;
title('2D FFT');
xlabel('Doppler velocity (m/s)');
ylabel('Range (m)');

%% CFAR implementation

% Slide Window through the complete Range Doppler Map

% Select the number of Training Cells in both the dimensions.
Tr = 10;
Td = 8;

% Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
Gr = 4;
Gd = 4;

% offset the threshold by SNR value in dB
offset = 6;

% Vectorization approach
% https://www.mathworks.com/help/matlab/matlab_prog/vectorization.html

mask = ones(2 * Tr + 2 * Gr + 1, 2 * Td + 2 * Gd + 1);
centre_coord = [Tr + Gr + 1, Td + Gd + 1];
mask(centre_coord(1) - Gr : centre_coord(1) + Gr,...,
    centre_coord(2) - Gd : centre_coord(2) + Gd) = 0;
mask = mask / sum(mask(:));
mask = mask(end:-1:1, end:-1:1);

% Use 2-D convolution -> conv2
% https://www.mathworks.com/help/matlab/ref/conv2.html
% Convert power to dB to add the offset -> pow2db
% https://www.mathworks.com/help/signal/ref/pow2db.html
% The convolution defines the threshold

threshold = conv2(db2pow(RDM), mask, 'same');
threshold = pow2db(threshold) + offset;

% Any values less than the threshold are 0, else 1
RDM(RDM < threshold) = 0;
RDM(RDM >= threshold) = 1;

% The process above will generate a thresholded block, which is smaller 
% than the Range Doppler Map as the CUT cannot be located at the edges of
% matrix. Hence, few cells will not be thresholded. To keep the map size 
% same set those values to 0.

RDM(1 : Tr + Gr, :) = 0;
RDM(Nr/2 - (Gr + Tr) + 1 : end, :) = 0;
RDM(:, 1 : Td + Gd) = 0;
RDM(:, Nd - (Gd + Td) + 1 : end) = 0;

%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.
figure(3)
surf(doppler_axis,range_axis, RDM);
colormap default;
colorbar;
title('CFAR RESULTS');
xlabel('Doppler velocity (m/s)');
ylabel('Range (m)');
%% References

% https://www.mathworks.com/help/radar/ug/automotive-adaptive-cruise-control-using-fmcw-technology.html 
% https://github.com/rayryeng/Udacity_Sensor_Fusion_Nanodegree/tree/master/SFND_Radar/FinalProject
