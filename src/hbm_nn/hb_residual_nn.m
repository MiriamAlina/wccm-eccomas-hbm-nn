%========================================================================
% DESCRIPTION: 
% Matlab function setting up the frequency-domain residual vector 'R' and 
% its derivatives for given frequency 'Om' and vector of harmonics of the 
% generalized coordiantes 'X'. The corresponding time-domain model 
% equation is 
% 
%       System.M * \ddot q + System.D * \dot q + System.K * q + ...
%                   f_nl(q,\dot q) - f_ex(t) = 0.
%========================================================================
% This file is part of NLvib.
% 
% If you use NLvib, please refer to the book:
%   M. Krack, J. Gross: Harmonic Balance for Nonlinear Vibration
%   Problems. Springer, 2019. https://doi.org/10.1007/978-3-030-14023-6.
% 
% COPYRIGHT AND LICENSING: 
% NLvib Version 1.3 Copyright (C) 2020  Malte Krack  
%										(malte.krack@ila.uni-stuttgart.de) 
%                     					Johann Gross 
%										(johann.gross@ila.uni-stuttgart.de)
%                     					University of Stuttgart
% This program comes with ABSOLUTELY NO WARRANTY. 
% NLvib is free software, you can redistribute and/or modify it under the
% GNU General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% For details on license and warranty, see http://www.gnu.org/licenses
% or gpl-3.0.txt.
%========================================================================
function [R,dR,Q] = ...
    hb_residual_nn(X,system,H,N,analysis_type,varargin)
%% Handle input variables depending on the modus

% System matrices
M = system.M;
D = system.D;
K = system.K;

% number of degrees of freedom
n = size(M,1);

% Conversion from sine-cosine to complex-exponential representation
I0 = 1:n; ID = n+(1:H*n);
IC = n+repmat(1:n,1,H)+n*kron(0:2:2*(H-1),ones(1,n)); IS = IC+n;
dX = eye(length(X));
Q = zeros(n*(H+1),1);   dQ = zeros(size(Q,1),size(dX,2));
Q(I0) = X(I0);          dQ(I0,:) = dX(I0,:);
Q(ID) = X(IC)-1i*X(IS); dQ(ID,:) = dX(IC,:)-1i*dX(IS,:);

% Handle analysis type
if nargin<=4 || isempty(analysis_type)
    % Default analysis: frequency response
    analysis_type = 'frf';
end
switch lower(analysis_type)
    case {'frf','frequency response'}
        % Frequency response analysis: X = [Q;Om]
        
        % Excitation 'excitation' is the fundamental harmonic of the
        % external forcing
        Fex1 = system.Fex1;
        % Setup excitation vector
        Fex = zeros(n*(H+1),1);
        h = 1;% Forcing of h-th harmonic
        Fex(h*n+(1:n)) = Fex1;
        Fex_cs = zeros(size(X,1)-1,1);
        Fex_cs(I0) = real(Fex(I0));
        Fex_cs(IC) = real(Fex(ID));
        Fex_cs(IS) = -imag(Fex(ID));
        % dFex = zeros(size(Fex,1),length(X));
        dFex = zeros(size(Fex,1),length(X)-1);
        % dFex_cs = zeros(size(X,1)-1,size(X,1));
        dFex_cs = zeros(size(X,1)-1,size(X,1)-1);
        dFex_cs(I0,:) = real(dFex(I0,:));
        dFex_cs(IC,:) = real(dFex(ID,:)); 
        dFex_cs(IS,:) = -imag(dFex(ID,:));
        
        % Derivative of damping (does not depend on unknowns in the case of
        % frequency response)
        % dD_dalpha = 0*M; dalpha = zeros(1,length(X));
        
        % Excitation frequency
        Om  =  X(end)	;
        dOm = dX(end,:)	;
        
        % Scaling of dynamic force equilibrium
        %if length(varargin)<2 || isempty(varargin{2})
        %    fscl = 1;
        %else
        %    fscl = varargin{2};
        %end
    case {'nma','nonlinear modal analysis'}
        % Nonlinear modal analysis:  X = [Psi;Om;del;log10a]
        
        % Interpret additional input
        inorm = varargin{1};
        
        % Modal mass
        a = exp(log(10)*X(end));
        da = log(10)*exp(log(10)*X(end))*dX(end,:);
        
        % In this case, the harmonics are amplitude-normalized. We thus
        % have to scale them by the amplitude a.
        Psi = Q; dPsi = dQ;
        Q = Psi*a;
        % dQ = dPsi*a + Psi*da;
        
        % Modal frequency
        Om = X(end-2);
        % dOm = dX(end-2,:);
        
        % Modal damping ratio
        del = X(end-1);
        %ddel = dX(end-1,:);
        
        % Extended Periodic Motion concept: artifical negative viscous 
        % mass-proportional damping
        alpha = 2*Om*del;
        %dalpha = 2*dOm*del+2*Om*ddel;
        D = D-alpha*M;
        %dD_dalpha = -M;
        
        % No external forcing
        %Fex = zeros(n*(H+1),1);
        %dFex = zeros(size(Fex,1),length(X));
        
        % Scaling of dynamic force equilibrium
        %if length(varargin)<2 || isempty(varargin{2})
        %    fscl = 1;
        %else
        %    fscl = varargin{2};
        %end
    otherwise
        error(['Unknown analysis type ' analysis.type '.']);
end
%% Computation of the Fourier coefficients of the nonlinear forces and the 
% Jacobian using AFT
%{
[Fnl_AFT,dFnl_AFT] = HB_nonlinear_forces_AFT(Q,dQ,Om,dOm,H,N,...
    system.nonlinear_elements);
Fnl_cs_AFT = zeros(size(X,1)-1,1);
Fnl_cs_AFT(I0) = real(Fnl_AFT(I0));
Fnl_cs_AFT(IC) = real(Fnl_AFT(ID));
Fnl_cs_AFT(IS) = -imag(Fnl_AFT(ID));
dFnl_cs_AFT = zeros(size(X,1)-1,size(X,1));
dFnl_cs_AFT(I0,:) = real(dFnl_AFT(I0,:));
dFnl_cs_AFT(IC,:) = real(dFnl_AFT(ID,:));
dFnl_cs_AFT(IS,:) = -imag(dFnl_AFT(ID,:));
%}

%% Computation of the Fourier coefficients of the nonlinear forces and the
% Jacobian using the Neural Network
NN_id = get_NN_id(analysis_type, varargin{:});
[Fnl_NN,dFnl_NN] = HB_nonlinear_forces_NN(NN_id, X, N, H, system);
Fnl_cs = Fnl_NN;
dFnl_cs = dFnl_NN;

%% Plot displacement X in every step

%{
 persistent hLine1 hFig frameCounter gifFilename
if isempty(hLine1) || ~isvalid(hLine1)
    gifFilename = fullfile(get_debug_output_dir(), 'q_evolution.gif');
    frameCounter = 0;

    hFig = figure(1); clf;
    hFig.Position = [200 200 500 300];  % [x y width height] in Pixeln
    set(gca, 'FontSize', 12, 'LineWidth', 1);
    hold on;
    xlabel('Full Multi-DOF Coefficient Vector');
    ylabel('Amplitude');
    hLine1 = plot(X(1:end-1), 's', 'MarkerFaceColor', '#1D3557', 'MarkerEdgeColor', '#1D3557', 'DisplayName', 'X');
    legend('show', 'Location', 'eastoutside', 'Orientation', 'vertical');
    hold off;
else
    set(hLine1, 'YData', X(1:end-1));
end
drawnow limitrate nocallbacks;
frameCounter = frameCounter + 1;
    frame = getframe(hFig);
    [A, map] = rgb2ind(frame2im(frame), 256);
    if frameCounter == 1
        imwrite(A, map, gifFilename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(A, map, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end 
%}

%% Plot nonlinear force Fnl in every step
%{
persistent hLineA hFigA hLineB gifFilenameA frameCounterA
if isempty(hLineA) || ~isvalid(hLineA)
    gifFilenameA = fullfile(get_debug_output_dir(), 'Fnl_evolution.gif');
    frameCounterA = 0;

    hFigA = figure(1); clf;
    hFigA.Position = [200 200 500 300];  % [x y width height] in Pixeln
    set(gca, 'FontSize', 12, 'LineWidth', 1);
    hold on;
    title('Obtained Nonlinear Force Coefficients');
    xlabel('Full Multi-DOF Coefficient Vector');
    ylabel('Amplitude');
    hLineA = plot(Fnl_cs_AFT, 's', 'MarkerFaceColor', '#1D3557', 'MarkerEdgeColor', '#1D3557', 'DisplayName', 'Fnl\_AFT');
    hLineB = plot(Fnl_NN, 'o', 'MarkerSize', 4, 'MarkerFaceColor', '#00b695', 'MarkerEdgeColor', '#00b695', 'DisplayName', 'Fnl\_NN');
    ylim([-150 150]);
    legend('show', 'Location', 'eastoutside', 'Orientation', 'vertical');
    hold off;
else
    set(hLineA, 'YData', Fnl_cs_AFT);
    set(hLineB, 'YData', Fnl_NN);
end
drawnow limitrate nocallbacks;
frameCounterA = frameCounterA + 1;
    frame = getframe(hFigA);
    [A, map] = rgb2ind(frame2im(frame), 256);
    if frameCounterA == 1
        imwrite(A, map, gifFilenameA, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(A, map, gifFilenameA, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end  
%}
%% Plot Jacobian dFnl matrices in every step

%{
 persistent hFig hAx1 hAx2 hAx3 hImg1 hImg2 hImg3 gifFilename frameCounter
if isempty(hFig) || ~isvalid(hFig)

    gifFilename = fullfile(get_debug_output_dir(), 'dFnl_evolution.gif');
    frameCounter = 0;

    % --- Figure Setup ---
    hFig = figure(1); clf;
    hFig.Position = [200 200 1000 300];   % Größer wegen 3 Plots nebeneinander

    tiledlayout(1, 3, "Padding", "compact", "TileSpacing", "compact");

    % === Plot 1: dFnl_cs_AFT ===
    hAx1 = nexttile;
    hImg1 = imagesc(dFnl_cs_AFT);
    axis(hAx1, 'equal', 'tight');
    title('dFnl\_cs\_AFT');
    colormap(hAx1, sky);
    clim(hAx1, [-10 10]*1e6);
    colorbar;
    set(hAx1, 'FontSize', 12, 'LineWidth', 1);

    % === Plot 2: dFnl_NN ===
    hAx2 = nexttile;
    hImg2 = imagesc(dFnl_NN);
    axis(hAx2, 'equal', 'tight');
    title('dFnl\_NN');
    colormap(hAx2, sky);
    clim(hAx2, [-10 10]*1e6);
    colorbar;
    set(hAx2, 'FontSize', 12, 'LineWidth', 1);

    % === Plot 3: Differenz ===
    hAx3 = nexttile;
    diffMat = dFnl_NN - dFnl_cs_AFT;
    hImg3 = imagesc(diffMat);
    axis(hAx3, 'equal', 'tight');
    title('Difference: NN - AFT');
    colormap(hAx3, sky);
    clim(hAx3, [-10 3]*1e6);
    colorbar;
    set(hAx3, 'FontSize', 12, 'LineWidth', 1);

else
    % Update existing plots
    set(hImg1, 'CData', dFnl_cs_AFT);
    set(hImg2, 'CData', dFnl_NN);
    set(hImg3, 'CData', dFnl_NN - dFnl_cs_AFT);

    clim(hAx1, [-10 10]*1e6);
    clim(hAx2, [-10 10]*1e6);
    clim(hAx3, [-10 3]*1e6);
end

drawnow limitrate nocallbacks;

% --- GIF export ---
frameCounter = frameCounter + 1;
frame = getframe(hFig);
[A, map] = rgb2ind(frame2im(frame), 256);

if frameCounter == 1
    imwrite(A, map, gifFilename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
else
    imwrite(A, map, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
end 
%}


%% Assembly of the residual and the Jacobian in cosine sine representation
HDM = zeros(2*H,1); % harmonic differentiation matrix
HDM(2:2:end) = 1:H;
HDM = ( diag(HDM,1) - diag(HDM,-1) )*Om;

S = kron(HDM^2,M) + kron(HDM^1,D) + kron(HDM^0,K);
R_cs = S*X(1:end-1) + Fnl_cs - Fex_cs;
R = R_cs;

dR_cs = S + dFnl_cs - dFex_cs;
dR = dR_cs;

%% Save condition number of Jacobian and current frequency in every function call
%{
w_aux = system.nonlinear_elements{1}.force_direction(:);
W_aux = kron(eye(2*H+1), w_aux);
J_AFT_full = pinv(W_aux) * dFnl_cs_AFT(:,1:end-1) * pinv(W_aux)';
J_NN_full  = pinv(W_aux) * dFnl_NN * pinv(W_aux)';
idx = [2 3 6 7];
J_AFT_loc = J_AFT_full(idx, idx);
J_NN_loc  = J_NN_full(idx, idx);
% Frobenius norm error between AFT and NN Jacobian
fnorm_error = norm(J_AFT_loc - J_NN_loc, 'fro') / norm(J_AFT_loc, 'fro');
s_AFT = svd(full(J_AFT_loc));
s_NN  = svd(full(J_NN_loc));
cond_AFT = max(s_AFT) / min(s_AFT);
cond_NN  = max(s_NN)  / min(s_NN);
persistent isFirstCall
filename = fullfile(get_debug_output_dir(), ...
    sprintf('cond_Om_force%d_kt%d_muN%d.csv', ...
    system.Fex1(end-1), ...
    system.nonlinear_elements{1}.stiffness, ...
    system.nonlinear_elements{1}.friction_limit_force));
T = table(fnorm_error, min(s_AFT), min(s_NN), cond_AFT, cond_NN, Om, ...
    'VariableNames', {'fnorm_error', 's_min_AFT', 's_min_NN', 'cond_AFT', 'cond_NN', 'Om'});
if isempty(isFirstCall)
    isFirstCall = false;
    writetable(T, filename);
else
    writetable(T, filename, 'WriteMode', 'append', 'WriteVariableNames', false);
end
%}

%% Nonlinear modal analysis (NMA)
if strcmpi(analysis_type,'nma') || ...
        strcmpi(analysis_type,'nonlinear modal analysis')
    % Scale dynamic force equilibrium by modal amplitude
    % NOTE: We first evaluate the derivative, as we then overwrite R!
    dR(1:end-2,:) = dR(1:end-2,:)/a-R(1:end-2)/a^2*da;
    R(1:end-2) = R(1:end-2)/a;
    
    % Amplitude normalization: The mass of the nonlinear mode shape (all
    % harmonics) is enforced to be one.
    R(end-1) = real(Psi'*kron(eye(H+1),M)*Psi-1);
    dR(end-1,:) = real(2*(Psi'*kron(eye(H+1),M))*dPsi);
    
    % Phase normalization: Velocity of coordinate 'inorm' is enforced to be 
    % zero at t=0.
    R(end) = (1:H)*imag(Psi(inorm+(n:n:H*n)));
    dR(end,:) = (1:H)*imag(dPsi(inorm+(n:n:H*n),:));
end
end
%% Computation of the Fourier coefficients of the nonlinear forces and the 
% Jacobian using AFT
function [F,dF] = ...
    HB_nonlinear_forces_AFT(Q,dQ,~,~,H,N,nonlinear_elements)
%% Initialize output
F = zeros(size(Q));
dF = zeros(size(F,1),size(dQ,2));

%% Iterate on nonlinear elements
for nl=1:length(nonlinear_elements)
    % Specify time samples along period
    tau = (0:2*pi/N:2*pi-2*pi/N)';
    
    % If the nonlinear element is hysteretic (e.g. elastic dry friction
    % element), we march 2 periods to reach a stablizized hysteresis.
    if nonlinear_elements{nl}.ishysteretic
        tau = [tau;tau+2*pi];
    end
    
    if nonlinear_elements{nl}.islocal
        % Standard case of a local nonlinearity describing a discrete
        % nonlinear element attached to a mechanical system, where the
        % contribution \Delta f to the global nonlinear force vector f is
        %   \Delta f = w * fnl(qnl,unl)
        % where the force fnl is scalar, acts in the direction w and 
        % depends on
        %   qnl=w'*q, unl = w'*u.
        
        % Determine force direction associated with nonlinear element
        if size(Q,1)==H+1
            w = 1;
        else
            w = nonlinear_elements{nl}.force_direction;
        end
        W = kron(eye(H+1),w);
        
        % Apply inverse discrete Fourier transform
        H_iDFT = exp(1i*tau*(0:H));
        qnl = real(H_iDFT*(W'*Q));
        dqnl = real(H_iDFT*(W'*dQ));
        %% Evaluate nonlinear force in time domain
        switch lower(nonlinear_elements{nl}.type)
            case 'elasticdryfriction'
                % Throw error if the nonlinear element has not been
                % declared as hysteretic
                if ~nonlinear_elements{nl}.ishysteretic
                    error(['Nonlinear elements of type ' ...
                        '<elasticDryFriction> must have property' ...
                        '<ishysteretic> set to 1.']);
                end
                % Set stiffness of elastic dry friction element (aka
                % Jenkins element)
                k = nonlinear_elements{nl}.stiffness;
                fc = nonlinear_elements{nl}.friction_limit_force;
                
                % Initialize force vector and Coulomb slider position
                fnl = zeros(size( qnl ));
                qsl = zeros(size( qnl ));
                dfnl = zeros(length(fnl),size(dQ,2));
                
                % Assume stuck conditions
                dfnl(1,:) = k*dqnl(1,:);
                
                % Iterative computation of force
                for ij = 2:length(fnl)
                    
                    % Predictor step
                    fnl(ij) = k*( qnl(ij) - qsl(ij-1) );
                    
                    % Corrector step
                    if(abs(fnl(ij)) >= fc)
                        fnl(ij) = fc*sign(fnl(ij));
                        qsl(ij) = qnl(ij) - fnl(ij)/k;
                    else
                        qsl(ij) = qsl(ij-1);
                        dfnl(ij,:) = k*(dqnl(ij,:)-dqnl(ij-1,:))+...
                            dfnl(ij-1,:);
                    end
                end
            otherwise
                error(['Unknown nonlinear element ' ...
                    nonlinear_elements{nl}.type '.']);
        end
        %% Forward Discrete Fourier Transform
        
        % Apply FFT
        Fnlc = fft(fnl(end-N+1:end))/N;
        dFnlc = fft(dfnl(end-N+1:end,:))/N;
        
        % Truncate and convert to half-spectrum notation
        Fnl = [real(Fnlc(1));2*Fnlc(2:H+1)];
        dFnl = [real(dFnlc(1,:));2*dFnlc(2:H+1,:)];

        % Store current force into global force vector
        F = F + W*Fnl;
        dF = dF + W*dFnl;
    end
end
end


function NN_id = get_NN_id(analysis_type, varargin)
switch lower(analysis_type)
    case {'frf','frequency response'}
        nn_id_arg_index = 1;
    case {'nma','nonlinear modal analysis'}
        nn_id_arg_index = 2;
    otherwise
        nn_id_arg_index = [];
end

if ~isempty(nn_id_arg_index) && numel(varargin) >= nn_id_arg_index && ...
        ~isempty(varargin{nn_id_arg_index})
    NN_id = varargin{nn_id_arg_index};
    return;
end

persistent cached_NN_id
if isempty(cached_NN_id)
    cached_NN_id = select_model_id();
end
NN_id = cached_NN_id;
end


function pyModule = get_NN_python_module()
persistent cached_pyModule
if isempty(cached_pyModule)
    %% Ordner des aktuellen MATLAB-Skripts
    scriptDir = fileparts(mfilename('fullpath'));

    %% Eine Ebene höher liegt bereits src/
    srcDir = fileparts(scriptDir);

    pythonFile = fullfile(srcDir, ...
        'hbm_nn', 'nn_nonlinearity.py');

    assert(isfile(pythonFile), ...
        "Python-Datei wurde nicht gefunden: %s", pythonFile);

    %% src/ zum Python-Pfad hinzufügen
    if count(py.sys.path, py.str(srcDir)) == 0
        insert(py.sys.path, int32(0), py.str(srcDir));
    end

    %% Weil srcDir bereits auf src/ zeigt: ohne "src."
    moduleName = 'hbm_nn.nn_nonlinearity';

    py.importlib.invalidate_caches();
    cached_pyModule = py.importlib.import_module(moduleName);
end

pyModule = cached_pyModule;
end


function project_root = get_project_root()
current_file_dir = fileparts(mfilename('fullpath'));
src_dir = fileparts(current_file_dir);
project_root = fileparts(src_dir);
end


function debug_output_dir = get_debug_output_dir()
debug_output_dir = fullfile(get_project_root(), 'data');
if ~isfolder(debug_output_dir)
    mkdir(debug_output_dir);
end
end


%% Computation of the Fourier coefficients of the nonlinear forces and the 
% Jacobian using a Neural Network
function [F,dF] = ...
    HB_nonlinear_forces_NN(NN_path, X, ~, H, system)

pyModule = get_NN_python_module();
w = system.nonlinear_elements{1}.force_direction;
W = kron(eye(2*H+1),w);
coeffs = W'*X(1:end-1);

%% Plot Fourier coefficients of displacement in cosine-sine form
% persistent hFig colorList colorIndex
%if isempty(hFig) || ~isvalid(hFig)
%    hFig = figure(1); clf;
%    hold on;
%    title('Fourier coefficients of displacement in cosine-sine form', ...
%        'FontSize', 12, 'FontWeight', 'bold');
%    %xlabel('Coefficient');
%    %ylabel('Amplitude');
%    xticks(1:7);
%    xticklabels({'a_0','a_1','b_1','a_2','b_2','a_3','b_3'});
%    hexColors = {'#FAFAFA','#E1E1E1','#C8C8C8','#969696','#646464','#323232', ...
%                '#E63946','#F1FAEE','#A8DADC','#457B9D','#1D3557','#008b9a','#00b695'};
%    colorList = cellfun(@(c) sscanf(c(2:end),'%2x%2x%2x',[1 3])/255, hexColors, 'UniformOutput', false);
%    colorIndex = 1;
%end
%thisColor = colorList{colorIndex};
%colorIndex = mod(colorIndex, numel(colorList)) + 1;
%plot(coeffs, 'LineWidth', 1.2, 'Color', thisColor, 'DisplayName', 'X_{cs}');
%drawnow limitrate nocallbacks;

%% Prepare input for the Neural Network
a1 = coeffs(2);
b1 = coeffs(3);
a3 = coeffs(6);
b3 = coeffs(7);

% Phase shift input
ac = sqrt(a1^2 + b1^2);
phi = atan2(-b1,a1);

c1 = cos(phi); s1 = sin(phi);
c3 = cos(3*phi); s3 = sin(3*phi);

% Scale input
a1p = ac;
a3p = a3*c3 - b3*s3;
b3p = a3*s3 + b3*c3;

k = system.nonlinear_elements{1}.stiffness;
fc = system.nonlinear_elements{1}.friction_limit_force;
scale = k / fc;
nn_input = scale * [a1p, a3p, b3p];

%% Save nn_input in every iteration to file for debugging
%{
idx_force = find(system.Fex1~=0,1,'first');
force = system.Fex1(idx_force);
filename = fullfile(get_debug_output_dir(), ...
    sprintf('frc_inputs_force%d_kt%d_muN%d.csv', force, k, fc));
T = table(nn_input(1), nn_input(2), nn_input(3), ...
    'VariableNames', {'a1p','a3p','b3p'});
persistent isFirstCall
if isempty(isFirstCall)
    isFirstCall = false;
    writetable(T, filename);
else
    writetable(T, filename, 'WriteMode', 'append', 'WriteVariableNames', false);
end
%}

%% Evaluate the Neural Network
nn_result = pyModule.infer_nonlinear_force_and_jacobian(NN_path, nn_input);
nn_output = double(nn_result{1});
J_nn = fc * double(nn_result{2});

% Rescale output
yp = fc * nn_output;

% Derivatives of input transformation
dac_da1 = a1 / ac;
dac_db1 = b1 / ac;
dphi_da1 = b1 / (ac^2);
dphi_db1 = -a1 / (ac^2);

% Derivatives of rotated third harmonic
da3p_dphi = -3*a3*s3 - 3*b3*c3;
db3p_dphi =  3*a3*c3 - 3*b3*s3;

% Complete input Jacobian
J_in = zeros(3,4);
% a1*
J_in(1,1) = scale * dac_da1;
J_in(1,2) = scale * dac_db1;
% a3*
J_in(2,1) = scale * da3p_dphi * dphi_da1;
J_in(2,2) = scale * da3p_dphi * dphi_db1;
J_in(2,3) = scale * c3;
J_in(2,4) = scale * -s3;
% b3*
J_in(3,1) = scale * db3p_dphi * dphi_da1;
J_in(3,2) = scale * db3p_dphi * dphi_db1;
J_in(3,3) = scale * s3;
J_in(3,4) = scale * c3;

% Output rotation
R1 = [c1, s1; -s1, c1];
R3 = [c3, s3; -s3, c3];
R_out = blkdiag(R1, R3);

% Derivative of output w.r.t. phi
A1p = yp(1); B1p = yp(2);
A3p = yp(3); B3p = yp(4);

dy_dphi = zeros(4,1);
dy_dphi(1) = -A1p*s1 + B1p*c1;
dy_dphi(2) = -A1p*c1 - B1p*s1;
dy_dphi(3) = -3*A3p*s3 + 3*B3p*c3;
dy_dphi(4) = -3*A3p*c3 - 3*B3p*s3;

% Complete Jacobian 
J = R_out * J_nn * J_in;
J(:,1) = J(:,1) + dy_dphi * dphi_da1;
J(:,2) = J(:,2) + dy_dphi * dphi_db1;

% Reverse phase shift output
y = R_out * yp';
A1 = y(1); B1 = y(2);
A3 = y(3); B3 = y(4);
% Assemble output
F = zeros(2*H+1,1);
F(2:3) = [A1; B1];
F(6:7) = [A3; B3];
F = W*F;

% Full Jacobian of the Neural Network
J_full = zeros(2*H+1,2*H+1);
J_full(2:3,2:3) = J(1:2,1:2);
J_full(6:7,2:3) = J(3:4,1:2);
J_full(2:3,6:7) = J(1:2,3:4);
J_full(6:7,6:7) = J(3:4,3:4);

dF = W*J_full*W';

end
