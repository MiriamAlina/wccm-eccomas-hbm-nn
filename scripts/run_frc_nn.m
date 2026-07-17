clear;
close all;
clc;
addpath('src/hbm_nn/');

%% beam properties
% properties of the beam --------------------------------------------------
len         = .710;             % length
height      = .05;              % height in the bending direction
thickness   = .06;              % thickness in the third dimension
E           = 210e9*0.6725;     % Young's modulus
rho         = 7850;             % density
BCs         = 'clamped-free';   % constraints
n_nodes     = 15;              % number of equidistant nodes along length
% properties of the beam --------------------------------------------------
beam = FE_EulerBernoulliBeam(len,height,thickness,E,rho,BCs,n_nodes);

%% nonlinearity
% properties of the nonlinearity ------------------------------------------
% Specify elastic dry friction (Jenkins) element
kt      = 1e7;              % tangential stiffness
muN     = .5*53*4;          % limit friction force
inode   = 14;               % ~6th pocket
D       = 0.005;             % modal damping ratio
Dmodes  = 1:4;              % modal damped modes
dir     = 'trans';          % transverse direction
force   = 80;               % forcing in Newton
add_nonlinear_attachment(beam,inode,dir,'elasticDryFriction',...
    'stiffness',kt,'friction_limit_force',muN,'ishysteretic',1);
% properties of the nonlinearity ------------------------------------------

% vector recovering deflection at nonlinearity location
T_nl = beam.nonlinear_elements{1}.force_direction';

%% evaluate linear modes
% free modes --------------------------------------------------------------
[PHI_free,OM2] = eig(beam.K,beam.M);
om_free = sqrt(diag(OM2));
% Sorting
[om_free,ind]  = sort(om_free); PHI_free = PHI_free(:,ind);
% mass normalization
PHI_free       = repmat(1./diag(PHI_free'*beam.M*PHI_free)',size(PHI_free,1),1).*PHI_free;

% fixed spring modes ------------------------------------------------------
inl = find(T_nl); w = zeros(length(beam.K),1); w(inl) = 1; Kt = kt*(w*w');
[PHI_stick,OM2] = eig(beam.K+Kt,beam.M);
om_stick        = sqrt(diag(OM2));
% Sorting
[om_stick,ind]  = sort(om_stick); PHI_stick = PHI_stick(:,ind);
% mass normalization
PHI_stick       = repmat(1./diag(PHI_stick'*beam.M*PHI_stick)',size(PHI_stick,1),1).*PHI_stick;

%% set modal damping
n       = size(PHI_free,1);  % number of DoF
beam.D  =  beam.M*PHI_stick(:,Dmodes) * diag(2*D*om_stick(Dmodes)) * PHI_stick(:,Dmodes)'*beam.M;
%% add unit forcing at the tip
fnode   = n_nodes;                              % tip node as excitation node
i_ex    = find_coordinate(beam,fnode,'trans');  % excitation (Ex) node under all DoFs
T_ex    = 1:n == i_ex;                          % localize response under all DoFs

add_forcing(beam,fnode,dir,force);

%% forced response analysis with NLvib
% Analysis parameters
imod    = 1;                % mode to be analyzed
H       = 3;               % number of harmonics to be considered
N       = 2^10;             % sample points for the AFT (time discretisation)

analysis = 'FRF';
% Solve and continue w.r.t. Om  
Om_s    = round(om_stick(imod)*.1);   % start frequency
Om_e    = round(om_stick(imod)*1.5);  % end frequency    

% linear system with sticking contact and without contact
Q_stick = @(om,f) (-om^2*beam.M + 1i*om*beam.D + beam.K +Kt   )\(f*T_ex');
Q_free  = @(om,f) (-om^2*beam.M + 1i*om*beam.D + beam.K       )\(f*T_ex');
Fslip1 = 4*muN/pi;   % amplitude of 1st harmonic of Coulomb force %%%%% TODO: check for correctness
Q_slip = @(Om,f) (-Om^2*beam.M + 1i*Om*beam.D + beam.K) \ ...
                 ( f*T_ex' + Fslip1*w );  %%%%% 

% Initial guess (solution of underlying linear system - stick) (base harmonic/H=1)
Q1      = Q_stick(Om_s,force);
y0      = zeros((2*H+1)*length(Q1),1);
y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];

% Continuation parameter
ds      = 5;  % 10                    % nominal step size
Sopt    = struct(   'jac', 'x', ...
                    'flag',1, ...                           % continuation on
                    'stepadapt',0, ...                      % automatically adapt step size
                    'dynamicDscale',1, ...                  % scale residuum dynamically
                    'Dscale',[1e2*ones(size(y0));Om_s],... % residuum scale vector (size of x0 + step size)
                    'dsmin',1e-3, ...                       % minimal allowed step size
                    'dsmax',20, ...                          % maximal allowed step size
                    'stepmax',1e4);                         % maximal number of steps before aborting

[X_HB,Solinfo_HB] = solve_and_continue(y0, ...
                            @(X) hb_residual_nn(X,beam,H,N,analysis),...
                                            Om_s,Om_e,ds,Sopt);
% Interpret solver output
Om_HB   = X_HB(end,:);
Q_HB    = X_HB(1:end-1,:);    
    
% solution in complex representation
q_HB       = reshape(Q_HB,n,[],size(Q_HB,2));
q_HB       = q_HB(:,2:2:end,:)-1i*q_HB(:,3:2:end,:);
q_HB       = [Q_HB(1:n,:); reshape(q_HB,[],size(Q_HB,2))];

% Compute comparable measures: rms-amplitude without static parts
qrms_HB    = sqrt(squeeze(sum(reshape(Q_HB(n+1:end,:),n,[],length(Om_HB)).^2,2)));

%% plot forced response curve
figure;hold on
Qlin_stick = arrayfun(@(om) abs(T_ex*Q_stick(om,force)),Om_HB);
Qlin_free  = arrayfun(@(om) abs(T_ex*Q_free(om,force)) ,Om_HB);
Qlin_slip  = arrayfun(@(om) abs(T_ex*Q_slip(om,force)) ,Om_HB);  %%%%%

plot(Om_HB/om_stick(imod),Qlin_stick/len,'-','Color',[.3 .3 .3],'DisplayName','linear sticking contact')
plot(Om_HB/om_stick(imod),Qlin_slip/len ,'-','Color',[.5 .5 .5],'DisplayName','linear slipping contact')  %%%%%
plot(Om_HB/om_stick(imod),Qlin_free/len ,'-','Color',[.7 .7 .7],'DisplayName','linear free')
plot(Om_HB/om_stick(imod),qrms_HB(i_ex,:)/len,'r.-','DisplayName','nonlinear FRC')
xlabel({'excitation frequency','$\Omega_{\rm exc}/\omega_{\rm eig}$'},'Interpreter','Latex');
ylabel({'response amplitude';'$q_{\rm exc}/l_{\rm beam}$'},'Interpreter','Latex')
legend;grid on;box on;
ylim([0 5e-5])

freq_ratio = Om_HB/om_stick(imod);
Q_stick = Qlin_stick/len;
Q_free  = Qlin_free/len;
Q_slip  = Qlin_slip/len;  %%%%%
Q_nl    = qrms_HB(i_ex,:)/len;

T = table(freq_ratio(:), Q_stick(:), Q_free(:), Q_slip(:), Q_nl(:), Solinfo_HB.NIT(:), Solinfo_HB.FC(:), ...
          'VariableNames', {'freq_ratio','Q_stick','Q_free','Q_slip','Q_nl','NIT','FC'});  %%%%%
writetable(T, sprintf('data/nn_results_force%.0f_kt%.0f_muN%.0f.csv', force, kt, muN));