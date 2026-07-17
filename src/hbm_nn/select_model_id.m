function nn_id = select_model_id(models_dir, model_stem, timeout_seconds)
%SELECT_MODEL_ID Select a trained NN model id from the models directory.
%
% Lists matching model artifacts, lets the user select one by index, and
% defaults to the newest timestamp after Enter or timeout.

if nargin < 1 || isempty(models_dir)
    current_file_dir = fileparts(mfilename('fullpath'));
    src_dir = fileparts(current_file_dir);
    project_root = fileparts(src_dir);
    models_dir = fullfile(project_root, 'models');
end
if nargin < 2 || isempty(model_stem)
    model_stem = 'jenkins_h3';
end
if nargin < 3 || isempty(timeout_seconds)
    timeout_seconds = 15;
end

pattern = ['mlp_' model_stem '_*.pt'];
files = dir(fullfile(models_dir, pattern));
prefix = ['mlp_' model_stem '_'];
suffix = '.pt';

candidates = struct('id', {}, 'path', {}, 'timestamp', {}, 'datenum', {});
for i = 1:numel(files)
    name = files(i).name;
    if ~startsWith(name, prefix) || ~endsWith(name, suffix)
        continue;
    end

    nn_id = name(length(prefix) + 1:end - length(suffix));

    scaler_file = fullfile(models_dir, ...
        ['scaling_params_' model_stem '_' nn_id '.joblib']);
    if ~isfile(scaler_file)
        continue;
    end

    candidates(end + 1).id = nn_id; %#ok<AGROW>
    candidates(end).path = fullfile(files(i).folder, files(i).name);
    candidates(end).timestamp = timestamp_from_id(nn_id);
    candidates(end).datenum = files(i).datenum;
end

if isempty(candidates)
    error('No matching model artifacts found in %s for pattern %s.', ...
        models_dir, pattern);
end

timestamps = [candidates.timestamp];
datenums = [candidates.datenum];
[~, order] = sortrows([timestamps(:), datenums(:)], [-1, -2]);
candidates = candidates(order);
default_index = 1;

label = ['model ' strrep(model_stem, '_', ' ')];
fprintf('\nAvailable %s artifacts:\n', label);
for i = 1:numel(candidates)
    default_marker = '';
    if i == default_index
        default_marker = ' [default]';
    end
    modified = datestr(candidates(i).datenum, 'yyyy-mm-dd HH:MM:SS');
    fprintf('  [%d] %s  (%s, modified %s)%s\n', i, ...
        candidates(i).id, candidates(i).path, modified, default_marker);
end

prompt = sprintf(['Select %s [1-%d] or press Enter for default ' ...
    '(%ds timeout): '], label, numel(candidates), timeout_seconds);
response = timed_input(prompt, timeout_seconds);
response = strtrim(response);

if isempty(response)
    selected_index = default_index;
else
    selected_index = str2double(response);
    if isnan(selected_index) || selected_index < 1 || ...
            selected_index > numel(candidates) || ...
            selected_index ~= floor(selected_index)
        fprintf('Invalid selection ''%s''; using default.\n', response);
        selected_index = default_index;
    end
end

selected = candidates(selected_index);
fprintf('Selected %s: %s (%s)\n', label, selected.id, selected.path);
nn_id = selected.id;
end


function timestamp = timestamp_from_id(nn_id)
match = regexp(nn_id, '\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', ...
    'match', 'once');
if isempty(match)
    timestamp = -inf;
else
    timestamp = datenum(match, 'yyyy-mm-dd_HH-MM-SS');
end
end


function response = timed_input(prompt, timeout_seconds)
if timeout_seconds <= 0
    response = input(prompt, 's');
    return;
end

fprintf('%s', prompt);
response = '';

try
    reader = java.io.BufferedReader( ...
        java.io.InputStreamReader(java.lang.System.in));
    start_time = tic;
    while toc(start_time) < timeout_seconds
        if reader.ready()
            response = char(reader.readLine());
            return;
        end
        pause(0.05);
    end
    fprintf('\nSelection timed out; using default.\n');
catch
    fprintf('\nTimed input is not available; using default.\n');
end
end
