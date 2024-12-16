results = struct(); % Initialize a structure
intra_variances = []; % Initialize array to store intra-variance for each user

for user = 1:10
    % Load data for the current user
    user_id = sprintf('U%02d', user);
    time_freq_FDay = load(['CW-Data/' user_id '_Acc_TimeD_FreqD_FDay.mat']);
    time_freq_MDay = load(['CW-Data/' user_id '_Acc_TimeD_FreqD_MDay.mat']);
    
    % Combine the datasets for the user
    combined_time_freq = [time_freq_FDay.Acc_TDFD_Feat_Vec; time_freq_MDay.Acc_TDFD_Feat_Vec];

    % Compute descriptive statistics for the combined dataset
    stats.mean_combined_time_freq = mean(combined_time_freq);
    stats.var_combined_time_freq = var(combined_time_freq); % Variance per feature
    stats.std_combined_time_freq = std(combined_time_freq);

    % Compute intra-variance (mean of variances across all features)
    stats.intra_variance = mean(stats.var_combined_time_freq); % Mean variance

    % Save combined statistics for the current user in results structure
    results.(user_id) = stats; % Save all stats for the current user

    % Store intra-variance for plotting later
    intra_variances = [intra_variances; stats.intra_variance];

    % Display descriptive statistics for Time + Frequency Domain (FDay & MDay)
    disp(['Descriptive Statistics for User ' user_id ' - Combined (Time + Frequency Domain):']);
    disp(table(stats.mean_combined_time_freq', stats.var_combined_time_freq', stats.std_combined_time_freq', 'VariableNames', {'Mean', 'Variance', 'StdDev'}));
end

% Display Intra-Variance for Each User
disp('Intra-Variance for Each User:');
disp(array2table(intra_variances, 'VariableNames', {'Intra_Variance'}, 'RowNames', arrayfun(@(x) sprintf('U%02d', x), 1:10, 'UniformOutput', false)));


%% Plot Intra-Variance
figure;
bar(intra_variances);
title('Intra-Variance Across Users');
xlabel('User');
ylabel('Intra-Variance (Mean of Feature Variances)');
xticks(1:10);
xticklabels(arrayfun(@(x) sprintf('U%02d', x), 1:10, 'UniformOutput', false));
grid on;
