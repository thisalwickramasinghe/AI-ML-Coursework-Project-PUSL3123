% Initialize data structure
user_means = []; % To store mean feature vectors for all users
num_users = 10;  % Adjust if the number of users changes

% Collect mean feature vectors for each user
for user = 1:num_users
    user_id = sprintf('U%02d', user);
    
    % Load combined data for the user
    time_freq_FDay = load(['CW-Data/' user_id '_Acc_TimeD_FreqD_FDay.mat']);
    time_freq_MDay = load(['CW-Data/' user_id '_Acc_TimeD_FreqD_MDay.mat']);
    
    % Combine datasets for the user
    combined_time_freq = [time_freq_FDay.Acc_TDFD_Feat_Vec; time_freq_MDay.Acc_TDFD_Feat_Vec];
    
    % Compute and store the mean feature vector
    user_means = [user_means; mean(combined_time_freq)];
    
end

% Compute inter-variance (variance across user means for each feature)
inter_variance = var(user_means, 0, 1); % Variance across rows (users) for each feature

% Display results
disp('Inter-Variance (Feature-wise):');
disp(inter_variance);

%% Assuming 'inter_variance' is already calculated and available in the workspace

% Plot Inter-Variance Across Features
figure;
bar(inter_variance, 'FaceColor', [0.2 0.6 0.8]); % Customize bar color for better visualization
title('Inter-Variance Across Features');
xlabel('Feature Index');
ylabel('Inter-Variance');
grid on;

% Highlight Features with Significant Variance
hold on;
threshold = mean(inter_variance) + std(inter_variance); % Define a threshold for significant variance
significant_features = find(inter_variance > threshold); % Features above threshold
plot(significant_features, inter_variance(significant_features), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5); % Highlight points

% Add a threshold line
yline(threshold, '--r', 'LineWidth', 1.5, 'Label', 'Threshold');

% Add Legend
legend('Feature Variance', 'Significant Features', 'Threshold', 'Location', 'best');

