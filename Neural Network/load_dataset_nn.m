    % Initialize variables
    num_users = 10; % Total number of users
    total_data = []; % To store combined dataset for all users
    
    % Step 1: Combine FDay and MDay for each user and store in workspace
    for user = 1:num_users
        % Create user-specific dataset names
        user_id = sprintf('U%02d', user);
    
        % Load the FDay and MDay datasets for the current user
        time_freq_FDay = load([user_id '_Acc_TimeD_FreqD_FDay.mat']);  
        time_freq_MDay = load([user_id '_Acc_TimeD_FreqD_MDay.mat']);  
    
        % Combine FDay and MDay data (72 x 131)
        user_data = [time_freq_FDay.Acc_TDFD_Feat_Vec; time_freq_MDay.Acc_TDFD_Feat_Vec];
    
        % Save the dataset in the workspace
        assignin('base', sprintf('User%02d_FullData', user), user_data);
    
        % Add the current user's data to the total dataset
        total_data = [total_data; user_data];
    end
    
    % Step 2: Save the total combined dataset (720 x 131) in workspace
    assignin('base', 'TotalData', total_data);
    
    % Step 3: Create labeled datasets for each user (720 x 132)
    for user = 1:num_users
        % Create the label column
        labels = zeros(size(total_data, 1), 1);
        start_idx = (user - 1) * 72 + 1;
        end_idx = user * 72;
        labels(start_idx:end_idx) = 1; % Label the current user's rows as 1
    
        % Combine the total dataset with labels
        labeled_data = [total_data, labels];
    
        % Save the labeled dataset in the workspace
        assignin('base', sprintf('User%02d_LabeledData', user), labeled_data);
    end
