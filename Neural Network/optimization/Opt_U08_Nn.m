% Load the dataset
data = evalin('base', 'User08_LabeledData'); 

% Split dataset into inputs (features) and targets (labels)
inputs = data(:, 1:131); 
targets = categorical(data(:, 132)); 

% Normalize inputs
inputs = normalize(inputs);

% Define network layers
layers = [
    featureInputLayer(131, 'Name', 'input')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.5, 'Name', 'dropout1')   % Dropout with 50% rate
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(32, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(2, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];


% Define network layers for analyzeNetwork 
analyzeLayers = [
    featureInputLayer(131, 'Name', 'input')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(32, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(2, 'Name', 'fc_output') % Two output classes (0 or 1)
];

% Visualize the network architecture
analyzeNetwork(analyzeLayers);

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 500, ...
    'InitialLearnRate', 0.005, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Set up k-fold cross-validation
k = 5; 
cv = cvpartition(size(inputs, 1), 'KFold', k);
accuracies = zeros(k, 1);

for i = 1:k
    % Split data for the current fold
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    trainInputs = inputs(trainIdx, :);
    trainTargets = targets(trainIdx, :);
    testInputs = inputs(testIdx, :);
    testTargets = targets(testIdx, :);

    % Train the neural network
    net = trainNetwork(trainInputs, trainTargets, layers, options);

    % Test the network
    predictedTargets = classify(net, testInputs);

    
    accuracies(i) = sum(predictedTargets == testTargets) / numel(testTargets) * 100;
end

% Train the neural network
net = trainNetwork(trainInputs, trainTargets, layers, options);

% Test the network
predictedTargets = classify(net, testInputs);

% Evaluate performance
accuracy = sum(predictedTargets == testTargets) / numel(testTargets) * 100;
fprintf('Test Accuracy for user 8: %.2f%%\n', accuracy);

% Plot confusion matrix
figure;
confusionchart(testTargets, predictedTargets);
