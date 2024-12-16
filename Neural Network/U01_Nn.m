% Load the dataset
data = evalin('base', 'User01_LabeledData'); % Load "User01_LabeledData" from workspace

% Split dataset into inputs (features) and targets (labels)
inputs = data(:, 1:131);  % Features
targets = categorical(data(:, 132)); % Convert binary labels to categorical

% Normalize inputs
inputs = normalize(inputs);
    
% Define network layers
layers = [
    featureInputLayer(131, 'Name', 'input')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(2, 'Name', 'fc_output') % Two output classes (0 or 1)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 400, ...
    'InitialLearnRate', 0.001, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Split data into training (80%) and testing (20%)
cv = cvpartition(size(inputs, 1), 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

trainInputs = inputs(trainIdx, :);
trainTargets = targets(trainIdx, :);
testInputs = inputs(testIdx, :);
testTargets = targets(testIdx, :);

% Train the neural network  
net = trainNetwork(trainInputs, trainTargets, layers, options);

% Test the network
predictedTargets = classify(net, testInputs);

% Evaluate performance
accuracy = sum(predictedTargets == testTargets) / numel(testTargets) * 100;
fprintf('Test Accuracy with 80:20 split ratio: %.2f%%\n', accuracy);

% Plot confusion matrix
figure;
confusionchart(testTargets, predictedTargets);
