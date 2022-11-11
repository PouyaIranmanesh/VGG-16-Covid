clc;
clear all
close all

imds = imageDatastore('C:\Users\Asus Laptop\Desktop\pooya\Data' ...
     ,'IncludeSubfolders',true,'LabelSource','foldernames');     %loading the dataset
imdsTest = imageDatastore('C:\Users\Asus Laptop\Desktop\pooya\Test' ...
     ,'IncludeSubfolders',true,'LabelSource','foldernames');     %loading the testing dataset
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');     %splitting data for cross validation using holdout method
net = vgg16;
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);  %seperating the last 3 layers to make a customized network
numClasses = numel(categories(imdsTrain.Labels));  %number of categories
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];   %building the network
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);   %augmenting image size to match our network
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);   %augmenting image size to match our network
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);     %augmenting image size to match our network
options = trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',25, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,layers,options);
figure()
[YPred,scores] = classify(netTransfer,augimdsTest);
YValidation = imdsTest.Labels;
accuracy = mean(YPred == YValidation)
plotconfusion(imdsTest.Labels,YPred)
cgt = double(imdsTest.Labels);
cscores = scores;
saveas(gcf,'confusion.fig')
print('confusion','-dpng')
figure()
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(cgt,cscores(:,1),1);
plot(X,Y);
grid
xlabel('False positive rate')
ylabel('True positive rate')
title(['ROC for Classification CNN, AUC=' num2str(AUC)])
print('ROC','-dpng')
saveas(gcf,'ROC.fig')
save('matlab')