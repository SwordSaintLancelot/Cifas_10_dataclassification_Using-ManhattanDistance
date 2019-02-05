
% pathdatain = 'D:\MatlabProgram\cifar_10\cifar-10-batches-mat\cifar-10-batches-mat';
pathdatain = fileparts(mfilename('D:\MatlabProgram\cifar_10\cifar-10-batches-mat\cifar-10-batches-mat'));
addpath(genpath(fullfile(pathdatain, 'cifar-10-batches-mat')));
A = cifar_10_read_data('cifar-10-batches-mat\data_batch_1.mat','cifar-10-batches-mat/batches.meta.mat');
B = cifar_10_read_data('cifar-10-batches-mat\data_batch_2.mat','cifar-10-batches-mat/batches.meta.mat');
C = cifar_10_read_data('cifar-10-batches-mat\data_batch_3.mat','cifar-10-batches-mat/batches.meta.mat');
D = cifar_10_read_data('cifar-10-batches-mat\data_batch_4.mat','cifar-10-batches-mat/batches.meta.mat');
E = cifar_10_read_data('cifar-10-batches-mat\data_batch_5.mat','cifar-10-batches-mat/batches.meta.mat');
load('cifar-10-batches-mat/batches.meta.mat');
cifar_10_test = cifar_10_read_data('cifar-10-batches-mat\test_batch.mat','cifar-10-batches-mat/batches.meta.mat');
cifar_10_train = [A; B; C; D; E];
less_cifar_10_test = cifar_10_test(1:500,:);
less_cifar_10_train = cifar_10_train(1:10000,:);


class=label_names;
num_c = length(class);
plotImgCounter = 1;
imgTestInd = 3;
%%Visualisation

%% Vizualize label_names 
num_label_names = length(label_names);
samples_per_label_names = 5;
plotImgCounter = 1;
for indClass=1:num_label_names
    class_desc = label_names{indClass};
    
    rowsClass = cifar_10_train.Y==indClass-1;
    imgClass = cifar_10_train.Image(rowsClass);
    for indImg=1:samples_per_label_names
        img = imgClass{indImg};
        axis off;
        subplot(num_label_names, samples_per_label_names, plotImgCounter);
        imshow(img);
        plotImgCounter = plotImgCounter + 1;
    end        
end

%% evaluate
cifar_10_1NN = cifar_10_1NN();
X_train = double(cell2mat(less_cifar_10_train.X));
Y_train = double(less_cifar_10_train.Y);
% 
trnmodel = cifar_10_1NN.train(X_train,Y_train);
fprintf('training time: %d\n',trnmodel);
% %test
X_test = double(cell2mat(less_cifar_10_test.X));
Y_test = double(less_cifar_10_test.Y);
predict_image = cell2mat(less_cifar_10_test.Desc(imgTestInd));
fprintf('Predicting with a image of type %s\n',predict_image);
[maxscore, scores, Prediction] = cifar_10_1NN.predict(X_test(imgTestInd,:),1);
predictedDesc = label_names{maxscore+1};
fprintf('prediction: %d desc:%s(%d)\n',Prediction,predictedDesc,maxscore);
fprintf('Correct label %s\n',label_names{Y_test(imgTestInd)+1});
%%

% model
Mdl = cifar_10_evaluate(X_train,Y_train);
% rloss = resubLoss(Mdl);
maxscore_evaluate = predict(Mdl,X_test(imgTestInd,:));
predictedDesc = label_names{maxscore_evaluate+1};
fprintf('Using evaluate function 1NN desc:%s(%d)\n',predictedDesc,maxscore_evaluate);
countCorrect = 0;
countCorrectevaluate = 0;
for indTest=1:500   
    [maxscore, ~, ~] = cifar_10_1NN.predict(X_test(indTest,:),2);
    maxscore_evaluate = predict(Mdl,X_test(indTest,:)); 
    if (Y_test(indTest) == maxscore)
        countCorrect = countCorrect + 1;
    end
    if (Y_test(indTest) == maxscore_evaluate)
        countCorrectevaluate = countCorrectevaluate + 1;
    end
end

accuracy = (countCorrect)/500;
accuracy = accuracy*100;
accuracy_evaluate = (countCorrectevaluate)/500;
accuracy_evaluate = accuracy_evaluate*100;
fprintf('Accuracy my classifier: %d\n',accuracy);
disp(accuracy);
fprintf('Accuracy by evaluate 1NN: %d\n',accuracy_evaluate);
disp(accuracy_evaluate);


%using LDA

MdlLinear = fitcdiscr(X_train,Y_train);
Pred_Y = predict(MdlLinear,X_test);
%                                 SVMModel = fitcsvm(Train_X,Train_Y);
%                                 disp('#######  Testing The SVM Classsifier ##########')
%                                 [Pred_Y]=predict(SVMModel,Test_X);
temp_Acc=100*mean(Pred_Y==Y_test);
fprintf('Using matlab lda:(%d)\n',temp_Acc);

