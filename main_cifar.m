

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
less_cifar_10_train = cifar_10_train(1:5000,:);

num_c = length(label_names);
plotImgCounter = 1;
imgTestIdx = 3;
%%
classifier = cifar_10_1NN();
X_train = double(less_cifar_10_train);
Y_train = double(less_cifar_10_train_labels);
% 
trnmodel = classifier.train(X_train,Y_train);
fprintf('training: %d\n',trnmodel);
% %test
X_test = double(less_cifar_10_test);
Y_test = double(less_cifar_10_test_labels);
% predictWith = cell2mat(less_cifar_10_test_labels.Desc(imgTestIdx));
% fprintf('Predicting with a image of type %s\n',predictWith);
[maxscore, scores, Prediction] = classifier.predict(X_test(imgTestIdx,:),1);
predictedDesc = class{maxscore+1};
fprintf('time to predict: %d desc:%s(%d)\n',Prediction,predictedDesc,maxscore);
fprintf('Correct answer should be %s\n',class{Y_test(imgTestIdx)+1});
%%

% model
Mdl = cifar_10_evaluate(X_train,Y_train);
% rloss = resubLoss(Mdl);
maxscore_matlab = predict(Mdl,X_test(imgTestIdx,:));
predictedDesc = class{maxscore_matlab+1};
fprintf('Using matlab knn(k=1): desc:%s(%d)\n',predictedDesc,maxscore_matlab);
countCorrect = 0;
countCorrectMatlab = 0;
for indTest=1:500      
    maxscore_matlab = predict(Mdl,X_test(indTest,:));   
    if (Y_test(indTest) == maxscore_matlab)
        countCorrectMatlab = countCorrectMatlab + 1;
    end
end

accuracy = (countCorrectMatlab)/500;
fprintf('matlab knn(1): %d\n',accuracy);



%using LDA

MdlLinear = fitcdiscr(X_train,Y_train);
Pred_Y = predict(MdlLinear,X_test);
%                                 SVMModel = fitcsvm(Train_X',Train_Y);
%                                 disp('#######  Testing The SVM Classsifier ##########')
%                                 [Pred_Y]=predict(SVMModel,Test_X');
temp_Acc=100*mean(Pred_Y==Y_test);
fprintf('Using matlab lda:(%d)\n',temp_Acc);

