
clc, clear all, close all

%% Eğitilmiş modelin indirilmesi
% doTraining = true;
% if ~doTraining && ~exist("yolov2ResNet50VehicleExample_19b.mat","file")    
%     disp("Downloading pretrained detector (98 MB)...");
%     pretrainedURL = "https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat";
%     websave("yolov2ResNet50VehicleExample_19b.mat",pretrainedURL);
% end
% unzip vehicleDatasetImages.zip
% data = load("vehicleDatasetGroundTruth.mat");
% vehicleDataset = data.vehicleDataset;

%%

% Detection Yolo

% %İnput Size
imageSize = [224 224 3];

load baretson.mat
numClasses = 1;

anchorBoxes = [
    43 59
    18 22
    23 29
    84 109
];

% Eğitilmiş ağın yüklenmesi
base = resnet50

inputLayer = base.Layers(1)
middle = base.Layers(2:174)
finallayer = base.Layers(175:end)

baseNetwork = [inputLayer
                middle
                finallayer]

% Özellik çıkarma katmanı
featureLayer = 'activation_40_relu';

%% YOLO v2 Nesne Tespit Ağı

lgraph = yolov2Layers(imageSize, numClasses, anchorBoxes, base, featureLayer );

options = trainingOptions ('sgdm',...
          'MiniBatchSize', 128, ...
          'InitialLearnRate', 0.000001, ...
          'MaxEpochs', 25 , ...
          'CheckpointPath', tempdir, ...
          'Shuffle','every-epoch');

% %  % %
vehicleDataset = baretson;

% detector = trainYOLOv2ObjectDetector(baretson,lgraph,options)

matlabroot = 'C:\Users\Rumeysa SAKIN\Desktop\BARETLİ ÇALIŞMA 2\data';

DatasetPath = fullfile(matlabroot);

Data = imageDatastore(DatasetPath, ...
        'IncludeSubfolders', true, 'LabelSource', 'foldernames');

CountLabel = Data.countEachLabel;

trainData1 = Data;

% Eğitim görsellerinin gösterilmesi
% for mm = 1:36
%     a = readimage(trainData1,mm);
% 
%     figure(1),
%     subplot(6,6,mm)
%     imshow(a)
%     title('Training images  ')
% 
% end



%% EĞİTİM

[trainData] = Data;


% Resnet-50 ağını yeniden eğitme

netWidth = 16;
layers = [
     imageInputLayer([224, 224, 3], 'Name', 'input')
    


    convolution2dLayer(3, netWidth, 'Padding', 'same', 'Name', 'convInp')

    batchNormalizationLayer('Name', 'BNInp')

    reluLayer('Name', 'reluInp')

    convolutionalUnit(netWidth, 1, 'S1U1')
    additionLayer(2, 'Name', 'add11')
    reluLayer('Name', 'relu11')
    convolutionalUnit(netWidth, 1, 'S1U2')
    additionLayer(2, 'Name', 'add12')
    reluLayer('Name', 'relu12')

    convolutionalUnit(2*netWidth, 2, 'S2U1')
    additionLayer(2, 'Name', 'add21')
    reluLayer('Name', 'relu21')
    convolutionalUnit(2*netWidth, 1, 'S2U2')
    additionLayer(2, 'Name', 'add22')
    reluLayer('Name', 'relu22')

    convolutionalUnit(4*netWidth, 2, 'S3U1')
    additionLayer(2, 'Name', 'add31')
    reluLayer('Name', 'relu31')
    convolutionalUnit(4*netWidth, 1, 'S3U2')
    additionLayer(2, 'Name', 'add32')
    reluLayer('Name', 'relu32')

    averagePooling2dLayer(8, 'Name', 'globalPool')

    fullyConnectedLayer(2, 'Name', 'fcFinal') 

    softmaxLayer('Name', 'softmax')

    classificationLayer('Name', 'classoutput')
];

lgraph = layerGraph(layers);


lgraph = connectLayers(lgraph, 'reluInp', 'add11/in2');
lgraph = connectLayers(lgraph, 'relu11', 'add12/in2');


skip1 = [
    convolution2dLayer(1, 2*netWidth, 'Stride', 2, 'Name', 'skipConv1')
    batchNormalizationLayer('Name', 'skipBN1')];

lgraph = addLayers(lgraph, skip1);

lgraph = connectLayers(lgraph, 'relu12', 'skipConv1');

lgraph = connectLayers(lgraph, 'skipBN1', 'add21/in2');

lgraph = connectLayers(lgraph, 'relu21', 'add22/in2');

skip2 = [

convolution2dLayer(1, 4*netWidth, 'Stride', 2, 'Name', 'skipConv2')

batchNormalizationLayer('Name', 'skipBN2')];

lgraph = addLayers(lgraph, skip2);

lgraph = connectLayers(lgraph, 'relu22', 'skipConv2');

lgraph = connectLayers(lgraph, 'skipBN2', 'add31/in2');

% Son Katman
lgraph = connectLayers(lgraph, 'relu31', 'add32/in2');


% Eğitim detayları
options = trainingOptions('adam', ...
    'MiniBatchSize', 128, ...
    'MaxEpochs', 30, ... % was 6
    'InitialLearnRate', 0.000001);



% Ağın Eğitilmesi

trainedNet1 = trainNetwork(trainData, lgraph, options);

class = trainedNet1;
% 
save('class1.mat', 'class')

% Train the Network 
% 
% convNet = trainNetwork(trainData, layers, options);

load newtrain.mat
% detector = new;
detector = yolov2ObjectDetector(); % YOLOv2 algılama modelinden bir nesne oluşturuluyor

% 
% %Train1=detector;
% 
%  % Load classi.mat
% 
load class1.mat

convNet = class;

% Input Video

inp = input('Enter File : ');

v = VideoReader(inp)

myVideo = VideoWriter('Output_ACF7.avi');
myVideo.Quality = 50;
myVideo.FrameRate = 15;
k=0;

outclass = []

outclass1 = []
%cd data2
%cd data

for i=1:1:200
    str = int2str(i);
    i


    
% %
videoFrame = read(v,i);
I = videoFrame;

I = imresize (I, [224, 224]);

% Run the detector

[bboxes,scores, label] = detect(detector,I)

% label
%
% %
% % imwrite(videoFrame, str)
%
% if isempty(bbox)
    % detect1 = a1;
%else
%
%detect1 = insertShape(I, 'Rectangle', bboxes);

% % end
% figure(1)
% imshow(detect1)
% % pause(0.5)

%

all = [];
lab = [];
kk = 0;


if isempty (bboxes)
    bboxes = [1 1 1 1];
lab = ['none'];

end

for ii = 1: size(bboxes,1)
    kk = kk+1;

        %se = int2str(kk)
    %str1 = strcat(str,se,'.png')
cr = imcrop(I, bboxes(ii,:));

cr = imresize(cr, [224 224]);

out = classify(convNet, cr);


% % imwrite(ccr, str1)
% out = round(rand(1,1));

if out == 'baretli'
    baretson ='baretli';
    predict = 1

    lab = [lab; {baretson}];

elseif out == 'baretsiz'
    baretson = 'baretsiz';
    predict = 2

    lab = [lab; {baretson}];
else
    lab = [lab ; 'none']
    predict = 3
end

outclass = [outclass ; predict];
outclass1 = [outclass1; lab]

%   if out == 'car';
%        lab = [lab ; {baret}];
%   else
%         lab = [lab; {baret}];
% 
%         lab = [lab; 'none'];
%   end

end


label_str = cell(size(lab,1),1);
conf_val = lab;
for iii = 1:size(lab,1)
    label_str{iii} = conf_val{iii};
end

% set the position for the rectangles as [x y width height]

position = bboxes ;

%  Insert the labels 


RGB = insertObjectAnnotation(I, 'rectangle', position(1:size(position,1), :), label_str(1: size(position,1)) ,...
    'TextBoxOpacity', 0.9, 'FontSize', 12);

% Display the annotated image.
% 
% figure
% imshow(RGB)
% title('Annotated chips');
% 
open(myVideo)
    writeVideo(myVideo,(RGB))
end

close(myVideo)
pause(5)

vv = VideoReader('Output_ACF7.avi')

% cd label 
%    fps = 0

for zz = 1:200
    % Read frames from video 
    im = read(vv,zz);
    im = imresize(im, [700,700]);

    figure(2),imshow(im)

end

% % Predit output 

fullpredict = outclass';

%b% ground truth 

groundtruth = ones(1, length(outclass))

CPredicted = fullpredict;
Labels = groundtruth;
C = confusionmat(Labels, CPredicted);
        OverallAccTra = sum(Labels == CPredicted) / length(CPredicted);
        Acc = zeros(1,1);
        Sens = zeros(1,1);
        Spec = zeros(1,1);
        Prec = zeros(1,1);
        FlSc = zeros(1,1);
        MCC = zeros(1,1);
        FPR = zeros(1,1);
        ERate = zeros(1,1);
        for i = 1:length(1)
            TP = C(i,i);
            TN = sum(C(:)) - sum(C(:,i)) - sum(C(i,:)) + C(i,i);
            FP = sum(C(:,i)) - C(i,i);
            FN = sum(C(i,:)) - C(i,i);
            Acc(i) = (TP + TN) / (TP + TN + FP + FN);
            Sens(i) = TP / (TP + FN);
            Spec(i) = TN / (TN + FP);
            Prec(i) = TP / (TP + FN);
            FlSc(i) = 2*TP / (2*TP + FP + FN);
        end

figure, bar ([(Acc) ; (Sens) ; (Prec) ; (FlSc) ])
xlabel('1-Accuracy  2-Sensitivty    3-Specificty    4-Fl Score')
ylabel('Values')
legend('Proposed', 'Exsisting')
title('Performance')

sprintf('Acc is : %2f ', (Acc))

sprintf('Sens is : %2f', (Sens))

sprintf('Precision is : %2f', (Prec))

