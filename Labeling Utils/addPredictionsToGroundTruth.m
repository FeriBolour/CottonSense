function GT = addPredictionsToGroundTruth(gTruth, masksFolder, newPathDataSource)

%masksFolder = '/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/Training/Base Model/Predictions/TestingSet Masks';
%load('gTruth.mat');
newGT = gTruth;

%% ChangeFilePaths in gTruth
splitted = split(newGT.DataSource.Source{1},'/');
if length(splitted) == 1
    splitted = split(newGT.DataSource.Source{1},'\');
    currentPathDataSource = join(splitted(1:end-1),'\');
else
    currentPathDataSource = join(splitted(1:end-1),'/');
end

%newPathDataSource = "/home/avl/Projects/Cotton Imaging Project/Data/Datasets02272022/7030_images/test_images";
alternativePaths = {[currentPathDataSource newPathDataSource]};
unresolvedPaths = changeFilePaths(newGT,alternativePaths);

if ~isempty(unresolvedPaths)
    disp("unresolvedPaths is not Empty!")
    return
end

dataSource = newGT.DataSource;
labelDefs = newGT.LabelDefinitions;
labelData = newGT.LabelData;

%% LabelData
maskFiles = dir(fullfile(masksFolder,'*.png'));
outputName = strcat(masksFolder, '.mat');

nfiles = length(maskFiles);    % Number of files found

for ii=1:nfiles
    
    currentfilename = maskFiles(ii).name;
    
    name = split(currentfilename,'_');
    name = join(name(1:end-2),'_');
    
    currentimage = imread(fullfile(masksFolder,maskFiles(ii).name));
    newMask = imbinarize(currentimage);
    
    
    
    label = split(currentfilename, '_');
    label = label{end};
    label = split(label, '.');
    label = label{1};
    
    index = strcmp(dataSource.Source, fullfile(newPathDataSource, strcat(name, '.jpg')));
    prevAnn = gTruth.LabelData(index,:);
    exists = checkMask(prevAnn, newMask);
    
    try
        if exists == false
            polygon = mask2poly(newMask);
            vertices = {[polygon.X' polygon.Y']};
            
            switch label
                
                case 'OpenBoll'
                    newAnn = labelData.OpenBoll(index);
                    newAnn{1}(end+1,1) = vertices;
                    labelData.OpenBoll(index) = newAnn;
                    
                case 'ClosedBoll'
                    newAnn = labelData.ClosedBoll(index);
                    newAnn{1}(end+1,1) = vertices;
                    labelData.ClosedBoll(index) = newAnn;
                    
                case 'Flower'
                    newAnn = labelData.Flower(index);
                    newAnn{1}(end+1,1) = vertices;
                    labelData.Flower(index) = newAnn;
                    
                case 'Square'
                    newAnn = labelData.Square(index);
                    newAnn{1}(end+1,1) = vertices;
                    labelData.Square(index) = newAnn;
                    
                otherwise
                    fprintf("Unexpected Class Name for %s in index %d \n", name, find(index));
                    
            end
        end
    end
    
end
%% Construct GroundTruth

GT = groundTruth(dataSource, labelDefs , labelData);

%% Check for existing Mask
function exists = checkMask(prevAnn, newMask)

exists = false;
W = size(newMask,1);
L = size(newMask,2);

for jj = 1:length(prevAnn.OpenBoll{1})
    temp = prevAnn.OpenBoll{1}{jj};
    prevMask = poly2mask(temp(:,1), temp(:,2), W, L);
    logicalAnd = prevMask & newMask;
    if length(find(logicalAnd)) > 15
        exists = true;
        break
    end
end

if exists == false
    for jj = 1:length(prevAnn.ClosedBoll{1})
        temp = prevAnn.ClosedBoll{1}{jj};
        prevMask = poly2mask(temp(:,1), temp(:,2), W, L);
        logicalAnd = prevMask & newMask;
        if length(find(logicalAnd)) > 15
            exists = true;
            break
        end
    end
end

if exists == false
    for jj = 1:length(prevAnn.Flower{1})
        temp = prevAnn.Flower{1}{jj};
        prevMask = poly2mask(temp(:,1), temp(:,2), W, L);
        logicalAnd = prevMask & newMask;
        if length(find(logicalAnd)) > 15
            exists = true;
            break
        end
    end
end

if exists == false
    for jj = 1:length(prevAnn.Square{1})
        temp = prevAnn.Square{1}{jj};
        prevMask = poly2mask(temp(:,1), temp(:,2), W, L);
        logicalAnd = prevMask & newMask;
        if length(find(logicalAnd)) > 15
            exists = true;
            break
        end
    end
end


end

end