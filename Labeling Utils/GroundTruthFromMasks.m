function GT = GroundTruthFromMasks(LabelDefs, imageFolder, masksFolder)

%% DataSource
imageFiles = dir(fullfile(imageFolder,'*.jpg'));
DataSource = cell(length(imageFiles), 1);
for n = 1:length(imageFiles)
    DataSource{n} = fullfile(imageFolder,imageFiles(n).name  );
end
%% LabelData
maskFiles = dir(fullfile(masksFolder,'*.png'));
outputName = strcat(masksFolder, '.mat');

nfiles = length(maskFiles);    % Number of files found

LabelData = table('Size', [length(imageFiles) 4], ...
    'VariableTypes', {'cell','cell', 'cell', 'cell'}, ...
    'VariableNames', {'OpenBoll','ClosedBoll', 'Flower', 'Square'});

for ii=1:nfiles
    
    currentfilename = maskFiles(ii).name;
    
    name = split(currentfilename,'_');
    name = join(name(1:end-2),'_');

    currentimage = imread(fullfile(masksFolder,maskFiles(ii).name));
    bw = imbinarize(currentimage);
    
    try
        polygon = mask2poly(bw);
        vertices = {[polygon.X' polygon.Y']};
        
        label = split(currentfilename, '_');
        label = label{end};
        label = split(label, '.');
        label = label{1};
        
        index = strcmp(DataSource, fullfile(imageFolder, strcat(name, '.jpg')));
        
        switch label
            
            case 'OpenBoll'
                temp = LabelData.OpenBoll(index);
                temp{1}(end+1,1) = vertices;
                LabelData.OpenBoll(index) = temp;
                
            case 'ClosedBoll'
                temp = LabelData.ClosedBoll(index);
                temp{1}(end+1,1) = vertices;
                LabelData.ClosedBoll(index) = temp;
                
            case 'Flower'
                temp = LabelData.Flower(index);
                temp{1}(end+1,1) = vertices;
                LabelData.Flower(index) = temp;
            
            case 'Square'
                temp = LabelData.Square(index);
                temp{1}(end+1,1) = vertices;
                LabelData.Square(index) = temp;
                
            otherwise
                fprintf("Unexpected Class Name for %s in index %d \n", name, find(index));
                
        end
        
    end

end
%% Construct GroundTruth

GT = groundTruth(groundTruthDataSource(DataSource), LabelDefs , LabelData);

%save(outputName, 'T');

end