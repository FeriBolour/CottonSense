function T = PolygonsFromMasks(imageFolder, numOfImages)

imagefiles = dir(fullfile(imageFolder,'*.png'));
outputName = strcat(imageFolder, '.mat');

nfiles = length(imagefiles);    % Number of files found
%
% currentfilename = fullfile(trainImgFolder,imagefiles(1).name);
% currentimage = imread(currentfilename);
% bw = imbinarize(currentimage);

T = table('Size', [numOfImages 2], ...
    'VariableTypes', {'string','cell'}, ...
    'VariableNames', {'ImageName','Labels'});

imageCounter = 1;
prevName = "";

for ii=1:nfiles
    
    currentfilename = imagefiles(ii).name;
    
    name = split(currentfilename,'_');
    name = join(name(1:end-2),'_');
    
    if strcmp(prevName,name)   % Same Image
        
        annotationCounter = annotationCounter + 1;
        
    else                       % Next Image
        
        if ii ~= 1
            T.Labels(imageCounter) = {T2};
            imageCounter = imageCounter + 1;
        end
        T2 = table('Size', [1 2], ...
            'VariableTypes', {'string','cell'}, ...
            'VariableNames', {'Category','Polygon'});
        T.ImageName(imageCounter) = name;
        annotationCounter = 1;
    end
    
    currentimage = imread(fullfile(imageFolder,imagefiles(ii).name));
    bw = imbinarize(currentimage);
    
    try
        polygon = mask2poly(bw);
        vertices = {[polygon.X' polygon.Y']};
        
        label = split(currentfilename, '_');
        label = label{end};
        label = split(label, '.');
        label = label{1};
        
        if annotationCounter == 1
            T2.Category(1) = label;
            T2.Polygon(1) = vertices;
        else
            T2 = [T2; {label, vertices}];
        end
        
        
    end
    
    if ii == nfiles
        T.Labels(imageCounter) = {T2};
    else
        prevName = name;
    end
    
end

%save(outputName, 'T');

end