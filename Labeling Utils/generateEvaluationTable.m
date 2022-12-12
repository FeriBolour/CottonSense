function T = generateEvaluationTable(imageFolder)

imageFiles = dir(fullfile(imageFolder,'*.jpg'));

T = table('Size', [length(imageFiles) 2], ...
    'VariableTypes', {'string','logical'}, ...
    'VariableNames', {'ImageName','Evaluated'});

for n = 1:length(imageFiles)
    T.ImageName(n) = imageFiles(n).name;
end

writetable(T, strcat('EvaluatedImages_',imageFolder,'.xlsx'))

end