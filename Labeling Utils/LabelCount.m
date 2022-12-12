function x = LabelCount(gTruth)
%load(filename); %Load GroundTruth here 
%[filepath,name,ext] = fileparts(filename);

numOB = 0; %OpenBoll
for i = 1:height(gTruth.LabelData)
    numOB = numOB + length(gTruth.LabelData.OpenBoll{i,1});
end
disp(['Number of Open Bolls is ', num2str(numOB) ]);

numCB = 0; %ClosedBoll
for i = 1:height(gTruth.LabelData)
    numCB = numCB + length(gTruth.LabelData.ClosedBoll{i,1});
end
disp(['Number of Closed Bolls is ', num2str(numCB) ]);

numF = 0; %Flower
for i = 1:height(gTruth.LabelData)
    numF = numF + length(gTruth.LabelData.Flower{i,1});
end
disp(['Number of Flowers is ', num2str(numF) ]);

numS = 0; %Square
for i = 1:height(gTruth.LabelData)
    numS = numS + length(gTruth.LabelData.Square{i,1});
end
disp(['Number of Squares is ', num2str(numS) ]);

x = categorical({'OpenBolls','ClosedBolls','Flower','Square'});
x = reordercats(x,{'OpenBolls','ClosedBolls','Flower','Square'});
y = [numOB numCB numF numS];
b = bar(x,y);
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom');

end