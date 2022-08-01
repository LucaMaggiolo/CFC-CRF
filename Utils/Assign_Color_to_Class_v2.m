function [ Class_RGB ] = Assign_Color_to_Class_v2( label )
%Assign_Color_to_Class
%   This function assigns colors to class in base of a predefined colormap
%   (this apply only for 6 possible classes)
Class_RGB = label;
Class_RGB(:,:,1) = 255*((label==1)+(label==5)+(label==6));
Class_RGB(:,:,2) = 255*(1-(label==2)-(label==6));
Class_RGB(:,:,3) = 255*((label==1)+(label==2)+(label==3));

end

%colormap
%  255 255 255;    % 1  Impervious surfaces
%   0   0  255;    % 2  Buildings 
%   0  255 255;    % 3  Low vegetation
%   0  255  0;     % 4  Tree 
%  255 255  0;     % 5  Car
%  255  0   0;     % 6  Clutter / background

% inverse tranform from RGB groundtruth to class labels
%gt = gt/255;
% class_gt(:,:) = 1*(((gt(:,:,1)==1)+(gt(:,:,2)==1)+(gt(:,:,3)==1))==3)+...
%                 2*(((gt(:,:,1)==0)+(gt(:,:,2)==0)+(gt(:,:,3)==1))==3)+...
%                 3*(((gt(:,:,1)==0)+(gt(:,:,2)==1)+(gt(:,:,3)==1))==3)+...
%                 4*(((gt(:,:,1)==0)+(gt(:,:,2)==1)+(gt(:,:,3)==0))==3)+...
%                 5*(((gt(:,:,1)==1)+(gt(:,:,2)==1)+(gt(:,:,3)==0))==3)+...
%                 6*(((gt(:,:,1)==1)+(gt(:,:,2)==0)+(gt(:,:,3)==0))==3);

