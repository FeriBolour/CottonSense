function P = mask2poly(mask)
%MASK2POLY Convert mask to region-of-interest polygons.
%   POLY = MASK2POLY(BW) computes region-of-interest polygons from a binary
%   region-of-interest mask, BW. Pixels in BW that are inside the polygon
%   (X,Y) are 1; pixels outside the polygon are 0.  The class of BW is
%   logical. Polygons are returned in POLY, an array of struct with fields:
%
%      Length   - Number of coordinate points defining the polygon
%      X        - X coordinate points of the edge
%      Y        - Y coordinate points of the edge
%      IsFilled - False if representing a hole in a (larger) polygon
%
%   Example:
%      mask = imread('circuit.tif')>100;
%      P = mask2poly(mask);
%      figure;
%      imagesc(mask);
%      axis on; box off; grid on; 
%      hold on;
%      for n = 1:numel(P)
%         if P(n).IsFilled
%            color = 'w';
%         else
%            color = 'y';
%         end
%         plot(P(n).X,P(n).Y,color);
%      end
%      hold off;
%
%   See also POLY2MASK, CONTOURCS.

% Version 1.2.2 (Jun. 30, 2015)
% Written by: Takeshi Ikuma
% Created: Mar. 21, 2014
% Revision History:
%  v.1 (Mar. 21, 2014) - initial release
%  v.1.1 (Apr. 4, 2014) - fixed missing IsFilled field if mask contains no
%                         ROI
%  v.1.2 (Dec. 12, 2014) - fixed error when mask covers the entire area
%                         (caused by missing IsFilled field)
%                        - fixed the bug in the algorithm to determine
%                          IsFilled output
%  v.1.2.1 (May. 1, 2015)  - fixed issue dealing with open mask edges
%  v.1.2.2 (Jun. 30, 2015) - bug fix

narginchk(1,1);
if ~(islogical(mask)&&ismatrix(mask))
   error('BW must be a 2-D matrix of logical values.');
end

% run contour computation algorithm first
P = rmfield(contourcs(double(mask),[0.5 0.5]),'Level');
P(:).IsFilled = [];

% check for open contours (those who touches one or more edges)
flag = arrayfun(@(p)p.X(end)-p.X(1),P)~=0 | arrayfun(@(p)p.Y(end)-p.Y(1),P)~=0;
Pedge = P(flag);
P(flag) = [];

sz = size(mask);

if ~isempty(Pedge) % An edge of at least one polygon is along an edge of the mask
   % marge edges to form a polygon
   N = numel(Pedge);
   
   pairs = zeros(N,2);
   
   corners = [1 1 sz(2) sz(2);1 sz(1) 1 sz(1)];
   
   xedges = [arrayfun(@(p)p.X(1),Pedge) arrayfun(@(p)p.X(end),Pedge)];
   yedges = [arrayfun(@(p)p.Y(1),Pedge) arrayfun(@(p)p.Y(end),Pedge)];
   
   via_corners = cell(N,1);
   for n = 1:N
      % pick the starting edge
      idx = find(pairs==0,1); % I-th edge at the beginning (J=1) or at the end (J=2)
      pairs(idx) = n;

      % set the edge end point as the reference point
      x0 = xedges(idx);
      y0 = yedges(idx);
      xedges(idx) = 0;
      yedges(idx) = 0;
      
      if x0==1 % on the left edge of the mask
         if mask(floor(y0),x0) % mask extends towards the top of the image (cw)
            cornerorder = [1 3 4 2];
         else % mask extends towards the bottom of the image (ccw)
            cornerorder = [2 4 3 1];
         end
      elseif y0==1 % on the top edge of the mask
         if mask(y0,floor(x0)) % mask extends towards the left of the image (ccw)
            cornerorder = [1 2 4 3];
         else % mask extends towards the right of the image (cw)
            cornerorder = [3 4 2 1];
         end
      elseif x0==sz(2) % on the right edge of the mask
         if mask(floor(y0),x0) % mask extends towards the top of the image (cw)
            cornerorder = [3 1 2 4];
         else % mask extends towards the bottom of the image (ccw)
            cornerorder = [4 2 1 3];
         end
      else % on the bottom edge of the mask
         if mask(y0,floor(x0)) % mask extends towards the left of the image (cw)
            cornerorder = [2 1 3 4];
         else % mask extends towards the right of the image (ccw)
            cornerorder = [4 3 1 2];
         end
      end
      
      x = x0;
      y = y0;
      via_corners{n} = zeros(2,6); % first & last columns are the edge line end points
      via_corners{n}(:,1) = [x;y];
      for m = 1:5
         if m<5
            c = corners(:,cornerorder(m));
         else
            c = [x0;y0];
         end
         if c(1)==x % along vertical image edge
            idx = find(xedges==x); % get all ends which end on the same edge
            if ~isempty(idx)
               [ysorted,Isorted] = sort(yedges(idx));
               if c(2)<y % going up
                  I = find(ysorted<y,1,'last');
               else % going down
                  I = find(ysorted>y,1,'first');
               end
               if isempty(I)
                  idx(:) = [];
               else
                  idx = idx(Isorted(I));
               end
            end
         else % along horizontal edge
            idx = find(yedges==y); % get all ends which end on the same edge
            if ~isempty(idx)
               [xsorted,Isorted] = sort(xedges(idx));
               if c(1)<x % going left
                  I = find(xsorted<x,1,'last');
               else % going right
                  I = find(xsorted>x,1,'first');
               end
               if isempty(I)
                  idx(:) = [];
               else
                  idx = idx(Isorted(I));
               end
            end
         end
         
         if isempty(idx) % no mating end found, go around the corner
            via_corners{n}(:,m+1) = c;
            x = c(1);
            y = c(2);
         else % mating edge end found
            break;
         end
      end
      
      if isempty(idx)
         error('No mating end point to an end point of a mask edge found. BUG in mask2poly');
      end
      
      % mark the pair
      pairs(idx) = n;
      via_corners{n}(1,m+1) = xedges(idx);
      via_corners{n}(2,m+1) = yedges(idx);
      xedges(idx) = 0;
      yedges(idx) = 0;
   end
   
   % Combine 
   N = sum(arrayfun(@(p)p.Length,Pedge))+4;
   keep = false(size(Pedge));
   I0 = 1;
   while ~isempty(I0)
      x = zeros(1,N);
      y = zeros(1,N);
      n0 = Pedge(I0).Length;
      idx = 1:n0;
      x(idx) = Pedge(I0).X;
      y(idx) = Pedge(I0).Y;
      J0 = 2;
      I1 = I0;
      while pairs(I1,J0)>0
         % get the pair ID & its mate
         p = pairs(I1,J0); % starting pair
         pairs(I1,J0) = 0;  % clear the pairing info
         [I1,J1] = find(pairs==p,1);
         pairs(I1,J1) = 0;
         
         % if went through any corners, add their corrdinates
         c = via_corners{p};
         Nc = find(c(1,:)>0,1,'last');
         if Nc>2 % mask includes image corners
            n1 = n0 + Nc - 2;
            idx = (n0+1):n1;
            
            if c(1,1)==x(n0) && c(2,1)==y(n0)
               cidx = 2:(Nc-1);
            else
               cidx = (Nc-1):-1:2;
            end
            x(idx) = c(1,cidx);
            y(idx) = c(2,cidx);
            N = N - Nc;
            n0 = n1;
         end
         
         if I0~=I1
            % append the next segment
            Ndata = Pedge(I1).Length;
            n1 = n0 + Ndata;
            idx = (n0+1):n1;
            n0 = n1;
            if J1==2 % need to flip
               x(idx) = fliplr(Pedge(I1).X);
               y(idx) = fliplr(Pedge(I1).Y);
               J0 = 1;
            else
               x(idx) = Pedge(I1).X;
               y(idx) = Pedge(I1).Y;
               J0 = 2;
            end
            N = N - Ndata;
         end
      end
      
      % update the mask edge 
      idx = 1:n0;
      Pedge(I0).Length = n0;
      Pedge(I0).X = x(idx);
      Pedge(I0).Y = y(idx);
      keep(I0) = true;
      
      % look for the next edge
      I0 = find(pairs(:,2)>0,1);
   end

   % combine all polygons
   P = [P;Pedge(keep)];
elseif mask(1,1)
   P(end+1,1) = struct('Length',5,'X',[1 sz(2) sz(2)+0.5 sz(2)+0.5 sz(2) 1 0.5 0.5],...
      'Y',[0.5 0.5 1 sz(1) sz(1)+0.5 sz(1)+0.5 sz(1) 1],'IsFilled',true);
end

% Convert to pixel edge to pixel center
for n = 1:numel(P)
   x = P(n).X;
   y = P(n).Y;
   
   % check for fill/hole condition
   x0 = round(mean(x));
   y0 = round(mean(y));
   N = sum(x>x0&y==y0);
   y1 = y(x==(sz(2)+0.5));
   if numel(y1)>1
      y1(:) = sort(y1);
      N = N + (y0>=y1(1) && y0<=y1(2));
   end
   P(n).IsFilled = xor(mask(y0,x0),mod(N,2)==0);
   
   % Move the polygon to be on the pixel center instead of on its edge
   tf = mod(x,1)~=0; % true if 
   x(:) = floor(x);
   y = floor(y);
   x(x==0) = 1;
   y(y==0) = 1;
   
   I = ~mask(sub2ind(sz,y,x));
   Ix = tf&I;
   x(Ix) = x(Ix) + 1;
   Iy = ~tf&I;
   y(Iy) = y(Iy) + 1;

   % remove repeated coordinates
   I = find(diff(x)==0 & diff(y)==0);
   x(I) = [];
   y(I) = [];
   
   if numel(x)>2
      % assume that contourc produce uniform data point
      dy = diff(y);
      dx = diff(x);
      
      dy2 = diff(dy);
      dx2 = diff(dx);
     
      I = find(dy2==0 & dx2==0)+1;
      x(I) = [];
      y(I) = [];
   end
   
   P(n).X = x;
   P(n).Y = y;
   P(n).Length = numel(x);
end

end
