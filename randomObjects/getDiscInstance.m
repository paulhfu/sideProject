function [img, mask] = getDiscInstance()
%GETDISCINSTANCE Summary of this function goes here
%   Detailed explanation goes here
N = 256;
mask = zeros(N,N);
sz = randi(20);
ub = 60 + sz + randi(10);
lb = 55 + sz + randi(10);
x_off = 35 - randi(70);
y_off = 35 - randi(70);
img = getBg(N);
for x = -N/2:N/2
    for y = -N/2:N/2
        d = sqrt((x_off+x)^2+(y_off+y)^2);
        if d <= ub
            mask(x+(N/2), y+(N/2)) = 1;
            if d >= lb
                img(x+(N/2), y+(N/2)) = img(x+(N/2), y+(N/2))*2;
            else
                a=1;
%                 img(x+(N/2), y+(N/2)) = rand();
            end
        end
    end
end     
img = mat2gray(img);
end

