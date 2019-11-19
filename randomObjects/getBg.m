
function [out] = getBg(N)
%UNTITLED Summary of this function goes here
    conc_im = zeros(N,N);
    for i=1:50
        conc_im(randi([50 100]),randi([50 100])) = rand();
    end
    for i=1:20
        conc_im(randi([5 10]),randi([5 10])) = rand();
    end
    conc_im = (real((ifft2(conc_im))));
    conc_im(:,:,:) = conc_im(:,:) - min(min(conc_im(:,:)));
    out = mat2gray(conc_im(:,:));
end

