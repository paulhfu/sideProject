function [out] = getBg3d()
%UNTITLED Summary of this function goes here
    N = 256;
    conc_im = zeros(N,N,N);
    for j=1:N
        for i=1:50
            conc_im(randi([50 100]),randi([50 100]),randi([50 100])) = rand();
        end
        for i=1:10
            conc_im(randi([5 10]),randi([5 10]),randi([5 10])) = rand();
        end
    end
    conc_im = (real((ifft2(conc_im))));
    conc_im(:,:,:) = conc_im(:,:,j) - min(min(min(conc_im(:,:,j))));
    out = mat2gray(conc_im(:,:,j));
end

