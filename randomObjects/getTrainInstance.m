function [supps, supp_masks, queryBg, query_mask_fin] = getTrainInstance()
    %disc
    N=512;
    queryBg = getBg(3, .3);  % get background
    suppBg = getBg(5, .3);
    query_mask_fin = (zeros(N,N,3)==1);

    %uneven polygons
    % polygon
    supps = zeros(5, N, N);
    supp_masks = zeros(5, N, N)==1;
    numl = randi([5 12]);
    size = randi([100 200]);
    x = rand(1,numl)*size;
    y = rand(1,numl)*size;

    x_o = x + rand()*(N - size);
    y_o = y + rand()*(N - size);

    supp_masks(1,:,:) = poly2mask(x_o, y_o, N, N);
    supps(1,:,:)  = supp_masks(1,:,:) * (rand()+0.3) + reshape(suppBg(:,:,1), [1,N,N]);

    for i=2:5
        nx = x*((rand()+0.1)*2) + rand()*(N - size);
        ny = y*((rand()+0.1)*2) + rand()*(N - size);
    %     scal = rand()-0.5;
    %     if scal <= 0
    %         offx = scal*min(nx)/2;
    %     else
    %         offx = scal*(N-max(nx))/2;
    %     end
    %     scal = rand()-0.5;
    %     if scal <= 0
    %         offy = scal*min(ny)/2;
    %     else
    %         offy = scal*(N-max(ny))/2;
    %     end
    %     nx = abs(nx + offx);
    %     ny = abs(ny + offy);

        supp_masks(i,:,:) = poly2mask(nx, ny, N, N);
        supps(i,:,:) = supp_masks(i,:,:) * (rand()/2) + reshape(suppBg(:,:,i), [1,N,N]);
    end

    num_obj = round((abs(randn())));
    if num_obj<=0
        num_obj = 1;
    end

    for i=1:num_obj  % have multiple objects in one image
        nx = x*((rand()+0.1)*2) + rand()*(N - size);
        ny = y*((rand()+0.1)*2) + rand()*(N - size);
    %     scal = rand()-0.5;
    %     if scal <= 0
    %         offx = scal*min(nx)/2;
    %     else
    %         offx = scal*(N-max(nx))/2;
    %     end
    %     scal = rand()-0.5;
    %     if scal <= 0
    %         offy = scal*min(ny)/2;
    %     else
    %         offy = scal*(N-max(ny))/2;
    %     end
    %     nx = abs(nx + offx);
    %     ny = abs(ny + offy);

        query_mask = poly2mask(nx, ny, N, N);
        query = zeros(N,N,3);
        for j=1:3
            query(:,:,j) = query_mask * (rand()/2);
        end
        queryBg  = query + queryBg;
        query_mask_fin = query_mask_fin| + (query_mask + (zeros(N,N,3)==1));
end

