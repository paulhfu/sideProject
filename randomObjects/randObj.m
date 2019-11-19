clc;close all;clear all

for i=1:20
    [img, mask] = getDiscInstance();
    imwrite(img,strcat('ilastik/img/raw/img',num2str(i),'.jpg'),'JPEG');
    imwrite(mask,strcat('ilastik/img/mask/mask',num2str(i),'.jpg'),'JPEG');
end
% qo_dir = 'query/origin/';
% qg_dir = 'query/groundtruth/';
% so_dir = 'support/origin/';
% sg_dir = 'support/groundtruth/';
% mkdir(qo_dir)
% mkdir(qg_dir)
% mkdir(so_dir)
% mkdir(sg_dir)
% 
% for i=1:800
%     [supps, supp_masks, query, query_mask] = getTrainInstance();
%     imwrite(query,strcat(qo_dir,num2str(i),'.jpg'),'JPEG');
%     imwrite(query_mask,strcat(qg_dir,num2str(i),'.jpg'),'JPEG');
%     
%     o_dir = strcat(so_dir,num2str(i),'/');
%     g_dir = strcat(sg_dir,num2str(i),'/');
%     mkdir(o_dir);
%     mkdir(g_dir);
%     for j=1:5
%         imwrite(squeeze(supps(j,:,:)),strcat(o_dir,num2str(i),num2str(j),'.jpg'),'JPEG');
%         imwrite(squeeze(supp_masks(j,:,:)),strcat(g_dir,num2str(i), num2str(j),'.jpg'),'JPEG');
%     end
%     
%     figure(1)
%     imshow(supp)
%     figure(2)
%     imshow(supp_mask)
%     figure(3)
%     imshow(query)
%     figure(4)
%     imshow(query_mask)
% end
