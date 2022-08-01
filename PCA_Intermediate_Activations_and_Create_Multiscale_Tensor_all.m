
%Compute PCA on intermediate Activations and build the Multiscale Tensor
folder = fileparts(which('ClusterLevel_FullyConnected_CRF')); 

%dataset is either 'Vaihingen' or 'Potsdam'
dataset = 'Vaihingen'; 
%CNN_model is either 'Hypercolumn' or 'Unet'
CNN_model = 'Unet';
cur_dir = pwd;

if strcmp(dataset,'Potsdam')
	%Given the big size of the images, PCA are precomputed and saved before creating the multiscale tensor
	dir = strcat(folder,'\data\Potsdam\');
    rowIdx = [6 6 6 7 7 7];
    colIdx = [7 8 9 7 8 9];	
    if strcmp(CNN_model,'Hypercolumn')
        save_dir = 'Pots_Hyperc_Tensors';
        mkdir(save_dir)
        post_dir = 'outputs\Hyperc_TFgpu_Pots_f2_p2_100epochs.h5\';
        act_dir = 'outputs\Hyperc_TFgpu_Pots_f2_p2_100epochs.h5\activations_L2\';

        %compute PCA
        for num = 2
            img_num = strcat(num2str(rowIdx(num)),'_',num2str(colIdx(num)));
            act_name = strcat(act_dir,'top_potsdam_',img_num,'_ActL2.tif');
            act_tif = Tiff(strcat(dir,act_name),'r');
            act = read(act_tif);
            close(act_tif);

            act_row = reshape(act, [size(act,1)*size(act,2),128]);
            act_row_noBor = reshape(act(5:end-4,5:end-4,:), [(size(act,1)-8)*(size(act,2)-8),128]);
            tic;
            [coeff,score,latent] = pca(act_row_noBor);
            t=toc;
            % Calculate eigenvalues and eigenvectors of the covariance matrix
            covarianceMatrix = cov(act_row);
            [V,D] = eig(covarianceMatrix);

            act_PCA_row = act_row*coeff;
            act2_PCA = reshape(act_PCA_row,[size(act,1),size(act,2),128]);

            L2_PCA = act2_PCA(:,:,1:3);
            cd(save_dir)
            save(strcat('PCA_L2_Hyperc_',img_num,'_ScribGt.mat'),'L2_PCA');
            cd(cur_dir)
        end

        %% Create Multiscale tensor
        for num = 2%:length(colIdx)
            img_num = strcat(num2str(rowIdx(num)),'_',num2str(colIdx(num)));
            %save(strcat('PCA_L2_Hyperc_',img_num,'_ScribGt.mat'),'L2_PCA');
            PCA_L2 = matfile(strcat(save_dir,'/PCA_L2_Hyperc_',img_num,'_ScribGt.mat'));
            act2_PCA = PCA_L2.L2_PCA;
            figure; imshow(act2_PCA);

            IRRG_name = strcat('tiles\top_potsdam_',img_num,'_IRRG.tif');
            ndsm_name = strcat('ndsm\dsm_potsdam_0',num2str(rowIdx(num)),'_0',num2str(colIdx(num)),'_normalized_lastools.jpg');
            IRRG = imread(strcat(dir,IRRG_name));
            ndsm = imread(strcat(dir,ndsm_name));

            res_act2_PCA = imresize(act2_PCA,2);

            Multiscale_Tensor(:,:,5:7) = single(res_act2_PCA);
            %Multiscale_Tensor(:,:,8:10) = single(res_act4_PCA);
            Multiscale_Tensor(:,:,1:3) = IRRG;%_cut;
            Multiscale_Tensor(:,:,4) = ndsm;%_cut;

            cd(save_dir)
            save(strcat('Multiscale_Tensor_Hyperc_',img_num,'_ScribGt.mat'),'Multiscale_Tensor');
            cd(cur_dir)

            name = strcat(post_dir,'top_potsdam_',img_num,'_posteriors.tif');
            posteriors_tif = Tiff(strcat(dir,name),'r');
            posteriors = read(posteriors_tif);
            close(posteriors_tif);

            %max(posteriors(:));
            cd(save_dir)
            %posteriors = posteriors(1:sy,1:sx,:);
            save(strcat('posteriors_img_',img_num,'.mat'),'posteriors');
            cd(cur_dir)
        end
    elseif strcmp(CNN_model,'Unet')
        fprintf('Non data available for this combination.\n')
    else
        fprintf('Non valid CNN model.\nCNN model can be either Unet or Hyercolumn.\n')
    end
elseif strcmp(dataset,'Vaihingen')
    rowIdx = [11 15 28 30];
    dir = strcat(folder,'\data\Vaihingen\');
    if strcmp(CNN_model,'Hypercolumn')
        fprintf('Non data available for this combination.\n')
        return
    elseif strcmp(CNN_model,'Unet')
        %Given the small size of the images, PCA are computed on he fly
        save_dir = 'Vai_Unet_Tensors';
        mkdir(save_dir)
        post_dir = 'outputs\Unet_f2_p2\';
        act_dir = 'outputs\Unet_f2_p2\activations_L2\';
        for num = 4%:length(colIdx)
            img_num = num2str(rowIdx(num));
            act_name = strcat(act_dir,'top_mosaic_09cm_area',img_num,'_ActL2.tif');
            act_tif = Tiff(strcat(dir,act_name),'r');
            act = read(act_tif);
            close(act_tif);

            act_row = reshape(act, [size(act,1)*size(act,2),64]);
            act_row_noBor = reshape(act(5:end-4,5:end-4,:), [(size(act,1)-8)*(size(act,2)-8),64]);
            tic;
            %[coeff,score,latent] = pca(act_row_noBor);
            [coeff,~,~] = pca(act_row_noBor);
            t=toc;
            % Calculate eigenvalues and eigenvectors of the covariance matrix
            covarianceMatrix = cov(act_row);
            [V,D] = eig(covarianceMatrix);

            act_PCA_row = act_row*coeff;
            act2_PCA = reshape(act_PCA_row,[size(act,1),size(act,2),64]);

            %check eigenv
            tot = sum(D(:));
            used_PC = sum(sum(D(end-2:end,end-2:end)));
            ratio = used_PC/tot;

            act_name = strcat(act_dir,'top_mosaic_09cm_area',img_num,'_ActL4.tif');
            act_tif = Tiff(strcat(dir,act_name),'r');
            act = read(act_tif);
            close(act_tif);
            xsy = size(act,1);
            xsx = size(act,2);
            filt = size(act,3);

            %figure; imshow(act(:,:,1:3));
            act_row = reshape(act, [xsy*xsx,filt]);
            act_row_noBor = reshape(act(5:end-4,5:end-4,:), [(xsy-8)*(xsx-8),filt]);
            [coeff,score,latent] = pca(act_row_noBor);

            act_PCA_row = act_row*coeff;
            act4_PCA = reshape(act_PCA_row,[xsy,xsx,filt]);

            res_act2_PCA = imresize(act2_PCA(:,:,1:3),2);
            res_act4_PCA = imresize(act4_PCA(:,:,1:3),4);
            sy = size(res_act4_PCA,1);
            sx = size(res_act4_PCA,2);

            max_double = max(res_act2_PCA(:));
            max_single = max(single(res_act2_PCA(:)));

            IRRG_name = strcat('img\top_mosaic_09cm_area',img_num,'.tif');
            ndsm_name = strcat('ndsm\dsm_09cm_matching_area',img_num,'_normalized.jpg');
            IRRG = imread(strcat(dir,IRRG_name));
            ndsm = imread(strcat(dir,ndsm_name));

            IRRG_cut = IRRG(1:sy,1:sx,:);
            ndsm_cut = ndsm(1:sy,1:sx);

            Multiscale_Tensor(:,:,5:7) = single(res_act2_PCA);
            Multiscale_Tensor(:,:,8:10) = single(res_act4_PCA);
            Multiscale_Tensor(:,:,1:3) = IRRG_cut;
            Multiscale_Tensor(:,:,4) = ndsm_cut;

            cd(save_dir)
            save(strcat('Multiscale_Tensor_Unet_',img_num,'_ScribGt.mat'),'Multiscale_Tensor');
            cd(cur_dir)

            name = strcat(post_dir,'top_mosaic_09cm_area',img_num,'_posteriors.tif');
            posteriors_tif = Tiff(strcat(dir,name),'r');
            posteriors = read(posteriors_tif);
            close(posteriors_tif);

            %max(posteriors(:));
            cd(save_dir)
            posteriors = posteriors(1:sy,1:sx,:);
            save(strcat('posteriors_img_',img_num,'.mat'),'posteriors');
            cd(cur_dir)
        end
    else
        fprintf('Non valid CNN model.\nCNN model can be either Unet or Hyercolumn.\n')
    end
else
    fprintf('Non valid dataset.\nDataset can be either Vaihingen or Potsdam.\n')
end

return;

%% Run Clustering

%rng('shuffle');
rowIdx = [6 6 6 7 7 7];
colIdx = [7 8 9 7 8 9];
num = 5;
img_num = strcat(num2str(rowIdx(num)),'_',num2str(colIdx(num)));
Param.Clust = struct('use_clusters', true, 'cluster_num_global', 256, 'rand_clustering', true,...
                     'cluster_neighb', 4, 'normalize', true, 'spatial_weight', 0.02,...
                     'unary_multiplier', 2800, 'c_scale_num', []);

cur_dir = pwd;
data_dir = 'Pots_Hyperc_Tensors';
cd(data_dir);
matTensor = matfile(strcat('Multiscale_Tensor_Hyperc_',img_num,'_ScribGt.mat'));
matPrediction = matfile(strcat('posteriors_img_',img_num,'.mat'));
cd(cur_dir);
%Use this part to create new kinds of tensors
Multiscale_Tensor = matTensor.Multiscale_Tensor;

dir = 'C:\Users\lucam\OneDrive - unige.it\Documenti\Python\Unet_Cl-FC-CRF\data\Potsdam\';
post_dir = 'outputs\Hyperc_TFgpu_Pots_f2_p2_100epochs.h5\';
name = strcat(post_dir,'top_potsdam_',img_num,'_posteriors.tif');
posteriors_tif = Tiff(strcat(dir,name),'r');
posteriors = read(posteriors_tif);
close(posteriors_tif); %features_full.prediction = matPrediction.prediction_noBorders;
features_full.prediction = matPrediction.posteriors;

fprintf('computing clusters...\n');
[Cluster_Data.Centroids, Cluster_Data.Posterior_Prob, Cluster_Data.Variance] = ...
    Generate_Clusters_Unary_Grid_New( Multiscale_Tensor, features_full.prediction,...
            Param.Clust.cluster_num_global, Param.Clust.rand_clustering, 30);
     
% x_tot = Multiscale_Tensor; % act = features_full.prediction;
% cluster_num = Param.Clust.cluster_num_global; % rand_grid = Param.Clust.rand_clustering;

save_clust = true;
if save_clust
    Clusters = struct('Centroids',Cluster_Data.Centroids,'Posterior_Prob',Cluster_Data.Posterior_Prob,...
        'Variance', Cluster_Data.Variance);
    name_cl = strcat('Clusters_img_',img_num,...
                     sprintf('_scrib_tensor_Hyperc_%icl',...
                             Param.Clust.cluster_num_global));
    cd(data_dir);
    if isfile(name_cl)
        name_cl_new = strcat(name_cl,'_new');
        save(name_cl_new,'Clusters','-v7.3');
    else
        save(name_cl,'Clusters','-v7.3');
    end
    cd(cur_dir);
end 