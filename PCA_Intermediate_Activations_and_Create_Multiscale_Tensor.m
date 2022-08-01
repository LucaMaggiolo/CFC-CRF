
%Compute PCA on intermediate Activations and build the Multiscale Tensor
folder = fileparts(which('PCA_Intermediate_Activations_and_Create_Multiscale_Tensor')); 

dataset = 'Vaihingen';  %dataset is either 'Vaihingen' or 'Potsdam'
CNN_model = 'Unet';     %CNN_model is either 'Hypercolumn' or 'Unet'
cur_dir = pwd;
%Example data only for the case of Vaihingen dataset with Unet model

if strcmp(dataset,'Potsdam')
	%Given the big size of the images, PCA are precomputed and saved before creating the multiscale tensor
	dir = strcat(folder,'\data\Potsdam\');
    rowIdx = [6 6 6 7 7 7];
    colIdx = [7 8 9 7 8 9];	
    if strcmp(CNN_model,'Hypercolumn')
        fprintf('No example available for this combination.\n')
    elseif strcmp(CNN_model,'Unet')
        fprintf('No example available for this combination.\n')
    else
        fprintf('Non valid CNN model.\nCNN model can be either Unet or Hyercolumn.\n')
    end
elseif strcmp(dataset,'Vaihingen')
    rowIdx = [11 15 28 30];
    dir = strcat(folder,'\data\Vaihingen\');
    if strcmp(CNN_model,'Hypercolumn')
        fprintf('No example available for this combination.\n')
    elseif strcmp(CNN_model,'Unet')
        %Given the small size of the images, PCA are computed on he fly
        save_dir = 'Tensors-Clusters_data\Vai_Unet_Tensors';
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
