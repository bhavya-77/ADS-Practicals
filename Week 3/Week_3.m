% Defining padding function
function I_padded = pad_matrix(I, S)

    % Getting the size of the structuring element 
    [S_row, S_col] = size(S);
    
    % Determining the padding size
    padRow = floor(S_row / 2);
    padCol = floor(S_col / 2);
    
    % Initializing the padded matrix with zeros
    [I_row, I_col] = size(I);
    I_padded = zeros(I_row + 2 * padRow, I_col + 2 * padCol);

    % Placing the original matrix in the center
    I_padded(padRow + 1 : padRow + I_row, padCol + 1 : padCol + I_col) = I;

    % Replicating edges
    % Top and Bottom Padding
    I_padded(1:padRow, padCol + 1 : padCol + I_col) = repmat(I(1, :), padRow, 1);
    I_padded(end - padRow + 1:end, padCol + 1 : padCol + I_col) = repmat(I(end, :), padRow, 1);

    % Left and Right Padding
    I_padded(:, 1:padCol) = repmat(I_padded(:, padCol + 1), 1, padCol);
    I_padded(:, end - padCol + 1:end) = repmat(I_padded(:, end - padCol), 1, padCol);

    % Filling corners
    I_padded(1:padRow, 1:padCol) = I_padded(padRow + 1, padCol + 1);
    I_padded(1:padRow, end - padCol + 1:end) = I_padded(padRow + 1, end - padCol);
    I_padded(end - padRow + 1:end, 1:padCol) = I_padded(end - padRow, padCol + 1);
    I_padded(end - padRow + 1:end, end - padCol + 1:end) = I_padded(end - padRow, end - padCol);

end
%%
% Defining mean function
function I_mean = mean_filter(I, S)

    % Getting the size of both matrices
    [I_row, I_col] = size(I);
    [S_row, S_col] = size(S);

    % Padding the input matrix using replicate padding
    I_padded = pad_matrix(I, S);

    % Initializing the output
    I_mean = zeros(I_row, I_col);

    % Sliding operation
    % Iterating through rows
    for i = 1:I_row

        % Iterating through columns
        for j = 1:I_col

            % Extracting region corresponding to S
            region = I_padded(i : i+S_row-1, j : j+S_col-1);

            % Considering values under structuring element
            region_values = region(S==1);

            % Computing mean value for that index
            I_mean(i, j) = sum(region_values)/numel(region);

        end

    end

end
%%
% Defining manual_median function
function med_val = manual_median(region)
    
    % Flattenning and sorting the region
    region_values = sort(region(:));

    n = numel(region_values);

    if mod(n, 2) == 1
        med_val = region_values((n + 1) / 2); % Odd number of elements
    else
        med_val = (region_values(n / 2) + region_values(n / 2 + 1)) / 2; % Even number of elements
    end

end

% Defining median function
function I_median = median_filter(I, S)

    % Getting the size of both matrices
    [I_row, I_col] = size(I);
    [S_row, S_col] = size(S);

    % Padding the input matrix using replicate padding
    I_padded = pad_matrix(I, S);

    % Initializing the output
    I_median = zeros(I_row, I_col);

    % Sliding operation
    % Iterating through rows
    for i = 1:I_row

        % Iterating through columns
        for j = 1:I_col

            % Extracting region corresponding to S
            region = I_padded(i : i+S_row-1, j : j+S_col-1);

            % Computing median value for that index
            I_median(i, j) = manual_median(region);

        end

    end

end
%%
% Defining gaussian function
function I_gaussian = gaussian_filter(I, sigma, n)

    % Creating a Gaussian filter with sigma given
    sigma = sqrt(sigma);
    
    % Creating a grid for the filter
    r = floor(n/2);
    [x, y] = meshgrid(0-r:0+r, 0-r:0+r);
    
    
    % Gaussian filter formula
    gaussian_filter = exp(-(x.^2 + y.^2)/(2*(sigma^2)));
    
    % Normalizing the calculated values
    gaussian_filter = gaussian_filter / sum(gaussian_filter(:));
    
    
    % Convolving matrix I with the Gaussian filter
    I_gaussian = conv2(I, gaussian_filter, 'same');

end
%%
% Defining matrix I
I = [10 20 85 97 55;
    40 60 70 66 52;
    9 70 90 87 12;
    15 54 33 60 11;
    6 26 73 59 9]
%%
% Defining strcturing element
S = [1 1 1;
    1 1 1;
    1 1 1]
%%
% Calculating I_mean
I_mean = mean_filter(I, S)
%%
% Calculating I_median
I_median = median_filter(I, S)
%%
% Calculating I_gaussian
I_gaussian = gaussian_filter(I, 3, 3)
%%
% Working on Images provided
%%
% Defining function to convert image to grayscale
function image_gray = convert_to_grayscale(I)

    % Checking the number of channels in the input image
    if ndims(I) == 3 && size(I, 3) == 3

        % If the image has 3 channels (RGB), convert to grayscale manually
        [rows, cols, ~] = size(I);
        image_gray = zeros(rows, cols); % Initialize grayscale image

        % Looping through each pixel and calculating the grayscale value
        for i = 1:rows
            for j = 1:cols
                R = I(i, j, 1); % Red channel
                G = I(i, j, 2); % Green channel
                B = I(i, j, 3); % Blue channel

                % Calculating grayscale intensity using the luminosity method
                image_gray(i, j) = 0.2989 * R + 0.5870 * G + 0.1140 * B;
            end
        end

        % Converting to uint8 for display
        image_gray = uint8(image_gray);
        disp('Image was RGB and has been converted to Grayscale.');

    else
        % If the image is already grayscale, return it as is
        image_gray = I;
        disp('Image is already Grayscale or single-channel. No conversion needed.');

    end

end
%%
% noise_1
%%
% Defining the URL of the image hosted on GitHub
image_url = 'https://raw.githubusercontent.com/bhavya-77/ADS-Practicals/main/Week%203/noise_1.jpg';

options = weboptions('Timeout', 60);

% Fetch the image from the URL and save it locally
try
    websave('noise_1.jpg', image_url);
    fprintf('Image successfully downloaded');
catch
    error('Failed to fetch the image. Check the URL or internet connection.');
end
%%
% Reading the image
image_1 = imread('noise_1.jpg');

figure;
imshow(image_1);
title('Original Image');

% Converting to grayscale
image_1 = convert_to_grayscale(image_1);

figure;
imshow(uint8(image_1));
title('Grayscale Converted Image');

% Applying Mean filter
I_mean_filtered = mean_filter(image_1, S);

figure;
imshow(uint8(I_mean_filtered));
title('Mean Filtered Image');

% Applying Median filter
I_median_filtered = median_filter(image_1, S);

figure;
imshow(uint8(I_median_filtered));
title('Median Filtered Image');

% Applying Median filter
I_gaussian_filtered = gaussian_filter(image_1, 3, 3);

figure;
imshow(uint8(I_gaussian_filtered));
title('Gaussian Filtered Image');
%%
% noise_2
%%
% Defining the URL of the image hosted on GitHub
image_url = 'https://raw.githubusercontent.com/bhavya-77/ADS-Practicals/main/Week%203/noise_2.jpg';

options = weboptions('Timeout', 60);

% Fetch the image from the URL and save it locally
try
    websave('noise_2.jpg', image_url);
    fprintf('Image successfully downloaded');
catch
    error('Failed to fetch the image. Check the URL or internet connection.');
end
%%
% Reading the image
image_2 = imread('noise_2.jpg');

figure;
imshow(image_2);
title('Original Image');

% Converting to grayscale
image_2 = convert_to_grayscale(image_2);

figure;
imshow(uint8(image_2));
title('Grayscale Converted Image');

% Applying Mean filter
I_mean_filtered = mean_filter(image_2, S);

figure;
imshow(uint8(I_mean_filtered));
title('Mean Filtered Image');

% Applying Median filter
I_median_filtered = median_filter(image_2, S);

figure;
imshow(uint8(I_median_filtered));
title('Median Filtered Image');

% Applying Median filter
I_gaussian_filtered = gaussian_filter(image_2, 3, 3);

figure;
imshow(uint8(I_gaussian_filtered));
title('Gaussian Filtered Image');