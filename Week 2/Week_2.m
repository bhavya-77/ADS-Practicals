% Deinfing minimum function
function min_val = manual_min(region)

    % Initializing with large value
    min_val = inf;
    
    % Iterating through the elements
    for i = 1:numel(region)
        if region(i) < min_val
            min_val = region(i);
        end
    end

end
%%
% Deinfing maximum function
function max_val = manual_max(region)

    % Initializing with small value
    max_val = -inf;
    
    % Iterating through the elements
    for i = 1:numel(region)
        if region(i) > max_val
            max_val = region(i);
        end
    end

end
%%
% Defining padding function
function I_padded = pad_matrix(I, S, pad_value)

    % Getting the size of both matrices
    [I_row, I_col] = size(I);
    [S_row, S_col] = size(S);
    
    % Determining the padding size
    padRow = floor(S_row / 2);
    padCol = floor(S_col / 2);
    
    % Creating the padded matrix and fill it with the padding value
    I_padded = ones(I_row + 2*padRow, I_col + 2*padCol) * pad_value;

    % Placing original matrix in the center of the padded matrix
    I_padded(padRow+1 : end-padRow, padCol+1 : end-padCol) = I;

end
%%
% Defining erosion function
function I_eroded = erosion(I, S)

    % Getting the size of both matrices
    [I_row, I_col] = size(I);
    [S_row, S_col] = size(S);
    
    % Padding the input matrix I (infinity for Erosion)
    I_padded = pad_matrix(I, S, inf);

    % Initializing the output
    I_eroded = zeros(I_row, I_col);

    % Iterating through rows
    for i = 1:I_row
        
        % Iterating through columns
        for j = 1:I_col
            
            % Extracting region corresponding to S
            region = I_padded(i : i+S_row-1, j : j+S_col-1);

            % Considering values under structuring element
            region_values = region(S==1);

            % Computing minimum value for that region
            I_eroded(i, j) = manual_min(region_values);

        end

    end

end
%%
% Defining dilation function
function I_dilated = dilation(I, S)

    % Getting the size of both matrices
    [I_row, I_col] = size(I);
    [S_row, S_col] = size(S);
    
    % Padding the input matrix I (-inf for Dilation)
    I_padded = pad_matrix(I, S, -inf);

    % Initializing the output
    I_dilated = zeros(I_row, I_col);

    % Iterating through rows
    for i = 1:I_row
        
        % Iterating through columns
        for j = 1:I_col
            
            % Extracting region corresponding to S
            region = I_padded(i : i+S_row-1, j : j+S_col-1);

            % Considering values under structuring element
            region_values = region(S==1);

            % Computing maximum value for that region
            I_dilated(i, j) = manual_max(region_values);

        end

    end

end
%%
% Defining opening function
function I_opened = opening(I, S)
    
    % Step 1: Erosion
    I_eroded = erosion(I, S);
    
    % Step 2: Dilation
    I_opened = dilation(I_eroded, S);
    
end
%%
% Defining closing function
function I_closed = closing(I, S)
    
    % Step 1: Dilation
    I_dilated = dilation(I, S);

    % Step 2: Erosion
    I_closed = erosion(I_dilated, S);

end
%%
% Defining white top-hat function
function I_white_tophat = white_tophat(I, S)
    
    % Step 1: Opening
    I_open = opening(I, S);

    % Step 2:
    I_white_tophat = I - I_open;

end
%%
% Defining black top-hat function
function I_black_tophat = black_tophat(I, S)
    
    % Step 1: Closing
    I_close = closing(I, S);

    % Step 2:
    I_black_tophat = I_close - I;

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
% Calculate White Top-Hat Transformation
I_white_tophat = white_tophat(I, S)
%%
% Calculate White Top-Hat Transformation
I_black_tophat = black_tophat(I, S)