function [] = mae(train)

[M, N] = size(train);

%X = train(:, [2, 4:end-1]);
%X = train(:, [4:end-1]);
X = train(:, 4);
X = [ones(M, 1) X];
y = train(:, end);


% baseline, from sample solution
yhat = train(:, 4);
mae = mean(abs(y - train(:, 4)));
fprintf('baseline MAE : %f\n', mae);

% composite
yhat = train(:, 4);
row_idxs = find(yhat == 0);
for i = 1:length(row_idxs)
    row_idx = row_idxs(i);
    for j = 7:-1:5
        if train(row_idx, j) > 0
            yhat(row_idx) = train(row_idx, j);
            break
        end
    end

    if (yhat(row_idx) > 0)
        continue
    end

    for j = [8, 11, 10, 9]
        if train(row_idx, j) > 0
            yhat(row_idx) = train(row_idx, j);
            break
        end
    end
end

mae = mean(abs(y - yhat));
fprintf('composite MAE : %f\n', mae);

% linear regression
w = (X' * X) \ X' * y;
err = mean(abs(y - X * w));
fprintf('linear regression MAE : %f\n', err);

end
