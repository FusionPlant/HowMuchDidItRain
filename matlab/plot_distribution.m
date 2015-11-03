function [] = plot_distribution(data)
% data is M x 1 vector

figure
subplot(2, 2, 1)
hist(data, 100)
title('distribution all expectation')

subplot(2, 2, 2)
hist(data(find(data <= 100)), 100)
title('distribution all expectation under 100')

subplot(2, 2, 3)
hist(data(find(data <= 20)), 100)
title('distribution all expectation under 20')

subplot(2, 2, 4)
hist(data(find(data <= 10)), 100)
title('distribution all expectation under 10')

end
